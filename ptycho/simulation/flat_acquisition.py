"""Deterministic production generation of canonical flat acquisitions.

This adapter owns recipe identity and persistence.  The protected TensorFlow
physics leaf remains in :mod:`ptycho.raw_data`; this module supplies explicit
coordinate/noise streams and projects singleton extraction only for that leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np

from ptycho.config.config import (
    ModelConfig,
    SimulationConfig,
    TrainingConfig,
    simulation_config_sha256,
    simulation_config_to_dict,
)
from ptycho.simulation.identity import (
    array_sha256,
    canonical_sha256,
    file_sha256,
)
from ptycho.simulation import object_producers as _object_producers
from ptycho.workflows.synthetic_config import (
    ResolvedSyntheticWorkflow,
    synthetic_workflow_to_dict,
)


STORAGE_LAYOUT = "flat_acquisition_v1"
#: ``docs/specs/spec-ptycho-core.md``: the only supported measurement-units
#: pairs.  Partial or contradictory pairs SHALL error.
_SUPPORTED_MEASUREMENT_PAIRS = frozenset(
    {
        ("legacy_v1", "normalized_amplitude"),
        ("ci_intensity_v2", "count_intensity"),
    }
)
COUNT_INTENSITY_DOMAIN = "count_intensity"
MANIFEST_SCHEMA = "flat-acquisition-manifest-v3"
FROZEN_OBJECT_MANIFEST_SCHEMA = "flat-acquisition-manifest-v4"
TRUTH_FORWARD_CLOSURE_VERSION = "truth-forward-closure-v2"
TRUTH_FORWARD_MIN_RELATIVE_L2_LIMIT = 0.005
TRUTH_FORWARD_NOISE_MULTIPLIER = 3.0
TRUTH_PATCH_MAX_RELATIVE_L2 = 1e-6
TRUTH_FORWARD_SAMPLES_PER_OBJECT = 16
MORPHOLOGY_ATTESTATION_VERSION = "dead-leaves-morphology-attestation-v1"
OBJECT_RECIPE = _object_producers.LINES_OBJECT_RECIPE
OBJECT_PRODUCER_SYMBOLS = _object_producers.LINES_OBJECT_PRODUCER_SYMBOLS
LinesObject = _object_producers.LinesObject
DeadLeavesObject = _object_producers.DeadLeavesObject
SEED_STREAM_NAMES = (
    "object",
    "train_coordinates",
    "train_noise",
    "test_coordinates",
    "test_noise",
    "grouping",
    "torch",
)
_SHARED_SPLIT_RECIPE_FIELDS = (
    "N",
    "probe.source",
    "probe.source_path",
    "probe.transform_pipeline",
    "probe.mask_diameter",
    "probe.ideal_scale",
    "probe.simulation_normalization_scale",
    "object.kind",
    "object.image_size",
    "object.set_phi",
    "object.patch_amplitude_normalization",
    "object.source_path",
    "scan.grid_size",
    "scan.position_layout",
)


@dataclass(frozen=True)
class FlatAcquisitionResult:
    """Paths published by one completed flat-acquisition generation."""

    dataset_root: Path
    source_path: Path
    train_path: Path
    test_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]


def derive_seed_lineage(base_seed: int) -> dict[str, int]:
    """Derive the seven fixed workflow streams from one public base seed."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError("base_seed must be a nonnegative integer")
    base_seed = int(base_seed)
    if base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    children = np.random.SeedSequence(base_seed).spawn(len(SEED_STREAM_NAMES))
    return {
        "base_seed": base_seed,
        **{
            name: int(child.generate_state(1, dtype=np.uint32)[0])
            for name, child in zip(SEED_STREAM_NAMES, children, strict=True)
        },
    }


def raster_scan_positions(
    *,
    n_positions: int,
    height: int,
    width: int,
    buffer: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return span-filling raster scan positions in row-major order.

    ``docs/plans/2026-08-04-synthetic-runner-scan-geometry.md`` §3.2: with
    ``side = sqrt(n_positions)`` and the buffered extent ``[b, extent - b]``,

        pitch_x = (width - 2b) / (side - 1)
        pitch_y = (height - 2b) / (side - 1)
        x[i, j] = b + j * pitch_x
        y[i, j] = b + i * pitch_y

    Coordinates are float64, row-major (``y`` slow, ``x`` fast), unjittered, and
    consume no randomness.  ``buffer`` is clamped exactly as the legacy leaf
    clamps it (``ptycho/nongrid_simulation.py``) so the two agree on the span.
    """

    side = math.isqrt(int(n_positions))
    if side * side != int(n_positions):
        raise ValueError(
            "raster position_layout requires a perfect-square pattern count; "
            f"got {n_positions}, nearest squares are {side ** 2} and "
            f"{(side + 1) ** 2}"
        )
    if side < 2:
        raise ValueError(
            "raster position_layout requires at least 2 positions per axis; "
            f"got a {side}x{side} grid"
        )
    clamped = min(float(buffer), min(int(height), int(width)) / 2 - 1)
    pitch_x = (int(width) - 2 * clamped) / (side - 1)
    pitch_y = (int(height) - 2 * clamped) / (side - 1)
    columns = clamped + pitch_x * np.arange(side, dtype=np.float64)
    rows = clamped + pitch_y * np.arange(side, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(rows, columns, indexing="ij")
    return (
        np.ascontiguousarray(grid_x.reshape(-1)),
        np.ascontiguousarray(grid_y.reshape(-1)),
    )


def fixed_pitch_raster_positions(
    *,
    n_positions: int,
    height: int,
    width: int,
    patch_size: int,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return legacy translation coordinates for a fixed-pitch square raster.

    ``RawData.get_image_patches`` translates an object padded by
    ``patch_size // 2``.  Its first exact source slice therefore has translation
    coordinate ``patch_size // 2`` (while its geometric pixel-center coordinate
    is ``(patch_size - 1) / 2``).  Unlike :func:`raster_scan_positions`, this
    layout never stretches its pitch to fill the object canvas.
    """

    side = math.isqrt(int(n_positions))
    if side * side != int(n_positions):
        raise ValueError(
            "fixed_pitch_raster position_layout requires a perfect-square "
            f"pattern count; got {n_positions}, nearest squares are "
            f"{side ** 2} and {(side + 1) ** 2}"
        )
    if side < 2:
        raise ValueError(
            "fixed_pitch_raster position_layout requires at least 2 "
            f"positions per axis; got a {side}x{side} grid"
        )
    pitch = float(pitch)
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("fixed_pitch_raster pitch must be positive and finite")
    patch_size = int(patch_size)
    height = int(height)
    width = int(width)
    if patch_size <= 0 or height < patch_size or width < patch_size:
        raise ValueError(
            "fixed_pitch_raster patch_size must be positive and fit the canvas"
        )
    origin = float(patch_size // 2)
    last = origin + (side - 1) * pitch
    max_x = width - patch_size + origin
    max_y = height - patch_size + origin
    if last > min(max_x, max_y) + 1e-12:
        raise ValueError(
            "fixed_pitch_raster does not fit the object canvas: "
            f"last center {last} exceeds ({max_x}, {max_y})"
        )
    axis = origin + pitch * np.arange(side, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    return (
        np.ascontiguousarray(grid_x.reshape(-1)),
        np.ascontiguousarray(grid_y.reshape(-1)),
    )


def _apply_patch_amplitude_normalization(
    payload: dict[str, np.ndarray],
    *,
    method: str,
) -> float:
    """Apply one declared split-wide object gauge to truth and diffraction."""

    if method == "none":
        return 1.0
    if method != "mean_patch_max":
        raise ValueError(f"unsupported patch amplitude normalization {method!r}")
    if "Y" not in payload or "diff3d" not in payload:
        raise ValueError("mean_patch_max requires Y and diff3d arrays")
    truth = np.asarray(payload["Y"])
    diffraction = np.asarray(payload["diff3d"])
    if truth.ndim != 3 or diffraction.shape != truth.shape:
        raise ValueError(
            "mean_patch_max requires aligned (M, N, N) Y and diff3d arrays"
        )
    maxima = np.max(np.abs(truth).astype(np.float64, copy=False), axis=(1, 2))
    scale = float(np.mean(maxima, dtype=np.float64))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("mean_patch_max object amplitude scale must be positive")
    payload["Y"] = np.ascontiguousarray(truth / scale, dtype=np.complex64)
    payload["diff3d"] = np.ascontiguousarray(
        diffraction / scale,
        dtype=np.float32,
    )
    payload["object_amplitude_scale"] = np.asarray(scale, dtype=np.float64)
    return scale


def derive_count_amplitude_scale(
    amplitudes: np.ndarray,
    nphotons: float,
) -> float:
    """Derive the dataset count-amplitude scale ``S`` from normalized amplitudes.

    ``S = sqrt(nphotons / mean_over_samples(sum_over_pixels(amplitude**2)))``.

    This is the numpy twin of
    :func:`ptycho_torch.helper.derive_intensity_scale_from_amplitudes`, which is
    the convention the CI training stack uses.  It is reimplemented here so the
    TensorFlow simulation worker never has to import torch; the two are pinned
    equal by ``test_count_amplitude_scale_matches_the_torch_reference_helper``.
    """

    samples = np.asarray(amplitudes, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("amplitudes must have flat shape (M, N, N)")
    if not np.isfinite(samples).all():
        raise ValueError("amplitudes must contain only finite values")
    nphotons = float(nphotons)
    if not np.isfinite(nphotons) or nphotons <= 0:
        raise ValueError("nphotons must be positive and finite")
    mean_intensity = float(np.square(samples).sum(axis=(1, 2)).mean())
    if not np.isfinite(mean_intensity) or mean_intensity <= 0:
        raise ValueError("amplitudes have degenerate energy")
    scale = float(np.sqrt(nphotons / mean_intensity))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("derived count amplitude scale must be positive and finite")
    return scale


def build_lines_object(rng: np.random.Generator) -> LinesObject:
    """Build the registered ``lines-object-v1`` producer."""

    return _object_producers.build_lines_object(rng)


def build_dead_leaves_object(
    rng: np.random.Generator,
    *,
    shape_rng: np.random.Generator,
) -> DeadLeavesObject:
    """Build the registered ``dead-leaves-object-v2`` producer."""

    return _object_producers.build_dead_leaves_object(
        rng,
        shape_rng=shape_rng,
    )


def _as_finite_array(value: Any, *, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def canonicalize_flat_acquisition(
    raw_data: Any,
    *,
    object_guess: np.ndarray,
) -> dict[str, np.ndarray]:
    """Validate and cast one protected-leaf result to the canonical flat schema."""

    xcoords = _as_finite_array(
        raw_data.xcoords, name="xcoords", dtype=np.dtype(np.float64)
    )
    ycoords = _as_finite_array(
        raw_data.ycoords, name="ycoords", dtype=np.dtype(np.float64)
    )
    if xcoords.ndim != 1 or ycoords.ndim != 1 or xcoords.shape != ycoords.shape:
        raise ValueError("xcoords and ycoords must be matching rank-1 arrays")
    sample_count = xcoords.shape[0]

    diffraction = np.asarray(raw_data.diff3d)
    if diffraction.ndim == 4:
        raise ValueError("rank-4 pre-grouped diffraction is not flat acquisition data")
    if diffraction.ndim == 2 and sample_count == 1:
        diffraction = diffraction[None, ...]
    if diffraction.ndim != 3:
        raise ValueError("diff3d must have shape (M, N, N)")
    diffraction = _as_finite_array(
        diffraction, name="diff3d", dtype=np.dtype(np.float32)
    )
    if (
        diffraction.shape[0] != sample_count
        or diffraction.shape[1] != diffraction.shape[2]
    ):
        raise ValueError("diff3d must have shape (M, N, N) matching coordinates")
    if np.any(diffraction < 0):
        raise ValueError("diff3d normalized amplitude must be nonnegative")
    N = diffraction.shape[1]

    probe = np.asarray(raw_data.probeGuess)
    if probe.ndim == 3 and probe.shape == (N, N, 1):
        probe = probe[..., 0]
    if not (
        probe.shape == (N, N)
        or (probe.ndim == 3 and probe.shape[-2:] == (N, N))
    ):
        raise ValueError(
            "probeGuess must have shape (N, N), (N, N, 1), or multimode (P, N, N)"
        )
    probe = _as_finite_array(
        probe, name="probeGuess", dtype=np.dtype(np.complex64)
    )

    truth = _as_finite_array(
        object_guess, name="objectGuess", dtype=np.dtype(np.complex64)
    )
    if truth.ndim != 2:
        raise ValueError("objectGuess must be a rank-2 truth canvas")

    scan_index_value = getattr(raw_data, "scan_index", None)
    if scan_index_value is None:
        scan_index_value = np.zeros(sample_count, dtype=np.int64)
    scan_index = np.ascontiguousarray(
        np.asarray(scan_index_value, dtype=np.int64)
    )
    if scan_index.shape != (sample_count,):
        raise ValueError("scan_index must have shape (M,)")
    object_index_value = getattr(raw_data, "object_index", None)
    if object_index_value is None:
        object_index_value = np.zeros(sample_count, dtype=np.int64)
    object_index = np.ascontiguousarray(
        np.asarray(object_index_value, dtype=np.int64)
    )
    if object_index.shape != (sample_count,):
        raise ValueError("object_index must have shape (M,)")

    payload = {
        "diff3d": diffraction,
        "xcoords": xcoords,
        "ycoords": ycoords,
        "probeGuess": probe,
        "objectGuess": truth,
        "scan_index": scan_index,
        "object_index": object_index,
    }
    truth_patches = getattr(raw_data, "Y", None)
    if truth_patches is not None:
        truth_patches = np.asarray(truth_patches)
        if truth_patches.ndim == 4 and truth_patches.shape[-1] == 1:
            truth_patches = truth_patches[..., 0]
        truth_patches = _as_finite_array(
            truth_patches,
            name="Y",
            dtype=np.dtype(np.complex64),
        )
        if truth_patches.shape != (sample_count, N, N):
            raise ValueError("Y must have shape (M, N, N) matching diffraction")
        payload["Y"] = truth_patches
    for name in ("xcoords_start", "ycoords_start"):
        value = getattr(raw_data, name, None)
        if value is None:
            continue
        coordinate = _as_finite_array(
            value, name=name, dtype=np.dtype(np.float64)
        )
        if coordinate.shape != (sample_count,):
            raise ValueError(f"{name} must have shape (M,)")
        payload[name] = coordinate
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reserve_destination(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    _reserve_destination(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")
    _reserve_destination(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _source_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        raise RuntimeError("cannot resolve source commit for flat acquisition identity")
    return commit


def _array_identity(payload: Mapping[str, np.ndarray]) -> tuple[
    dict[str, str], dict[str, list[int]], dict[str, str]
]:
    hashes = {name: array_sha256(value) for name, value in payload.items()}
    shapes = {name: list(value.shape) for name, value in payload.items()}
    dtypes = {name: value.dtype.name for name, value in payload.items()}
    return hashes, shapes, dtypes


def _prepare_probe(
    simulation: SimulationConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    from scripts.simulation.simulate_and_save import (
        prepare_probe_for_simulation_with_lineage,
    )

    placeholder = np.empty((simulation.N, simulation.N), dtype=np.complex64)
    return prepare_probe_for_simulation_with_lineage(placeholder, simulation)


def normalized_simulation_probe(
    training_probe: np.ndarray,
    normalization_scale: float | None,
) -> tuple[np.ndarray, float]:
    """Derive the acquisition illumination and its realized multiplier."""

    probe = np.ascontiguousarray(training_probe, dtype=np.complex64)
    if probe.ndim != 2 or probe.shape[0] != probe.shape[1]:
        raise ValueError(
            "simulation probe normalization supports one square 2-D probe"
        )
    if normalization_scale is None:
        simulated = probe.copy()
        realized_multiplier = 1.0
    else:
        N = int(probe.shape[0])
        centered = np.arange(N, dtype=np.float64) - N // 2 + 0.5
        xx, yy = np.meshgrid(centered, centered)
        mask = np.sqrt(xx * xx + yy * yy) < N // 4
        norm = float(normalization_scale) * float(
            np.mean(np.abs(mask * probe))
        )
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                "simulation probe normalization requires a finite nonzero "
                "masked probe mean"
            )
        simulated = np.ascontiguousarray(probe / norm, dtype=np.complex64)
        realized_multiplier = 1.0 / norm
    return simulated, realized_multiplier


def _derive_simulation_probe(
    training_probe: np.ndarray,
    simulation: SimulationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the acquisition illumination without mutating the stored probe."""

    probe = np.ascontiguousarray(training_probe, dtype=np.complex64)
    scale = simulation.probe.simulation_normalization_scale
    simulated, realized_multiplier = normalized_simulation_probe(probe, scale)
    return simulated, {
        "training_probe_sha256": array_sha256(probe),
        "simulation_probe_sha256": array_sha256(simulated),
        "simulation_normalization_scale": (
            None if scale is None else float(scale)
        ),
        "simulation_probe_realized_multiplier": realized_multiplier,
    }


def _split_scan_geometry(
    simulation: SimulationConfig,
    *,
    split: str,
) -> dict[str, Any]:
    """Return the realized raster geometry record for one split."""

    n_positions = simulation.object.diffractions_per_object
    height, width = simulation.object.image_size
    side = math.isqrt(int(n_positions))
    if simulation.scan.position_layout == "fixed_pitch_raster":
        outer_offset = (
            simulation.scan.outer_offset_train
            if split == "train"
            else simulation.scan.outer_offset_test
        )
        pitch = float(outer_offset) / 2.0
        origin = float(int(simulation.N) // 2)
        return {
            "side": side,
            "origin_x": origin,
            "origin_y": origin,
            "pitch_x": pitch,
            "pitch_y": pitch,
            "last_x": origin + (side - 1) * pitch,
            "last_y": origin + (side - 1) * pitch,
        }
    clamped = min(
        float(simulation.scan.buffer), min(int(height), int(width)) / 2 - 1
    )
    return {
        "side": side,
        "pitch_x": (int(width) - 2 * clamped) / (side - 1),
        "pitch_y": (int(height) - 2 * clamped) / (side - 1),
        "buffer": clamped,
    }


def _scan_geometry_record(resolved: ResolvedSyntheticWorkflow) -> dict[str, Any]:
    """Record the scan layout and, for raster, the realized per-axis pitch."""

    layout = resolved.simulation.train.scan.position_layout
    record: dict[str, Any] = {"position_layout": layout}
    if layout not in {"raster", "fixed_pitch_raster"}:
        return record
    if layout == "fixed_pitch_raster":
        record["coordinate_frame"] = "legacy_translation"
    for name, simulation in (
        ("train", resolved.simulation.train),
        ("test", resolved.simulation.test),
    ):
        record[name] = _split_scan_geometry(simulation, split=name)
    return record


def _split_coordinates(
    simulation: SimulationConfig,
    *,
    split: str,
    frame_order_recipe: str = "object-major-v1",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return explicit positions when the recipe pins a non-random layout."""

    layout = simulation.scan.position_layout
    if frame_order_recipe not in {
        "object-major-v1",
        "coordinate-major-interleaved-v1",
    }:
        raise ValueError(f"unsupported frame_order_recipe {frame_order_recipe!r}")
    if layout == "uniform_random":
        return None
    height, width = simulation.object.image_size
    if layout == "fixed_pitch_raster":
        outer_offset = (
            simulation.scan.outer_offset_train
            if split == "train"
            else simulation.scan.outer_offset_test
        )
        coordinates = fixed_pitch_raster_positions(
            n_positions=simulation.object.diffractions_per_object,
            height=height,
            width=width,
            patch_size=simulation.N,
            pitch=float(outer_offset) / 2.0,
        )
        return ordered_raster_coordinates(
            coordinates,
            frame_order_recipe=frame_order_recipe,
        )
    if layout != "raster":
        raise ValueError(f"unsupported scan.position_layout {layout!r}")
    coordinates = raster_scan_positions(
        n_positions=simulation.object.diffractions_per_object,
        height=height,
        width=width,
        buffer=simulation.scan.buffer,
    )
    return ordered_raster_coordinates(
        coordinates,
        frame_order_recipe=frame_order_recipe,
    )


def ordered_raster_coordinates(
    coordinates: tuple[np.ndarray, np.ndarray],
    *,
    frame_order_recipe: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the declared row traversal to one square raster."""

    xcoords, ycoords = coordinates
    if frame_order_recipe == "object-major-v1":
        return xcoords, ycoords
    if frame_order_recipe != "coordinate-major-interleaved-v1":
        raise ValueError(f"unsupported frame_order_recipe {frame_order_recipe!r}")
    side = math.isqrt(int(xcoords.size))
    if side * side != int(xcoords.size):
        raise ValueError("coordinate-major frame order requires a square raster")
    return (
        np.ascontiguousarray(xcoords.reshape(side, side).T.reshape(-1)),
        np.ascontiguousarray(ycoords.reshape(side, side).T.reshape(-1)),
    )


def _simulate_split(
    simulation: SimulationConfig,
    *,
    object_guess: np.ndarray,
    probe_guess: np.ndarray,
    coordinate_seed: int,
    detector_seed: int,
    coordinates: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    from scripts.simulation.simulate_and_save import _generate_simulated_data_legacy

    gridsize = simulation.scan.grid_size[0]
    training = TrainingConfig(
        model=ModelConfig(
            N=simulation.N,
            gridsize=gridsize,
            object_big=gridsize > 1,
            probe_big=gridsize > 1,
        ),
        training_groups=simulation.object.diffractions_per_object,
        nphotons=simulation.detector.photons_per_pattern,
    )
    raw_data, _ = _generate_simulated_data_legacy(
        config=training,
        simulation=simulation,
        object_guess=object_guess,
        probe_guess=probe_guess,
        buffer=simulation.scan.buffer,
        coordinate_seed=coordinate_seed,
        detector_seed=detector_seed,
        coordinates=coordinates,
    )
    return canonicalize_flat_acquisition(raw_data, object_guess=object_guess)


def _split_record(
    *,
    name: str,
    path: Path,
    payload: Mapping[str, np.ndarray],
    simulation: SimulationConfig,
    seed_lineage: Mapping[str, int],
    object_identity: Mapping[str, Any],
    object_seed_records: list[Mapping[str, int]],
    probe_lineage: Mapping[str, Any],
    measurement_domain: str,
    scale_contract_version: str,
    frame_order_recipe: str,
    source_backed: bool = False,
) -> dict[str, Any]:
    array_hashes, shapes, dtypes = _array_identity(payload)
    measurement_identity = {
        "measurement_domain": measurement_domain,
        "scale_contract_version": scale_contract_version,
        "photons_per_pattern": float(simulation.detector.photons_per_pattern),
    }
    normalization_record: dict[str, Any] | None = None
    if simulation.object.patch_amplitude_normalization != "none":
        scale_array = np.asarray(payload.get("object_amplitude_scale"))
        if scale_array.shape != ():
            raise ValueError(
                "normalized split requires scalar object_amplitude_scale"
            )
        scale = float(scale_array)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("object_amplitude_scale must be positive and finite")
        normalization_record = {
            "method": simulation.object.patch_amplitude_normalization,
            "scope": "split",
            "scale": scale,
        }
    persisted_seed_lineage = (
        _source_seed_lineage(seed_lineage)
        if source_backed
        else dict(seed_lineage)
    )
    seed_record_name = (
        "acquisition_seed_records" if source_backed else "object_seed_records"
    )
    persisted_seed_records = (
        _acquisition_seed_records(object_seed_records)
        if source_backed
        else [dict(item) for item in object_seed_records]
    )
    recipe_identity = {
        "split": name,
        "storage_layout": STORAGE_LAYOUT,
        "simulation_config_sha256": simulation_config_sha256(simulation),
        "object_identity": dict(object_identity),
        "raw_probe_sha256": probe_lineage["raw_probe_sha256"],
        "transformed_probe_sha256": probe_lineage["transformed_probe_sha256"],
        seed_record_name: persisted_seed_records,
        "coordinate_seed": object_seed_records[0]["coordinate_seed"],
        "detector_seed": object_seed_records[0]["detector_seed"],
        "measurement_identity": measurement_identity,
    }
    if frame_order_recipe != "object-major-v1":
        recipe_identity["frame_order_recipe"] = frame_order_recipe
    if normalization_record is not None:
        recipe_identity["object_amplitude_normalization"] = normalization_record
    split_recipe_sha256 = canonical_sha256(recipe_identity)
    dataset_identity = {
        "split_recipe_sha256": split_recipe_sha256,
        "array_sha256": array_hashes,
        "shapes": shapes,
        "dtypes": dtypes,
    }
    dataset_sha256 = canonical_sha256(dataset_identity)
    record = {
        "artifact_path": path.name,
        "storage_layout": STORAGE_LAYOUT,
        "simulation_config": simulation_config_to_dict(simulation),
        "simulation_config_sha256": simulation_config_sha256(simulation),
        "measurement_identity": measurement_identity,
        "seed_lineage": persisted_seed_lineage,
        seed_record_name: persisted_seed_records,
        "coordinate_seed": object_seed_records[0]["coordinate_seed"],
        "detector_seed": object_seed_records[0]["detector_seed"],
        "array_sha256": array_hashes,
        "shapes": shapes,
        "dtypes": dtypes,
        "split_recipe_identity": recipe_identity,
        "split_recipe_sha256": split_recipe_sha256,
        "dataset_recipe_sha256": split_recipe_sha256,
        "dataset_identity": dataset_identity,
        "dataset_sha256": dataset_sha256,
        "npz_sha256": file_sha256(path),
    }
    if normalization_record is not None:
        record["object_amplitude_normalization"] = normalization_record
    return record


def _runtime_environment() -> dict[str, Any]:
    import tensorflow as tf

    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_visible_gpu_count": len(tf.config.get_visible_devices("GPU")),
    }


def _validate_locked_flat_recipe(resolved: ResolvedSyntheticWorkflow) -> None:
    simulation_namespace = resolved.simulation
    version = simulation_namespace.scale_contract_version
    domain = simulation_namespace.measurement_domain
    if (version, domain) not in _SUPPORTED_MEASUREMENT_PAIRS:
        supported = ", ".join(
            f"{pair[0]}/{pair[1]}" for pair in sorted(_SUPPORTED_MEASUREMENT_PAIRS)
        )
        raise ValueError(
            "simulation.scale_contract_version and "
            "simulation.measurement_domain are one inseparable pair; got "
            f"{version!r}/{domain!r}, expected one of {supported}"
        )
    for name, simulation in (
        ("train", simulation_namespace.train),
        ("test", simulation_namespace.test),
    ):
        if simulation.scan.kind != "nongrid":
            raise ValueError(f"simulation.{name}.scan.kind must be 'nongrid'")
        for field_name, expected in (
            ("offset", 4),
            ("outer_offset_train", 8),
            ("outer_offset_test", 20),
        ):
            observed = getattr(simulation.scan, field_name)
            if observed != expected:
                raise ValueError(
                    f"simulation.{name}.scan.{field_name} must be exactly {expected}"
                )
        if simulation.detector.beamstop_diameter is not None:
            raise ValueError(
                f"simulation.{name}.detector.beamstop_diameter is unsupported "
                "by flat-acquisition v1"
            )


def _nested_field_value(record: object, field_path: str) -> object:
    value = record
    for field_name in field_path.split("."):
        value = getattr(value, field_name)
    return value


def _validate_shared_split_recipe(
    train: SimulationConfig,
    test: SimulationConfig,
) -> None:
    """Reject identity drift for arrays reused from the training recipe."""

    for field_path in _SHARED_SPLIT_RECIPE_FIELDS:
        train_value = _nested_field_value(train, field_path)
        test_value = _nested_field_value(test, field_path)
        if test_value != train_value:
            raise ValueError(
                f"simulation.test.{field_path} must match "
                f"simulation.train.{field_path} because flat-acquisition v1 "
                "reuses the training object and probe; "
                f"got {test_value!r} and {train_value!r}"
            )


def validate_split_manifest_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    split: str,
    split_recipe_sha256: str,
) -> tuple[dict[str, str], dict[str, list[int]], dict[str, str]]:
    """Validate one flat-acquisition split record's array identity against its NPZ.

    Covers the record-level walk shared by the training-stage verifier
    (``synthetic_pipeline._verify_split_artifact``) and the reconstruct-stage
    verifier (``inference._validate_flat_npz``): ``npz_sha256``, the
    ``array_sha256``/``shapes``/``dtypes`` triples, and the sealed
    ``dataset_identity``/``dataset_sha256``. Returns the computed identity so
    callers can run their per-caller residue checks without re-reading the NPZ.
    """
    if record.get("npz_sha256") != file_sha256(path):
        raise ValueError(f"dataset manifest splits.{split}.npz_sha256 mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            hashes: dict[str, str] = {}
            shapes: dict[str, list[int]] = {}
            dtypes: dict[str, str] = {}
            for name in archive.files:
                array = np.asarray(archive[name])
                hashes[name] = array_sha256(array)
                shapes[name] = list(array.shape)
                dtypes[name] = array.dtype.name
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid flat acquisition artifact at {path}: {error}") from error

    for name, computed in (
        ("array_sha256", hashes),
        ("shapes", shapes),
        ("dtypes", dtypes),
    ):
        recorded = record.get(name)
        if not isinstance(recorded, Mapping) or dict(recorded) != computed:
            raise ValueError(f"dataset manifest splits.{split}.{name} mismatch")

    dataset_identity = {
        "split_recipe_sha256": split_recipe_sha256,
        "array_sha256": hashes,
        "shapes": shapes,
        "dtypes": dtypes,
    }
    if record.get("dataset_identity") != dataset_identity:
        raise ValueError(f"dataset manifest splits.{split}.dataset_identity mismatch")
    if record.get("dataset_sha256") != canonical_sha256(dataset_identity):
        raise ValueError(f"dataset manifest splits.{split}.dataset_sha256 mismatch")
    return hashes, shapes, dtypes


def validate_flat_acquisition_workflow(
    resolved: ResolvedSyntheticWorkflow,
) -> None:
    """Validate every flat-acquisition-v1 restriction without doing work."""

    if not isinstance(resolved, ResolvedSyntheticWorkflow):
        raise TypeError("resolved must be a ResolvedSyntheticWorkflow")
    train_simulation = resolved.simulation.train
    test_simulation = resolved.simulation.test
    _validate_shared_split_recipe(train_simulation, test_simulation)
    if (
        resolved.simulation.frame_order_recipe
        == "coordinate-major-interleaved-v1"
        and train_simulation.scan.position_layout
        not in {"raster", "fixed_pitch_raster"}
    ):
        raise ValueError(
            "simulation.frame_order_recipe="
            "'coordinate-major-interleaved-v1' requires a raster "
            "scan.position_layout"
        )
    _validate_locked_flat_recipe(resolved)
    _object_producers.validate_object_recipe(
        train_simulation.object.kind,
        resolved.simulation.object_recipe,
    )
    source_path = train_simulation.object.source_path
    if (
        resolved.simulation.object_recipe
        == _object_producers.FROZEN_OBJECT_BANK_RECIPE
    ):
        if source_path is None:
            raise ValueError(
                "simulation.object.source_path is required for the frozen "
                "object-bank recipe"
            )
    elif source_path is not None:
        raise ValueError(
            "simulation.object.source_path is only supported by the frozen "
            "object-bank recipe"
        )
    if resolved.simulation.shared_object and (
        train_simulation.object.objects_per_probe != 1
        or test_simulation.object.objects_per_probe != 1
    ):
        raise ValueError(
            "simulation.shared_object=True requires exactly one train and test object"
        )
    if (
        test_simulation.object.objects_per_probe != 1
        and any(
            stage in resolved.workflow.stages
            for stage in ("reconstruct", "evaluate")
        )
    ):
        raise ValueError(
            "reconstruction or evaluation currently requires "
            "simulation.test_objects=1"
        )
    if train_simulation.seed is None or test_simulation.seed is None:
        raise ValueError("simulation base seed is required for deterministic generation")
    if train_simulation.seed != test_simulation.seed:
        raise ValueError("train and test SimulationConfig.seed must share one base seed")
    for name, simulation in (
        ("train", train_simulation),
        ("test", test_simulation),
    ):
        if simulation.object.image_size != (392, 392):
            raise ValueError(
                f"simulation.{name}.object.image_size must be (392, 392)"
            )
        if not simulation.object.set_phi:
            raise ValueError(f"simulation.{name}.object.set_phi must be True")
        if (
            simulation.probe.source == "custom"
            and simulation.probe.source_path is None
        ):
            raise ValueError(
                f"simulation.{name}.probe.source_path is required for a custom probe"
            )


def _apply_count_intensity_contract(
    split_payloads: dict[str, dict[str, np.ndarray]],
    *,
    training_probe: np.ndarray,
    simulation_probe: np.ndarray,
    nphotons: float,
    probe_lineage: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert simulated normalized amplitude to physical count intensity.

    ``ptycho.diffsim.illuminate_and_diffract`` draws Poisson photon counts and
    only then divides by its internal ``intensity_scale``, so count intensity is
    an exact algebraic post-transform of the stored normalized amplitude:

        counts = (amplitude * S) ** 2
        probe_physical = amplitude_probe * S

    No additional randomness is consumed, so the seed lineage and the realized
    noise are identical to the amplitude path.

    ``S`` is derived from the TRAIN split alone and reused for the test split,
    so both splits and the stored probe share one absolute gauge
    (``docs/specs/spec-ptycho-core.md``: counts and the calibrated probe must
    scale consistently).  This deliberately departs from
    the public synthetic workflow, which derives a per-split
    scale; a per-split probe would also break the manifest's single shared
    probe digest.
    """

    scale = derive_count_amplitude_scale(split_payloads["train"]["diff3d"], nphotons)
    physical_probe = np.ascontiguousarray(
        np.asarray(training_probe, dtype=np.complex128) * scale,
        dtype=np.complex64,
    )
    physical_simulation_probe = np.ascontiguousarray(
        np.asarray(simulation_probe, dtype=np.complex128) * scale,
        dtype=np.complex64,
    )
    for payload in split_payloads.values():
        counts = np.square(
            np.asarray(payload["diff3d"], dtype=np.float64) * scale
        )
        payload["diff3d"] = _as_finite_array(
            counts, name="diff3d", dtype=np.dtype(np.float32)
        )
        payload["probeGuess"] = physical_probe
        payload["probe_simulated"] = physical_simulation_probe
    probe_lineage["physical_probe_sha256"] = array_sha256(physical_probe)
    probe_lineage["physical_simulation_probe_sha256"] = array_sha256(
        physical_simulation_probe
    )
    probe_lineage["count_amplitude_scale"] = {
        "value": scale,
        "split": "train",
        "nphotons": nphotons,
        "method": "derive_intensity_scale_from_amplitudes",
    }
    return physical_probe, physical_simulation_probe


def _object_bank_seed_records(
    *,
    base_seed: int,
    split: str,
    count: int,
    seed_lineage: Mapping[str, int],
    shared_object: bool,
) -> list[dict[str, int]]:
    """Derive stable per-object streams without moving the legacy 1/1 streams."""

    if split not in {"train", "test"}:
        raise ValueError(f"unknown split {split!r}")
    records: list[dict[str, int]] = []
    for index in range(count):
        if split == "train" and index == 0:
            object_seed = int(seed_lineage["object"])
            coordinate_seed = int(seed_lineage["train_coordinates"])
            detector_seed = int(seed_lineage["train_noise"])
        elif split == "test" and index == 0:
            if shared_object:
                object_seed = int(seed_lineage["object"])
            else:
                object_seed = int(
                    np.random.SeedSequence([base_seed, 2, index, 0]).generate_state(
                        1, dtype=np.uint32
                    )[0]
                )
            coordinate_seed = int(seed_lineage["test_coordinates"])
            detector_seed = int(seed_lineage["test_noise"])
        else:
            split_code = 1 if split == "train" else 2
            children = np.random.SeedSequence(
                [base_seed, split_code, index]
            ).spawn(3)
            object_seed, coordinate_seed, detector_seed = (
                int(child.generate_state(1, dtype=np.uint32)[0])
                for child in children
            )
        records.append(
            {
                "index": index,
                "object_seed": object_seed,
                "coordinate_seed": coordinate_seed,
                "detector_seed": detector_seed,
            }
        )
    return records


def _source_seed_lineage(seed_lineage: Mapping[str, int]) -> dict[str, int]:
    """Retain acquisition/runtime streams without claiming an object seed."""

    return {
        name: int(value)
        for name, value in seed_lineage.items()
        if name != "object"
    }


def _acquisition_seed_records(
    object_seed_records: list[Mapping[str, int]],
) -> list[dict[str, int]]:
    """Project private generation streams to source-backed acquisition lineage."""

    return [
        {
            "index": int(record["index"]),
            "coordinate_seed": int(record["coordinate_seed"]),
            "detector_seed": int(record["detector_seed"]),
        }
        for record in object_seed_records
    ]


def _object_identity_record(
    synthetic_object: Any,
    *,
    split: str,
    index: int,
    seed: int | None,
    source_commit: str,
) -> dict[str, Any]:
    array = np.ascontiguousarray(synthetic_object.array, dtype=np.complex64)
    record = {
        "split": split,
        "index": index,
        "recipe": synthetic_object.recipe,
        "phase_identity": dict(synthetic_object.phase_identity),
        "producer_symbols": list(synthetic_object.producer_symbols),
        "source_commit": source_commit,
        "array_sha256": array_sha256(array),
        "shape": list(array.shape),
        "dtype": array.dtype.name,
    }
    if synthetic_object.recipe == _object_producers.FROZEN_OBJECT_BANK_RECIPE:
        record.update(
            {
                "identity_mode": "source",
                "source_identity": dict(synthetic_object.source_identity),
            }
        )
    else:
        if seed is None:
            raise ValueError("seed-generated object identity requires a seed")
        record.update(
            {
                "seed": seed,
                "rng_identity": dict(synthetic_object.rng_identity),
            }
        )
    return record


def _morphology_descriptor(
    object_array: np.ndarray,
    *,
    origin: int,
    side: int,
) -> dict[str, Any]:
    """Measure a Dead Leaves object on the exact tiled evaluation support."""

    amplitude = np.abs(np.asarray(object_array))[origin : origin + side, origin : origin + side]
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.shape != (side, side) or not np.isfinite(amplitude).all():
        raise ValueError("morphology attestation support is invalid")
    mean = float(np.mean(amplitude))
    standard_deviation = float(np.std(amplitude))
    if mean <= 0.0 or standard_deviation <= 0.0:
        raise ValueError("morphology attestation requires nonconstant positive amplitude")

    total_variation = float(
        (
            np.mean(np.abs(np.diff(amplitude, axis=0)))
            + np.mean(np.abs(np.diff(amplitude, axis=1)))
        )
        / 2.0
    )
    centered = amplitude - mean
    power = np.square(np.abs(np.fft.fft2(centered)))
    frequencies = np.fft.fftfreq(side)
    frequency_y, frequency_x = np.meshgrid(
        frequencies,
        frequencies,
        indexing="ij",
    )
    radius = np.hypot(frequency_y, frequency_x)
    total_power = float(np.sum(power))
    if total_power <= 0.0 or not np.isfinite(total_power):
        raise ValueError("morphology attestation requires nonzero spectral power")

    rounded = np.round(amplitude, decimals=6)
    _levels, counts = np.unique(rounded, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    effective_material_levels = float(
        np.exp(-np.sum(probabilities * np.log(probabilities)))
    )
    return {
        "coefficient_of_variation": standard_deviation / mean,
        "normalized_total_variation": total_variation / standard_deviation,
        "spectral_centroid_cycles_per_pixel": float(
            np.sum(radius * power) / total_power
        ),
        "low_frequency_power_fraction_at_1_over_64": float(
            np.sum(power[(radius > 0.0) & (radius <= 1.0 / 64.0)])
            / total_power
        ),
        "effective_material_levels_round_6_decimals": effective_material_levels,
        "material_level_count_round_6_decimals": int(len(counts)),
        "largest_material_level_fraction_round_6_decimals": float(
            np.max(counts) / counts.sum()
        ),
    }


def _morphology_attestation(
    resolved: ResolvedSyntheticWorkflow,
    object_banks: Mapping[str, list[Any]],
) -> dict[str, Any]:
    """Record model-blind split descriptors; never select objects from metrics."""

    simulation = resolved.simulation.test
    if simulation.object.kind != "dead_leaves":
        return {
            "version": MORPHOLOGY_ATTESTATION_VERSION,
            "applicable": False,
            "reason": "object kind is not dead_leaves",
        }
    if simulation.scan.position_layout != "fixed_pitch_raster":
        return {
            "version": MORPHOLOGY_ATTESTATION_VERSION,
            "applicable": False,
            "reason": "attestation requires fixed_pitch_raster evaluation support",
        }

    tile_size = int(simulation.scan.outer_offset_test) // 2
    raster_side = math.isqrt(int(simulation.object.diffractions_per_object))
    support_side = raster_side * tile_size
    support_origin = int(math.ceil((int(simulation.N) - tile_size) / 2.0))
    object_height, object_width = simulation.object.image_size
    if (
        tile_size <= 0
        or raster_side * raster_side != simulation.object.diffractions_per_object
        or support_origin + support_side > min(object_height, object_width)
    ):
        raise ValueError("morphology attestation support does not fit the object canvas")

    descriptors = {
        split: [
            {
                "object_index": index,
                **_morphology_descriptor(
                    getattr(synthetic_object, "array", synthetic_object),
                    origin=support_origin,
                    side=support_side,
                ),
            }
            for index, synthetic_object in enumerate(object_banks[split])
        ]
        for split in ("train", "test")
    }
    train_cv = float(
        np.mean([item["coefficient_of_variation"] for item in descriptors["train"]])
    )
    train_ntv = float(
        np.mean(
            [item["normalized_total_variation"] for item in descriptors["train"]]
        )
    )
    comparisons = []
    for item in descriptors["test"]:
        cv_excess = max(item["coefficient_of_variation"] / train_cv - 1.0, 0.0)
        ntv_deficit = max(
            1.0 - item["normalized_total_variation"] / train_ntv,
            0.0,
        )
        comparisons.append(
            {
                "test_object_index": item["object_index"],
                "coefficient_of_variation_relative_to_train_mean": (
                    item["coefficient_of_variation"] / train_cv - 1.0
                ),
                "normalized_total_variation_relative_to_train_mean": (
                    item["normalized_total_variation"] / train_ntv - 1.0
                ),
                "joint_cv_ntv_shift_score": min(cv_excess, ntv_deficit),
            }
        )
    return {
        "version": MORPHOLOGY_ATTESTATION_VERSION,
        "applicable": True,
        "role": "provenance_diagnostic_not_quality_gate",
        "selection_policy": "fixed_seed_lineage_without_model_metric_access",
        "support": {
            "basis": "tiled_test_reconstruction",
            "origin": [support_origin, support_origin],
            "shape": [support_side, support_side],
            "tile_size": tile_size,
            "raster_side": raster_side,
        },
        "descriptors": descriptors,
        "comparisons": comparisons,
    }


def _concatenate_object_payloads(
    payloads: list[dict[str, np.ndarray]],
    objects: list[Any],
    *,
    frame_order_recipe: str = "object-major-v1",
) -> dict[str, np.ndarray]:
    """Concatenate frame carriers while retaining a singular canvas only for K=1."""

    frame_fields = (
        "diff3d",
        "xcoords",
        "ycoords",
        "xcoords_start",
        "ycoords_start",
        "Y",
        "scan_index",
        "object_index",
    )
    if frame_order_recipe not in {
        "object-major-v1",
        "coordinate-major-interleaved-v1",
    }:
        raise ValueError(f"unsupported frame_order_recipe {frame_order_recipe!r}")
    combined: dict[str, np.ndarray] = {}
    for name in frame_fields:
        values = [payload[name] for payload in payloads if name in payload]
        if values:
            if len(values) != len(payloads):
                raise ValueError(f"object payloads disagree on optional field {name!r}")
            if frame_order_recipe == "object-major-v1" or len(values) == 1:
                ordered = np.concatenate(values, axis=0)
            else:
                row_count = values[0].shape[0]
                if any(value.shape[0] != row_count for value in values[1:]):
                    raise ValueError(
                        "coordinate-major frame order requires equal rows per object"
                    )
                ordered = np.stack(values, axis=1).reshape(
                    row_count * len(values),
                    *values[0].shape[1:],
                )
            combined[name] = np.ascontiguousarray(ordered)
    if len(objects) == 1:
        combined["objectGuess"] = np.ascontiguousarray(
            objects[0].array,
            dtype=np.complex64,
        )
    return combined


def _expected_patch_amplitude_scale(
    canvases: np.ndarray,
    *,
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    object_index: np.ndarray,
    patch_size: int,
) -> float:
    """Recompute the split-wide mean patch maximum from raw source canvases."""

    banks = np.asarray(canvases)
    xs = np.asarray(xcoords, dtype=np.float64)
    ys = np.asarray(ycoords, dtype=np.float64)
    indices = np.asarray(object_index, dtype=np.int64)
    if banks.ndim != 3 or xs.shape != ys.shape or xs.shape != indices.shape:
        raise ValueError("object normalization inputs have incompatible shapes")
    origin = float(int(patch_size) // 2)
    x_starts_float = xs - origin
    y_starts_float = ys - origin
    x_starts = np.rint(x_starts_float).astype(np.int64)
    y_starts = np.rint(y_starts_float).astype(np.int64)
    if not np.allclose(x_starts_float, x_starts, rtol=0.0, atol=1e-12) or not (
        np.allclose(y_starts_float, y_starts, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "mean_patch_max requires scan centers aligned to whole source patches"
        )
    maxima = np.empty(xs.size, dtype=np.float64)
    for row, (bank, y_start, x_start) in enumerate(
        zip(indices, y_starts, x_starts, strict=True)
    ):
        if bank < 0 or bank >= banks.shape[0]:
            raise ValueError("object_index exceeds the source object bank")
        patch = banks[bank][
            y_start : y_start + patch_size,
            x_start : x_start + patch_size,
        ]
        if patch.shape != (patch_size, patch_size):
            raise ValueError("object normalization patch exceeds its source canvas")
        maxima[row] = float(np.max(np.abs(patch)))
    scale = float(np.mean(maxima, dtype=np.float64))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("expected object amplitude scale must be positive")
    return scale


def _truth_forward_closure(
    split_payloads: Mapping[str, Mapping[str, np.ndarray]],
    *,
    base_seed: int,
    measurement_domain: str,
    photons_per_pattern: Mapping[str, float],
    object_banks: Mapping[str, np.ndarray],
    diffractions_per_object: Mapping[str, int],
    patch_amplitude_normalization: Mapping[str, str],
) -> dict[str, Any]:
    """Check object patches and photon generation at a fixed physical gauge."""

    import tensorflow as tf

    from ptycho.raw_data import get_image_patches, get_relative_coords

    records: list[dict[str, Any]] = []
    for split_code, split in enumerate(("train", "test"), start=1):
        payload = split_payloads[split]
        if "Y" not in payload or "probe_simulated" not in payload:
            raise ValueError(
                f"{split} acquisition requires Y and probe_simulated for closure"
            )
        observed = np.asarray(payload["diff3d"])
        truth = np.asarray(payload["Y"])
        probe = np.asarray(payload["probe_simulated"], dtype=np.complex128)
        object_index = np.asarray(payload["object_index"], dtype=np.int64)
        xcoords = np.asarray(payload["xcoords"], dtype=np.float64)
        ycoords = np.asarray(payload["ycoords"], dtype=np.float64)
        canvases = np.asarray(object_banks[split], dtype=np.complex64)
        if truth.shape != observed.shape:
            raise ValueError(f"{split} Y must match diffraction shape for closure")
        if object_index.shape != (observed.shape[0],):
            raise ValueError(f"{split} object_index must match diffraction rows")
        if canvases.ndim != 3:
            raise ValueError(f"{split} object bank must have shape (K, H, W)")
        photon_count = float(photons_per_pattern[split])
        if not np.isfinite(photon_count) or photon_count <= 0.0:
            raise ValueError(f"{split} photons_per_pattern must be positive")
        noise_reference = float(
            truth.shape[-1] / (2.0 * np.sqrt(photon_count))
        )
        relative_l2_limit = max(
            TRUTH_FORWARD_MIN_RELATIVE_L2_LIMIT,
            TRUTH_FORWARD_NOISE_MULTIPLIER * noise_reference,
        )
        unique_objects = sorted(int(item) for item in np.unique(object_index))
        if unique_objects != list(range(canvases.shape[0])):
            raise ValueError(f"{split} object_index does not cover its object bank")
        expected_rows = int(diffractions_per_object[split])
        if expected_rows <= 0:
            raise ValueError(f"{split} diffractions_per_object must be positive")
        if any(
            int(np.count_nonzero(object_index == bank_index)) != expected_rows
            for bank_index in unique_objects
        ):
            raise ValueError(
                f"{split} object_index row counts disagree with "
                "diffractions_per_object"
            )
        normalization_method = patch_amplitude_normalization[split]
        if normalization_method == "none":
            if "object_amplitude_scale" in payload:
                raise ValueError(
                    f"{split} object_amplitude_scale requires mean_patch_max"
                )
            object_amplitude_scale = 1.0
        elif normalization_method == "mean_patch_max":
            scale_array = np.asarray(payload.get("object_amplitude_scale"))
            if scale_array.shape != ():
                raise ValueError(
                    f"{split} mean_patch_max requires object_amplitude_scale"
                )
            object_amplitude_scale = float(scale_array)
            expected_scale = _expected_patch_amplitude_scale(
                canvases,
                xcoords=xcoords,
                ycoords=ycoords,
                object_index=object_index,
                patch_size=truth.shape[-1],
            )
            if not np.isclose(
                object_amplitude_scale,
                expected_scale,
                rtol=1e-7,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{split} object_amplitude_scale does not match "
                    "mean_patch_max"
                )
        else:
            raise ValueError(
                f"unsupported {split} patch amplitude normalization "
                f"{normalization_method!r}"
            )
        for bank_index in unique_objects:
            candidates = np.flatnonzero(object_index == bank_index)
            count = min(TRUTH_FORWARD_SAMPLES_PER_OBJECT, candidates.size)
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [base_seed, 0xC105, split_code, bank_index]
                )
            )
            selected = np.sort(
                rng.choice(candidates, size=count, replace=False)
            )
            selected_truth = np.asarray(truth[selected], dtype=np.complex128)
            predicted = np.abs(
                np.fft.fftshift(
                    np.fft.fft2(selected_truth * probe[None, ...]),
                    axes=(-2, -1),
                )
            ) / truth.shape[-1]
            observed_amplitude = (
                np.sqrt(np.asarray(observed[selected], dtype=np.float64))
                if measurement_domain == COUNT_INTENSITY_DOMAIN
                else np.asarray(observed[selected], dtype=np.float64)
            )
            denominator = float(np.linalg.norm(observed_amplitude))
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("truth-forward closure observed norm must be positive")
            relative_l2 = float(
                np.linalg.norm(observed_amplitude - predicted) / denominator
            )
            coords = np.zeros((count, 1, 2, 1), dtype=np.float64)
            coords[:, 0, 0, 0] = xcoords[selected]
            coords[:, 0, 1, 0] = ycoords[selected]
            global_offsets, local_offsets = get_relative_coords(coords)
            with tf.device("/CPU:0"):
                expected_truth = np.asarray(
                    get_image_patches(
                        canvases[bank_index],
                        global_offsets,
                        local_offsets,
                        N=truth.shape[-1],
                        gridsize=1,
                    )
                )[..., 0]
            expected_truth = expected_truth / object_amplitude_scale
            truth_denominator = float(np.linalg.norm(selected_truth))
            if not np.isfinite(truth_denominator) or truth_denominator <= 0.0:
                raise ValueError("truth-patch closure norm must be positive")
            truth_patch_relative_l2 = float(
                np.linalg.norm(selected_truth - expected_truth) / truth_denominator
            )
            passed = (
                relative_l2 <= relative_l2_limit
                and truth_patch_relative_l2 <= TRUTH_PATCH_MAX_RELATIVE_L2
            )
            records.append(
                {
                    "split": split,
                    "object_index": bank_index,
                    "sample_indices": [int(item) for item in selected],
                    "sample_count": int(count),
                    "relative_l2": relative_l2,
                    "relative_l2_limit": relative_l2_limit,
                    "poisson_noise_reference": noise_reference,
                    "truth_patch_relative_l2": truth_patch_relative_l2,
                    "truth_patch_relative_l2_limit": TRUTH_PATCH_MAX_RELATIVE_L2,
                    "object_amplitude_scale": object_amplitude_scale,
                    "patch_amplitude_normalization": normalization_method,
                    "passed": passed,
                }
            )
    return {
        "version": TRUTH_FORWARD_CLOSURE_VERSION,
        "sample_policy": "deterministic-random-per-object",
        "samples_per_object": TRUTH_FORWARD_SAMPLES_PER_OBJECT,
        "measurement_domain": measurement_domain,
        "relative_l2_limit_policy": "max(0.005,3*N/(2*sqrt(photons_per_pattern)))",
        "truth_patch_max_relative_l2": TRUTH_PATCH_MAX_RELATIVE_L2,
        "objects": records,
        "passed": bool(records) and all(item["passed"] for item in records),
    }


def generate_flat_acquisitions(
    resolved: ResolvedSyntheticWorkflow,
    output_root: str | Path,
) -> FlatAcquisitionResult:
    """Generate one source array and deterministic train/test flat NPZ files."""

    validate_flat_acquisition_workflow(resolved)
    train_simulation = resolved.simulation.train
    test_simulation = resolved.simulation.test

    semantic = synthetic_workflow_to_dict(resolved)
    dataset_root = Path(output_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    source_path = dataset_root / "source.npz"
    train_path = dataset_root / "train.npz"
    test_path = dataset_root / "test.npz"
    manifest_path = dataset_root / "manifest.json"
    for path in (source_path, train_path, test_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact {path}")

    seed_lineage = derive_seed_lineage(train_simulation.seed)
    split_seed_records = {
        name: _object_bank_seed_records(
            base_seed=train_simulation.seed,
            split=name,
            count=simulation.object.objects_per_probe,
            seed_lineage=seed_lineage,
            shared_object=resolved.simulation.shared_object,
        )
        for name, simulation in (
            ("train", train_simulation),
            ("test", test_simulation),
        )
    }
    source_commit = _source_commit()
    object_banks: dict[str, list[Any]] = {"train": [], "test": []}
    object_source: dict[str, Any] | None = None
    if (
        resolved.simulation.object_recipe
        == _object_producers.FROZEN_OBJECT_BANK_RECIPE
    ):
        source = train_simulation.object.source_path
        if source is None:
            raise ValueError(
                "simulation.object.source_path is required for the frozen "
                "object-bank recipe"
            )
        object_banks, object_source = _object_producers.load_frozen_object_banks(
            train_simulation.object.kind,
            source,
            train_count=train_simulation.object.objects_per_probe,
            test_count=test_simulation.object.objects_per_probe,
            image_size=train_simulation.object.image_size,
            shared_object=resolved.simulation.shared_object,
        )
    elif resolved.simulation.shared_object:
        shared = _object_producers.build_object_from_seed(
            train_simulation.object.kind,
            resolved.simulation.object_recipe,
            seed_lineage["object"],
        )
        object_banks = {"train": [shared], "test": [shared]}
    else:
        for name, simulation in (
            ("train", train_simulation),
            ("test", test_simulation),
        ):
            object_banks[name] = [
                _object_producers.build_object_from_seed(
                    simulation.object.kind,
                    resolved.simulation.object_recipe,
                    record["object_seed"],
                )
                for record in split_seed_records[name]
            ]
    object_identities = {
        name: [
            _object_identity_record(
                synthetic_object,
                split=name,
                index=index,
                seed=(
                    None
                    if object_source is not None
                    else split_seed_records[name][index]["object_seed"]
                ),
                source_commit=source_commit,
            )
            for index, synthetic_object in enumerate(object_banks[name])
        ]
        for name in ("train", "test")
    }
    evaluation_object = object_banks["test"][0]
    object_identity = object_identities["test"][0]
    morphology_attestation = _morphology_attestation(resolved, object_banks)
    probe_guess, probe_identity = _prepare_probe(train_simulation)
    probe_guess = np.ascontiguousarray(probe_guess, dtype=np.complex64)
    probe_simulated, simulation_probe_identity = _derive_simulation_probe(
        probe_guess,
        train_simulation,
    )
    probe_lineage = dict(probe_identity["probe_lineage"])
    probe_lineage.update(simulation_probe_identity)

    split_payloads: dict[str, dict[str, np.ndarray]] = {}
    for name, simulation in (
        ("train", train_simulation),
        ("test", test_simulation),
    ):
        coordinates = _split_coordinates(
            simulation,
            split=name,
            frame_order_recipe=resolved.simulation.frame_order_recipe,
        )
        object_payloads: list[dict[str, np.ndarray]] = []
        for index, synthetic_object in enumerate(object_banks[name]):
            record = split_seed_records[name][index]
            payload = _simulate_split(
                simulation,
                object_guess=synthetic_object.array,
                probe_guess=probe_simulated,
                coordinate_seed=record["coordinate_seed"],
                detector_seed=record["detector_seed"],
                coordinates=coordinates,
            )
            payload["object_index"] = np.full(
                payload["diff3d"].shape[0],
                index,
                dtype=np.int64,
            )
            object_payloads.append(payload)
        split_payloads[name] = _concatenate_object_payloads(
            object_payloads,
            object_banks[name],
            frame_order_recipe=resolved.simulation.frame_order_recipe,
        )
        _apply_patch_amplitude_normalization(
            split_payloads[name],
            method=simulation.object.patch_amplitude_normalization,
        )
        split_payloads[name]["probeGuess"] = probe_guess
        split_payloads[name]["probe_simulated"] = probe_simulated

    stored_probe = probe_guess
    stored_simulation_probe = probe_simulated
    if resolved.simulation.measurement_domain == COUNT_INTENSITY_DOMAIN:
        stored_probe, stored_simulation_probe = _apply_count_intensity_contract(
            split_payloads,
            training_probe=probe_guess,
            simulation_probe=probe_simulated,
            nphotons=float(train_simulation.detector.photons_per_pattern),
            probe_lineage=probe_lineage,
        )

    truth_forward_closure = _truth_forward_closure(
        split_payloads,
        base_seed=train_simulation.seed,
        measurement_domain=resolved.simulation.measurement_domain,
        photons_per_pattern={
            "train": float(train_simulation.detector.photons_per_pattern),
            "test": float(test_simulation.detector.photons_per_pattern),
        },
        object_banks={
            split: np.ascontiguousarray(
                np.stack([item.array for item in bank]),
                dtype=np.complex64,
            )
            for split, bank in object_banks.items()
        },
        diffractions_per_object={
            "train": int(train_simulation.object.diffractions_per_object),
            "test": int(test_simulation.object.diffractions_per_object),
        },
        patch_amplitude_normalization={
            "train": train_simulation.object.patch_amplitude_normalization,
            "test": test_simulation.object.patch_amplitude_normalization,
        },
    )
    if not truth_forward_closure["passed"]:
        raise RuntimeError(
            "truth-forward closure exceeds its relative-L2 limit: "
            f"{truth_forward_closure['objects']}"
        )

    source_payload = {
        "objectGuess": np.ascontiguousarray(
            evaluation_object.array,
            dtype=np.complex64,
        ),
        "trainObjectGuess": np.ascontiguousarray(
            np.stack([item.array for item in object_banks["train"]]),
            dtype=np.complex64,
        ),
        "testObjectGuess": np.ascontiguousarray(
            np.stack([item.array for item in object_banks["test"]]),
            dtype=np.complex64,
        ),
        "probeGuess": stored_probe,
        "probe_simulated": stored_simulation_probe,
    }
    _write_npz_atomic(source_path, source_payload)
    _write_npz_atomic(train_path, split_payloads["train"])
    _write_npz_atomic(test_path, split_payloads["test"])

    split_records = {
        name: _split_record(
            name=name,
            path=train_path if name == "train" else test_path,
            payload=split_payloads[name],
            simulation=simulation,
            seed_lineage=seed_lineage,
            object_identity=(
                {
                    key: object_identities[name][0][key]
                    for key in (
                        (
                            "recipe",
                            "identity_mode",
                            "source_identity",
                            "phase_identity",
                            "producer_symbols",
                            "source_commit",
                            "array_sha256",
                        )
                        if object_source is not None
                        else (
                            "recipe",
                            "rng_identity",
                            "phase_identity",
                            "producer_symbols",
                            "source_commit",
                            "array_sha256",
                        )
                    )
                }
                if len(object_identities[name]) == 1
                else {"objects": object_identities[name]}
            ),
            object_seed_records=split_seed_records[name],
            probe_lineage=probe_lineage,
            measurement_domain=resolved.simulation.measurement_domain,
            scale_contract_version=resolved.simulation.scale_contract_version,
            frame_order_recipe=resolved.simulation.frame_order_recipe,
            source_backed=object_source is not None,
        )
        for name, simulation in (
            ("train", train_simulation),
            ("test", test_simulation),
        )
    }
    manifest: dict[str, Any] = {
        "schema_version": (
            FROZEN_OBJECT_MANIFEST_SCHEMA
            if object_source is not None
            else MANIFEST_SCHEMA
        ),
        "storage_layout": STORAGE_LAYOUT,
        "profile": resolved.profile,
        "recipe_version": resolved.recipe_version,
        "artifacts": {
            "source": source_path.name,
            "train": train_path.name,
            "test": test_path.name,
        },
        "source_npz_sha256": file_sha256(source_path),
        "simulation": semantic["simulation"],
        "seed_lineage": (
            _source_seed_lineage(seed_lineage)
            if object_source is not None
            else seed_lineage
        ),
        "measurement_identity": {
            "measurement_domain": resolved.simulation.measurement_domain,
            "scale_contract_version": resolved.simulation.scale_contract_version,
        },
        "scan_geometry": _scan_geometry_record(resolved),
        "runtime_environment": _runtime_environment(),
        "object": {
            **object_identity,
        },
        "objects": object_identities,
        "morphology_attestation": morphology_attestation,
        "probe": probe_lineage,
        "truth_forward_closure": truth_forward_closure,
        "splits": split_records,
    }
    if object_source is not None:
        manifest["object_source"] = object_source
    _write_json_atomic(manifest_path, manifest)
    return FlatAcquisitionResult(
        dataset_root=dataset_root,
        source_path=source_path,
        train_path=train_path,
        test_path=test_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
