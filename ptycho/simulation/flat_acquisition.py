"""Deterministic production generation of canonical flat acquisitions.

This adapter owns recipe identity and persistence.  The protected TensorFlow
physics leaf remains in :mod:`ptycho.raw_data`; this module supplies explicit
coordinate/noise streams and projects singleton extraction only for that leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np

from ptycho.config.config import (
    DataConfig,
    ModelConfig,
    SamplingConfig,
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
from ptycho.workflows.synthetic_config import (
    ResolvedSyntheticWorkflow,
    synthetic_workflow_to_dict,
)


STORAGE_LAYOUT = "flat_acquisition_v1"
_SUPPORTED_MEASUREMENT_PAIRS = frozenset(
    {
        ("legacy_v1", "normalized_amplitude"),
        ("ci_intensity_v2", "count_intensity"),
    }
)
COUNT_INTENSITY_DOMAIN = "count_intensity"
MANIFEST_SCHEMA = "flat-acquisition-manifest-v1"
OBJECT_RECIPE = "lines-object-v1"
OBJECT_PRODUCER_SYMBOLS = (
    "ptycho.diffsim.mk_lines_img",
    "ptycho.diffsim.dummy_phi",
)
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
    "object.kind",
    "object.image_size",
    "object.objects_per_probe",
    "object.set_phi",
    "scan.grid_size",
)


@dataclass(frozen=True)
class LinesObject:
    """One locked lines-object array and its stable producer identity."""

    array: np.ndarray
    recipe: str = OBJECT_RECIPE
    producer_symbols: tuple[str, str] = OBJECT_PRODUCER_SYMBOLS


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


def derive_count_amplitude_scale(
    amplitudes: np.ndarray,
    nphotons: float,
) -> float:
    """Derive the count-amplitude scale from normalized amplitudes.

    The scale follows the Torch CI convention exactly:

    ``S = sqrt(nphotons / mean(sum(amplitude**2)))``.

    This NumPy implementation keeps the CUDA-hidden simulation worker free of
    Torch imports; its equality to the Torch helper is pinned by tests.
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
    """Build the exact ``lines-object-v1`` complex truth canvas."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    from ptycho import diffsim

    morphology = diffsim.mk_lines_img(784, nlines=400, rng=rng)
    amplitude = np.asarray(morphology)[196:-196, 196:-196, 0]
    phase = np.asarray(diffsim.dummy_phi(amplitude), dtype=np.float32)
    object_guess = np.asarray(
        amplitude * np.exp(1j * phase), dtype=np.complex64
    )
    if object_guess.shape != (392, 392):
        raise ValueError(
            "lines-object-v1 must produce shape (392, 392), got "
            f"{object_guess.shape}"
        )
    if not np.isfinite(object_guess).all():
        raise ValueError("lines-object-v1 produced nonfinite values")
    return LinesObject(array=np.ascontiguousarray(object_guess))


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

    payload = {
        "diff3d": diffraction,
        "xcoords": xcoords,
        "ycoords": ycoords,
        "probeGuess": probe,
        "objectGuess": truth,
        "scan_index": scan_index,
    }
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


def _simulate_split(
    simulation: SimulationConfig,
    *,
    object_guess: np.ndarray,
    probe_guess: np.ndarray,
    coordinate_seed: int,
    detector_seed: int,
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
        data=DataConfig(
            nphotons=simulation.detector.photons_per_pattern,
        ),
        sampling=SamplingConfig(
            n_groups=simulation.object.diffractions_per_object,
        ),
    )
    raw_data, _ = _generate_simulated_data_legacy(
        config=training,
        simulation=simulation,
        object_guess=object_guess,
        probe_guess=probe_guess,
        buffer=simulation.scan.buffer,
        coordinate_seed=coordinate_seed,
        detector_seed=detector_seed,
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
    probe_lineage: Mapping[str, Any],
    measurement_domain: str,
    scale_contract_version: str,
) -> dict[str, Any]:
    array_hashes, shapes, dtypes = _array_identity(payload)
    measurement_identity = {
        "measurement_domain": measurement_domain,
        "scale_contract_version": scale_contract_version,
        "photons_per_pattern": float(simulation.detector.photons_per_pattern),
    }
    recipe_identity = {
        "split": name,
        "storage_layout": STORAGE_LAYOUT,
        "simulation_config_sha256": simulation_config_sha256(simulation),
        "object_identity": dict(object_identity),
        "raw_probe_sha256": probe_lineage["raw_probe_sha256"],
        "transformed_probe_sha256": probe_lineage["transformed_probe_sha256"],
        "coordinate_seed": seed_lineage[f"{name}_coordinates"],
        "detector_seed": seed_lineage[f"{name}_noise"],
        "measurement_identity": measurement_identity,
    }
    split_recipe_sha256 = canonical_sha256(recipe_identity)
    dataset_identity = {
        "split_recipe_sha256": split_recipe_sha256,
        "array_sha256": array_hashes,
        "shapes": shapes,
        "dtypes": dtypes,
    }
    dataset_sha256 = canonical_sha256(dataset_identity)
    return {
        "artifact_path": path.name,
        "storage_layout": STORAGE_LAYOUT,
        "simulation_config": simulation_config_to_dict(simulation),
        "simulation_config_sha256": simulation_config_sha256(simulation),
        "measurement_identity": measurement_identity,
        "seed_lineage": dict(seed_lineage),
        "coordinate_seed": seed_lineage[f"{name}_coordinates"],
        "detector_seed": seed_lineage[f"{name}_noise"],
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


def _runtime_environment() -> dict[str, Any]:
    import tensorflow as tf

    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_visible_gpu_count": len(tf.config.get_visible_devices("GPU")),
    }


def _validate_locked_flat_recipe(resolved: ResolvedSyntheticWorkflow) -> None:
    simulation_namespace = resolved.simulation
    pair = (
        simulation_namespace.scale_contract_version,
        simulation_namespace.measurement_domain,
    )
    if pair not in _SUPPORTED_MEASUREMENT_PAIRS:
        supported = ", ".join(
            f"{version}/{domain}"
            for version, domain in sorted(_SUPPORTED_MEASUREMENT_PAIRS)
        )
        raise ValueError(
            "simulation.scale_contract_version and "
            "simulation.measurement_domain are one inseparable pair; got "
            f"{pair[0]!r}/{pair[1]!r}, expected one of {supported}"
        )
    for name, simulation in (
        ("train", simulation_namespace.train),
        ("test", simulation_namespace.test),
    ):
        if simulation.object.objects_per_probe != 1:
            raise ValueError(
                f"simulation.{name}.object.objects_per_probe must be exactly 1"
            )
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


def validate_flat_acquisition_workflow(
    resolved: ResolvedSyntheticWorkflow,
) -> None:
    """Validate every flat-acquisition-v1 restriction without doing work."""

    if not isinstance(resolved, ResolvedSyntheticWorkflow):
        raise TypeError("resolved must be a ResolvedSyntheticWorkflow")
    train_simulation = resolved.simulation.train
    test_simulation = resolved.simulation.test
    _validate_shared_split_recipe(train_simulation, test_simulation)
    _validate_locked_flat_recipe(resolved)
    if resolved.simulation.object_recipe != OBJECT_RECIPE:
        raise ValueError(
            f"unsupported object recipe {resolved.simulation.object_recipe!r}; "
            f"expected {OBJECT_RECIPE!r}"
        )
    if not resolved.simulation.shared_object:
        raise ValueError("flat-acquisition v1 requires simulation.shared_object=True")
    if train_simulation.seed is None or test_simulation.seed is None:
        raise ValueError("simulation base seed is required for deterministic generation")
    if train_simulation.seed != test_simulation.seed:
        raise ValueError("train and test SimulationConfig.seed must share one base seed")
    for name, simulation in (
        ("train", train_simulation),
        ("test", test_simulation),
    ):
        if simulation.object.kind != "lines":
            raise ValueError(f"simulation.{name}.object.kind must be 'lines'")
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
    probe_guess: np.ndarray,
    nphotons: float,
    probe_lineage: dict[str, Any],
) -> np.ndarray:
    """Convert normalized amplitudes to the CI count-intensity contract.

    The protected detector leaf draws Poisson counts before dividing by its
    amplitude scale, so the exact post-transform is
    ``counts = (amplitude * S) ** 2``. One ``S`` is derived from the training
    split and reused for every split.

    For flat-acquisition CI data, the stored ``probeGuess`` *is* the CI-scaled
    physical forward probe: ``probe_unscaled * S``. It is not the normalized
    model-input probe. This establishes a deterministic acquisition gauge, not
    an identifiable physical calibration; the persisted dose-closure startup
    solve is the runtime diagnostic for any decomposition mismatch.
    """

    scale = derive_count_amplitude_scale(
        split_payloads["train"]["diff3d"],
        nphotons,
    )
    physical_probe = np.ascontiguousarray(
        np.asarray(probe_guess, dtype=np.complex128) * scale,
        dtype=np.complex64,
    )
    for payload in split_payloads.values():
        counts = np.square(
            np.asarray(payload["diff3d"], dtype=np.float64) * scale
        )
        payload["diff3d"] = _as_finite_array(
            counts,
            name="diff3d",
            dtype=np.dtype(np.float32),
        )
        payload["probeGuess"] = physical_probe
    probe_lineage["physical_probe_sha256"] = array_sha256(physical_probe)
    probe_lineage["count_amplitude_scale"] = {
        "value": scale,
        "split": "train",
        "nphotons": float(nphotons),
        "method": "derive_intensity_scale_from_amplitudes",
    }
    return physical_probe


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
    lines_object = build_lines_object(
        np.random.default_rng(seed_lineage["object"])
    )
    probe_guess, probe_identity = _prepare_probe(train_simulation)
    probe_guess = np.ascontiguousarray(probe_guess, dtype=np.complex64)
    object_hash = array_sha256(lines_object.array)
    object_identity = {
        "recipe": lines_object.recipe,
        "producer_symbols": list(lines_object.producer_symbols),
        "source_commit": _source_commit(),
        "array_sha256": object_hash,
    }
    probe_lineage = dict(probe_identity["probe_lineage"])

    split_payloads: dict[str, dict[str, np.ndarray]] = {}
    for name, simulation in (
        ("train", train_simulation),
        ("test", test_simulation),
    ):
        split_payloads[name] = _simulate_split(
            simulation,
            object_guess=lines_object.array,
            probe_guess=probe_guess,
            coordinate_seed=seed_lineage[f"{name}_coordinates"],
            detector_seed=seed_lineage[f"{name}_noise"],
        )

    stored_probe = probe_guess
    if resolved.simulation.measurement_domain == COUNT_INTENSITY_DOMAIN:
        stored_probe = _apply_count_intensity_contract(
            split_payloads,
            probe_guess=probe_guess,
            nphotons=float(train_simulation.detector.photons_per_pattern),
            probe_lineage=probe_lineage,
        )

    source_payload = {
        "objectGuess": lines_object.array,
        "probeGuess": stored_probe,
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
            object_identity=object_identity,
            probe_lineage=probe_lineage,
            measurement_domain=resolved.simulation.measurement_domain,
            scale_contract_version=resolved.simulation.scale_contract_version,
        )
        for name, simulation in (
            ("train", train_simulation),
            ("test", test_simulation),
        )
    }
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
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
        "seed_lineage": seed_lineage,
        "measurement_identity": {
            "measurement_domain": resolved.simulation.measurement_domain,
            "scale_contract_version": resolved.simulation.scale_contract_version,
        },
        "runtime_environment": _runtime_environment(),
        "object": {
            **object_identity,
            "seed": seed_lineage["object"],
            "shape": list(lines_object.array.shape),
            "dtype": lines_object.array.dtype.name,
        },
        "probe": probe_lineage,
        "splits": split_records,
    }
    _write_json_atomic(manifest_path, manifest)
    return FlatAcquisitionResult(
        dataset_root=dataset_root,
        source_path=source_path,
        train_path=train_path,
        test_path=test_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
