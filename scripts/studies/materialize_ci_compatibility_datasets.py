"""Materialize immutable Dead Leaves and canonical-lines compatibility twins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _entry in (str(REPO), str(REPO / "scripts/studies")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import ci_compat_materializer_lib as lib  # noqa: E402
import make_synthetic_truth_datasets as M  # noqa: E402

from ptycho import diffsim  # noqa: E402
from ptycho_torch.datagen.objects import create_dead_leaves  # noqa: E402
from scripts.studies.ablation.dataset_provenance import (  # noqa: E402
    CI_COMPATIBILITY_PROVENANCE_V3,
    canonical_array_sha256,
)
from scripts.studies.ablation.datasets import (  # noqa: E402
    DatasetError,
    canonicalize_probe,
    load_checked_dataset_bundle,
    probe_l2_norm,
)

MaterializationError = lib.MaterializationError
MaterializationSpec = lib.MaterializationSpec

TARGET_PHOTONS_PER_IMAGE = 3_538_944.0
FAMILIES = ("deadleaves", "lines")
DATASET_IDS = tuple(
    sorted(
        f"{family}_{profile}"
        for family in FAMILIES
        for profile in ("ci_3p5m", "legacy_amp")
    )
)
PROVENANCE_FILENAME = "ci_compatibility_provenance.json"
DESCRIPTORS_FILENAME = "ci_compatibility_descriptors.json"
MATERIALIZER_ID = "ci_compatibility_twins_v3"
GROUPING_MAX_C = 4
GENERATION_BATCH_SIZE = 128
_NPZ_KEY_ORDER = (
    "xcoords",
    "ycoords",
    "xcoords_start",
    "ycoords_start",
    "diff3d",
    "probeGuess",
    "objectGuess",
    "scan_index",
    "ground_truth_patches",
    "probeGeometry",
    "_metadata",
)


@dataclass(frozen=True)
class SplitArtifact:
    dataset_id: str
    split: str
    path: str
    file_sha256: str
    probe_statistics: dict[str, Any]
    dose: dict[str, int | float] | None
    provenance: dict[str, Any]


def v2_array_sha256(array: np.ndarray, dtype: Any | None = None) -> str:
    return canonical_array_sha256(array, dtype)


def _target_mean_count(spec: MaterializationSpec) -> float:
    return TARGET_PHOTONS_PER_IMAGE / float(spec.detector_size**2)


def _materialization_profile(spec: MaterializationSpec) -> str:
    return (
        "claim_grade"
        if (
            spec.detector_size == 64
            and spec.object_resolution == 320
            and spec.train_positions == 5000
            and spec.test_positions == 1250
        )
        else "fixture"
    )


def _measurement_seeds(spec: MaterializationSpec) -> dict[str, dict[str, int]]:
    return {
        dataset_id: {
            "train": spec.measurement_seed + 2 * index + 1,
            "test": spec.measurement_seed + 2 * index + 2,
        }
        for index, dataset_id in enumerate(DATASET_IDS)
    }


def _deadleaves_object(spec: MaterializationSpec) -> np.ndarray:
    raw_object = create_dead_leaves(
        (spec.object_resolution, spec.object_resolution),
        M.DEAD_LEAVES_ARG,
        rng=np.random.default_rng(spec.object_seed),
    ).astype(np.complex64)
    return np.ascontiguousarray(
        M.compress_phase(raw_object, M.PHASE_MAX),
        dtype=np.complex64,
    )


def _lines_object(spec: MaterializationSpec) -> np.ndarray:
    crop_start = spec.object_resolution // 2
    crop_stop = crop_start + spec.object_resolution
    morphology = diffsim.mk_lines_img(
        N=2 * spec.object_resolution,
        nlines=400,
        rng=np.random.RandomState(spec.object_seed),
    )[crop_start:crop_stop, crop_start:crop_stop, 0]
    if not np.isfinite(morphology).all():
        raise MaterializationError("lines morphology must contain only finite values")
    morphology_min = float(morphology.min())
    morphology_max = float(morphology.max())
    if morphology_max == morphology_min:
        raise MaterializationError("lines morphology must not be constant")
    if morphology_min < 0.0 < morphology_max:
        scale = max(abs(morphology_min), abs(morphology_max))
        scaled = morphology / scale
        scaled_min = morphology_min / scale
        normalized = (scaled - scaled_min) / (morphology_max / scale - scaled_min)
    else:
        normalized = (morphology - morphology_min) / (morphology_max - morphology_min)
    if not np.isfinite(normalized).all():
        raise MaterializationError("lines morphology normalization must be finite")
    if float(normalized.min()) < 0.0 or float(normalized.max()) > 1.0:
        raise MaterializationError("lines morphology normalization must be in [0, 1]")
    amplitude = 0.3 + 0.7 * normalized
    phase = 0.5 * (2.0 * normalized - 1.0)
    obj = np.ascontiguousarray(amplitude * np.exp(1j * phase), dtype=np.complex64)
    if not np.isfinite(obj).all():
        raise MaterializationError("bounded lines object must be finite")
    return obj


def _coordinates(spec: MaterializationSpec) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for split, count, seed in (
        ("train", spec.train_positions, spec.train_coordinate_seed),
        ("test", spec.test_positions, spec.test_coordinate_seed),
    ):
        xcoords, ycoords = M.scan_positions(
            spec.object_resolution,
            spec.detector_size,
            count,
            seed,
            spec.scan_jitter,
        )
        result[split] = (
            np.ascontiguousarray(xcoords, dtype=np.float32),
            np.ascontiguousarray(ycoords, dtype=np.float32),
        )
    return result


def _batches(count: int) -> list[slice]:
    return [
        slice(start, min(start + GENERATION_BATCH_SIZE, count))
        for start in range(0, count, GENERATION_BATCH_SIZE)
    ]


def _write_deterministic_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
    scratch_root: Path,
) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key in _NPZ_KEY_ORDER:
            npy_path = scratch_root / f"{key}.npy"
            with npy_path.open("wb") as handle:
                np.lib.format.write_array(
                    handle,
                    np.asarray(arrays[key]),
                    allow_pickle=False,
                )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            with (
                npy_path.open("rb") as source,
                archive.open(info, "w", force_zip64=True) as destination,
            ):
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            npy_path.unlink()
    _fsync_file(path)


def _split_metadata(
    *,
    is_count: bool,
    target_mean_count: float,
    dose_intensity_scale: float,
    dose_amplitude_scale: float,
    poisson_seed: int,
) -> dict[str, Any]:
    if is_count:
        return {
            "schema_version": "1.0.0",
            "scale_contract_version": "ci_intensity_v2",
            "measurement_domain": "count_intensity",
            "probe_gauge": "physical_calibrated",
            "object_units": M.ABSOLUTE_OBJECT_UNITS,
            "probe_calibration": {
                "status": "calibrated",
                "method": "raw_object_probe_forward_to_requested_mean_count",
                "target_mean_count": float(target_mean_count),
                "dose_intensity_scale": dose_intensity_scale,
                "dose_amplitude_scale": dose_amplitude_scale,
            },
            "poisson_sampling": {"status": "fresh", "seed": int(poisson_seed)},
        }
    return {
        "schema_version": "1.0.0",
        "scale_contract_version": "legacy_v1",
        "measurement_domain": "normalized_amplitude",
        "probe_gauge": "legacy_normalized",
        "object_units": M.ABSOLUTE_OBJECT_UNITS,
        "normalization": {
            "method": "sqrt_fresh_poisson_counts_over_dose_amplitude_scale",
            "target_mean_count": float(target_mean_count),
            "dose_amplitude_scale": dose_amplitude_scale,
            "poisson_seed": int(poisson_seed),
        },
    }


def _write_dataset_split(
    *,
    spec: MaterializationSpec,
    dataset_id: str,
    split: str,
    obj: np.ndarray,
    raw_geometry: np.ndarray,
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    poisson_seed: int,
    staging_root: Path,
) -> SplitArtifact:
    is_count = dataset_id.endswith("ci_3p5m")
    count = len(xcoords)
    target_mean_count = _target_mean_count(spec)
    output_name = f"{dataset_id}_{split}.npz"
    output_path = staging_root / output_name
    with tempfile.TemporaryDirectory(
        prefix=f".{dataset_id}-{split}-", dir=staging_root
    ) as scratch_name:
        scratch_root = Path(scratch_name)
        patches = np.lib.format.open_memmap(
            scratch_root / "patches.dat",
            mode="w+",
            dtype=np.complex64,
            shape=(count, spec.detector_size, spec.detector_size),
        )
        raw_intensity_sum = 0.0
        raw_intensity_size = 0
        for batch in _batches(count):
            patch_batch = M.extract_object_patches(
                obj, xcoords[batch], ycoords[batch], spec.detector_size
            )
            patches[batch] = patch_batch
            expected = M.noiseless_detector_intensity(patch_batch, raw_geometry)
            raw_intensity_sum += float(expected.sum(dtype=np.float64))
            raw_intensity_size += expected.size
            del patch_batch, expected
        raw_mean = raw_intensity_sum / raw_intensity_size
        if not np.isfinite(raw_mean) or raw_mean <= 0:
            raise MaterializationError("raw object/probe forward mean must be positive")
        dose_intensity_scale = float(target_mean_count / raw_mean)
        dose_amplitude_scale = float(np.sqrt(dose_intensity_scale))
        physical_probe = np.asarray(
            raw_geometry * np.float32(dose_amplitude_scale), dtype=np.complex64
        )
        measurement_dtype = np.uint16 if is_count else np.float32
        measurement = np.lib.format.open_memmap(
            scratch_root / "measurement.dat",
            mode="w+",
            dtype=measurement_dtype,
            shape=(count, spec.detector_size, spec.detector_size),
        )
        rng = np.random.default_rng(poisson_seed)
        for batch in _batches(count):
            expected = M.noiseless_detector_intensity(patches[batch], physical_probe)
            counts_wide = rng.poisson(expected)
            max_count = int(counts_wide.max(initial=0))
            if max_count > np.iinfo(np.uint16).max:
                raise MaterializationError(
                    f"fresh Poisson count {max_count} exceeds uint16 range"
                )
            if is_count:
                measurement[batch] = counts_wide.astype(np.uint16)
            else:
                measurement[batch] = (
                    np.sqrt(counts_wide.astype(np.float64)) / dose_amplitude_scale
                ).astype(np.float32)
            del expected, counts_wide
        patches.flush()
        measurement.flush()
        stored_probe = physical_probe if is_count else raw_geometry
        metadata = _split_metadata(
            is_count=is_count,
            target_mean_count=target_mean_count,
            dose_intensity_scale=dose_intensity_scale,
            dose_amplitude_scale=dose_amplitude_scale,
            poisson_seed=poisson_seed,
        )
        arrays = {
            "xcoords": xcoords,
            "ycoords": ycoords,
            "xcoords_start": xcoords.copy(),
            "ycoords_start": ycoords.copy(),
            "diff3d": measurement,
            "probeGuess": stored_probe,
            "objectGuess": obj,
            "scan_index": np.zeros(count, dtype=np.int64),
            "ground_truth_patches": patches[..., None],
            "probeGeometry": raw_geometry,
            "_metadata": np.array(json.dumps(metadata, sort_keys=True)),
        }
        _write_deterministic_npz(output_path, arrays, scratch_root)
        probe_stats = lib.probe_statistics(stored_probe)
        dose = _dose_statistics(measurement) if is_count else None
        raw_norm = probe_l2_norm(raw_geometry)
        stored_norm = probe_l2_norm(stored_probe)
        provenance = {
            "path": output_name,
            "file_sha256": _file_sha256(output_path),
            "truth_sha256": v2_array_sha256(obj, np.complex64),
            "xcoords_sha256": v2_array_sha256(xcoords, np.float32),
            "ycoords_sha256": v2_array_sha256(ycoords, np.float32),
            "raw_probe_sha256": v2_array_sha256(raw_geometry, np.complex64),
            "stored_probe_sha256": v2_array_sha256(stored_probe, np.complex64),
            "probe_scale": float(stored_norm / raw_norm),
            "stored_probe_l2_norm": float(stored_norm),
        }
        if dose is not None:
            provenance["dose"] = dose
        del arrays, measurement, patches
    return SplitArtifact(
        dataset_id=dataset_id,
        split=split,
        path=output_name,
        file_sha256=provenance["file_sha256"],
        probe_statistics=probe_stats,
        dose=dose,
        provenance=provenance,
    )


def _stream_dataset_files(
    spec: MaterializationSpec,
    objects: dict[str, np.ndarray],
    raw_geometry: np.ndarray,
    coordinates: dict[str, tuple[np.ndarray, np.ndarray]],
    seeds: dict[str, dict[str, int]],
    staging_root: Path,
) -> dict[tuple[str, str], SplitArtifact]:
    artifacts = {}
    for family in FAMILIES:
        obj = objects[family]
        for split, (xcoords, ycoords) in coordinates.items():
            for profile in ("ci_3p5m", "legacy_amp"):
                dataset_id = f"{family}_{profile}"
                artifact = _write_dataset_split(
                    spec=spec,
                    dataset_id=dataset_id,
                    split=split,
                    obj=obj,
                    raw_geometry=raw_geometry,
                    xcoords=xcoords,
                    ycoords=ycoords,
                    poisson_seed=seeds[dataset_id][split],
                    staging_root=staging_root,
                )
                artifacts[(dataset_id, split)] = artifact
    return artifacts


def _dose_statistics(measurement: np.ndarray) -> dict[str, int | float]:
    stats = lib.dose_statistics(measurement)
    if stats["photons_per_image_min"] < 1_000_000:
        raise MaterializationError(
            "CI compatibility split violates the one-million-photon weakest-frame floor"
        )
    if (
        stats["max_observed_count"] >= stats["dtype_max"]
        or stats["saturation_fraction"] != 0.0
    ):
        raise MaterializationError("CI compatibility split contains saturated counts")
    return stats


def _descriptor_common(spec: MaterializationSpec, probe_modes: int) -> dict[str, Any]:
    return {
        "kind": "synthetic",
        "format": "npz_mmap",
        "truth": "object_truth",
        "truth_location": "embedded_test",
        "truth_key": "objectGuess",
        "measurement_key": "diff3d",
        "probe_key": "probeGuess",
        "x_key": "xcoords",
        "y_key": "ycoords",
        "coords_convention": "xy_pixels",
        "detector_shape": [spec.detector_size, spec.detector_size],
        "grouping_max_C": GROUPING_MAX_C,
        "probe_modes": probe_modes,
        "provenance": PROVENANCE_FILENAME,
    }


def _build_descriptors(
    spec: MaterializationSpec,
    artifacts: dict[tuple[str, str], SplitArtifact],
    raw_geometry: np.ndarray,
) -> dict[str, dict[str, Any]]:
    common = _descriptor_common(spec, int(raw_geometry.shape[0]))
    descriptors = {}
    for dataset_id in DATASET_IDS:
        is_count = dataset_id.endswith("ci_3p5m")
        split_artifacts = {
            split: artifacts[(dataset_id, split)] for split in ("train", "test")
        }
        descriptor = dict(common)
        descriptor.update(
            scale_contract_version="ci_intensity_v2" if is_count else "legacy_v1",
            measurement_domain="count_intensity"
            if is_count
            else "normalized_amplitude",
            train=split_artifacts["train"].path,
            test=split_artifacts["test"].path,
            train_sha256=split_artifacts["train"].file_sha256,
            test_sha256=split_artifacts["test"].file_sha256,
            probe={
                "source": spec.probe_source,
                "calibration": "count_amplitude" if is_count else "legacy_normalized",
                "gauge": "physical_count_amplitude"
                if is_count
                else "legacy_normalized",
                "mask_policy": "model_config",
                **(
                    {
                        "train_sha256": split_artifacts["train"].probe_statistics[
                            "sha256"
                        ],
                        "test_sha256": split_artifacts["test"].probe_statistics[
                            "sha256"
                        ],
                    }
                    if is_count
                    else {"sha256": split_artifacts["train"].probe_statistics["sha256"]}
                ),
            },
        )
        if is_count:
            descriptor["dose"] = {
                split: split_artifacts[split].dose for split in ("train", "test")
            }
        descriptors[dataset_id] = descriptor
    return descriptors


def _source_records(
    spec: MaterializationSpec, objects: dict[str, np.ndarray]
) -> dict[str, Any]:
    crop_start = spec.object_resolution // 2
    return {
        "deadleaves": {
            "generator": "create_dead_leaves",
            "parameters": {
                "max_iters": 700,
                "r_min_frac": 0.02,
                "r_max_frac": 0.18,
                "r_sigma": 3.0,
                "phase_max": 0.5,
                "seed": spec.object_seed,
            },
            "dtype": "complex64",
            "shape": list(objects["deadleaves"].shape),
            "sha256": v2_array_sha256(objects["deadleaves"], np.complex64),
        },
        "lines": {
            "generator": "grid_lines_rectangular_v1",
            "parameters": {
                "canvas_size": 2 * spec.object_resolution,
                "object_resolution": spec.object_resolution,
                "crop_start": crop_start,
                "crop_stop": crop_start + spec.object_resolution,
                "nlines": 400,
                "mapping": "rectangular_v1",
                "amplitude_min": 0.3,
                "amplitude_max": 1.0,
                "phase_min": -0.5,
                "phase_max": 0.5,
                "seed": spec.object_seed,
            },
            "dtype": "complex64",
            "shape": list(objects["lines"].shape),
            "sha256": v2_array_sha256(objects["lines"], np.complex64),
        },
    }


def _build_provenance(
    spec: MaterializationSpec,
    objects: dict[str, np.ndarray],
    coordinates: dict[str, tuple[np.ndarray, np.ndarray]],
    raw_geometry: np.ndarray,
    artifacts: dict[tuple[str, str], SplitArtifact],
    descriptors: dict[str, dict[str, Any]],
    seeds: dict[str, dict[str, int]],
) -> dict[str, Any]:
    return {
        "schema_version": CI_COMPATIBILITY_PROVENANCE_V3,
        "materializer_id": MATERIALIZER_ID,
        "materializer_version": 3,
        "generator_commit": lib.generator_commit(),
        "materialization_profile": _materialization_profile(spec),
        "expected_dataset_ids": list(DATASET_IDS),
        "seeds": {
            "object": spec.object_seed,
            "train_coordinates": spec.train_coordinate_seed,
            "test_coordinates": spec.test_coordinate_seed,
            "measurements": seeds,
        },
        "source_objects": _source_records(spec, objects),
        "coordinate_sets": {
            "shared_scan": {
                split: {
                    "count": len(values[0]),
                    "dtype": "float32",
                    "shape": [len(values[0])],
                    "x_sha256": v2_array_sha256(values[0], np.float32),
                    "y_sha256": v2_array_sha256(values[1], np.float32),
                }
                for split, values in coordinates.items()
            }
        },
        "probe_geometries": {
            "raw_probe": {
                "array_key": "probeGeometry",
                "dtype": "complex64",
                "shape": list(raw_geometry.shape),
                "sha256": v2_array_sha256(raw_geometry, np.complex64),
            }
        },
        "datasets": {
            dataset_id: {
                "family": dataset_id.split("_", 1)[0],
                "scale_contract_version": descriptors[dataset_id][
                    "scale_contract_version"
                ],
                "measurement_domain": descriptors[dataset_id]["measurement_domain"],
                "splits": {
                    split: artifacts[(dataset_id, split)].provenance
                    for split in ("train", "test")
                },
            }
            for dataset_id in DATASET_IDS
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_output_root(value: Path) -> Path:
    raw = Path(value)
    if ".." in raw.parts or raw.name in {"", ".", ".."}:
        raise MaterializationError(
            "output_root must be a direct lexical directory path"
        )
    lexical = raw if raw.is_absolute() else Path.cwd() / raw
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink():
            raise MaterializationError(
                f"output_root and its parent chain must not contain symlinks: {candidate}"
            )
    lexical.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (lexical.parent, *lexical.parent.parents):
        if candidate.is_symlink():
            raise MaterializationError(
                f"output_root parent chain became symlinked: {candidate}"
            )
    if lexical.exists() and not lexical.is_dir():
        raise MaterializationError("output_root must be a directory")
    return lexical


def _publication_backup_path(output_root: Path) -> Path:
    return output_root.with_name(f".{output_root.name}.publish-backup")


def _validate_published_bundle(root: Path) -> None:
    descriptor_path = root / DESCRIPTORS_FILENAME
    try:
        payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if set(payload) != {"schema_version", "datasets"}:
            raise DatasetError("published descriptor document is not closed")
        load_checked_dataset_bundle(payload["datasets"], repo_root=root)
    except (OSError, KeyError, json.JSONDecodeError, DatasetError) as exc:
        raise MaterializationError(
            f"published bundle validation failed: {exc}"
        ) from exc


def _recover_publication(output_root: Path) -> None:
    backup_root = _publication_backup_path(output_root)
    if backup_root.is_symlink():
        raise MaterializationError("publication backup must not be a symlink")
    if not backup_root.exists():
        return
    if not output_root.exists():
        _validate_published_bundle(backup_root)
        os.replace(backup_root, output_root)
        _fsync_directory(output_root.parent)
        return
    try:
        _validate_published_bundle(output_root)
    except MaterializationError:
        _validate_published_bundle(backup_root)
        shutil.rmtree(output_root)
        _fsync_directory(output_root.parent)
        os.replace(backup_root, output_root)
        _fsync_directory(output_root.parent)
    else:
        shutil.rmtree(backup_root)
        _fsync_directory(output_root.parent)


def _atomic_guarded_publish(
    staging_root: Path,
    output_root: Path,
    expected_hashes: dict[str, str],
) -> None:
    """Publish a validated directory transactionally without clobbering changes."""
    if output_root.exists():
        mismatched = [
            name
            for name, expected in expected_hashes.items()
            if (output_root / name).exists()
            and _file_sha256(output_root / name) != expected
        ]
        if mismatched:
            raise MaterializationError(
                "refusing to overwrite existing output(s) whose checksum differs "
                f"from the validated materialization: {sorted(mismatched)}"
            )
        if all((output_root / name).exists() for name in expected_hashes):
            shutil.rmtree(staging_root)
            return

    backup_root = _publication_backup_path(output_root)
    if backup_root.exists():
        raise MaterializationError(
            f"refusing publication while stale backup exists: {backup_root}"
        )
    moved_previous = False
    try:
        if output_root.exists():
            os.replace(output_root, backup_root)
            moved_previous = True
            _fsync_directory(output_root.parent)
        os.replace(staging_root, output_root)
        _fsync_directory(output_root.parent)
    except BaseException:
        if moved_previous and backup_root.exists():
            if output_root.exists():
                shutil.rmtree(output_root)
                _fsync_directory(output_root.parent)
            os.replace(backup_root, output_root)
            _fsync_directory(output_root.parent)
        raise
    else:
        if backup_root.exists():
            shutil.rmtree(backup_root)
            _fsync_directory(output_root.parent)


def materialize_ci_compatibility_datasets(
    spec: MaterializationSpec,
    *,
    raw_probe: np.ndarray,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    output_root = _safe_output_root(Path(output_root))
    _recover_publication(output_root)
    raw_geometry = canonicalize_probe(
        np.asarray(raw_probe, dtype=np.complex64),
        (spec.detector_size, spec.detector_size),
    )
    objects = {"deadleaves": _deadleaves_object(spec), "lines": _lines_object(spec)}
    if v2_array_sha256(objects["deadleaves"]) == v2_array_sha256(objects["lines"]):
        raise MaterializationError(
            "Dead Leaves and lines source objects must be distinct"
        )
    coordinates = _coordinates(spec)
    seeds = _measurement_seeds(spec)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        )
    )
    try:
        artifacts = _stream_dataset_files(
            spec,
            objects,
            raw_geometry,
            coordinates,
            seeds,
            staging_root,
        )
        descriptors = _build_descriptors(spec, artifacts, raw_geometry)
        provenance = _build_provenance(
            spec,
            objects,
            coordinates,
            raw_geometry,
            artifacts,
            descriptors,
            seeds,
        )
        provenance_bytes = (
            json.dumps(provenance, indent=1, sort_keys=True) + "\n"
        ).encode()
        provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
        for descriptor in descriptors.values():
            descriptor["provenance_sha256"] = provenance_sha256
        descriptors_bytes = (
            json.dumps(
                {"schema_version": 1, "datasets": descriptors},
                indent=1,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        _write_bytes_fsync(staging_root / PROVENANCE_FILENAME, provenance_bytes)
        _write_bytes_fsync(staging_root / DESCRIPTORS_FILENAME, descriptors_bytes)
        _fsync_directory(staging_root)
        checked = json.loads(descriptors_bytes)["datasets"]
        load_checked_dataset_bundle(checked, repo_root=staging_root)
        expected_hashes = {
            artifact.path: artifact.file_sha256 for artifact in artifacts.values()
        }
        expected_hashes[PROVENANCE_FILENAME] = provenance_sha256
        expected_hashes[DESCRIPTORS_FILENAME] = hashlib.sha256(
            descriptors_bytes
        ).hexdigest()
        _atomic_guarded_publish(staging_root, output_root, expected_hashes)
    except DatasetError as exc:
        raise MaterializationError(
            f"materialized bundle failed v2 preflight: {exc}"
        ) from exc
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return checked


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize four immutable CI compatibility twins."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--detector-size", type=int, default=64)
    parser.add_argument("--object-resolution", type=int, default=320)
    parser.add_argument("--object-seed", type=int, default=1064)
    parser.add_argument("--train-positions", type=int, default=5000)
    parser.add_argument("--test-positions", type=int, default=1250)
    parser.add_argument("--train-coordinate-seed", type=int, default=7)
    parser.add_argument("--test-coordinate-seed", type=int, default=8)
    parser.add_argument("--scan-jitter", type=float, default=1.5)
    parser.add_argument("--measurement-seed", type=int, default=43_200)
    parser.add_argument("--probe-src", type=Path, default=M.FLY64_PROBE_SRC)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    spec = MaterializationSpec(
        detector_size=args.detector_size,
        object_resolution=args.object_resolution,
        object_seed=args.object_seed,
        train_positions=args.train_positions,
        test_positions=args.test_positions,
        train_coordinate_seed=args.train_coordinate_seed,
        test_coordinate_seed=args.test_coordinate_seed,
        scan_jitter=args.scan_jitter,
        measurement_seed=args.measurement_seed,
        probe_source=str(args.probe_src),
    )
    descriptors = materialize_ci_compatibility_datasets(
        spec,
        raw_probe=M.load_probe(args.probe_src, N=args.detector_size),
        output_root=args.output_root,
    )
    for dataset_id in sorted(descriptors):
        descriptor = descriptors[dataset_id]
        print(
            f"{dataset_id}: train={descriptor['train']} test={descriptor['test']} test_sha256={descriptor['test_sha256']}"
        )
    print("wrote", Path(args.output_root) / PROVENANCE_FILENAME)
    print("wrote", Path(args.output_root) / DESCRIPTORS_FILENAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
