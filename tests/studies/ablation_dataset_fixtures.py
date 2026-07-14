from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np


def _api() -> ModuleType:
    from scripts.studies.ablation import datasets

    return datasets


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_probe(probe: np.ndarray) -> np.ndarray:
    if probe.ndim == 2:
        return np.ascontiguousarray(probe[None])
    if probe.shape[-1] == 1 and probe.shape[0] == probe.shape[1]:
        return np.ascontiguousarray(np.moveaxis(probe, -1, 0))
    return np.ascontiguousarray(probe)


def _probe_l2_norm(probe: np.ndarray) -> float:
    canonical = _canonical_probe(probe).astype(np.complex128, copy=False)
    return float(np.sqrt(np.sum(np.abs(canonical) ** 2, dtype=np.float64)))


def _probe_energies(probe: np.ndarray) -> tuple[float, list[float]]:
    canonical = _canonical_probe(probe).astype(np.complex128, copy=False)
    mode_energies = np.sum(np.abs(canonical) ** 2, axis=(-2, -1), dtype=np.float64)
    return float(mode_energies.sum()), [float(value) for value in mode_energies]


def _probe_geometry_sha256(probe: np.ndarray) -> str:
    canonical = _canonical_probe(probe).astype(np.complex128, copy=False)
    total, _ = _probe_energies(canonical)
    normalized = canonical / np.sqrt(total)
    quantized = np.round(normalized.real, 7) + 1j * np.round(normalized.imag, 7)
    return _array_sha256(np.asarray(quantized, dtype=np.complex128))


def _dose(measurement: np.ndarray) -> dict[str, int | float]:
    canonical = measurement[..., 0] if measurement.ndim == 4 else measurement
    photons = canonical.astype(np.float64).sum(axis=(-2, -1))
    dtype_max = int(np.iinfo(canonical.dtype).max)
    return {
        "counts_mean": float(canonical.mean(dtype=np.float64)),
        "photons_per_image_min": float(photons.min()),
        "photons_per_image_mean": float(photons.mean()),
        "max_observed_count": int(canonical.max()),
        "dtype_max": dtype_max,
        "saturation_fraction": float(
            np.count_nonzero(canonical == dtype_max) / canonical.size
        ),
    }


def _write_npz(
    path: Path,
    *,
    measurement: np.ndarray,
    probe: np.ndarray,
    samples: int | None = None,
    object_guess: np.ndarray | None = None,
    coordinates: tuple[np.ndarray, np.ndarray] | None = None,
    raw_probe_geometry: np.ndarray | None = None,
) -> None:
    if samples is None:
        samples = int(measurement.shape[0])
    if coordinates is None:
        side = int(np.sqrt(samples))
        if side * side == samples:
            x_grid, y_grid = np.meshgrid(
                np.arange(side, dtype=np.float64),
                np.arange(side, dtype=np.float64),
            )
            coordinates = x_grid.ravel(), y_grid.ravel()
        else:
            coordinates = (
                np.arange(samples, dtype=np.float64),
                np.zeros(samples, dtype=np.float64),
            )
    payload: dict[str, np.ndarray] = {
        "diff3d": measurement,
        "probeGuess": probe,
        "xcoords": coordinates[0],
        "ycoords": coordinates[1],
        "objectGuess": (
            np.ones((4, 4), dtype=np.complex64)
            if object_guess is None
            else object_guess
        ),
    }
    if raw_probe_geometry is not None:
        payload["probeGeometry"] = raw_probe_geometry
    np.savez(path, **payload)


def _integer_measurement_with_mean(
    samples: int, mean: float, *, detector_size: int = 2
) -> np.ndarray:
    size = samples * detector_size * detector_size
    total = int(round(mean * size))
    assert abs(total / size - mean) < 1e-12
    values = np.full(size, total // size, dtype=np.uint32)
    values[: total % size] += 1
    return values.reshape(samples, detector_size, detector_size)


def _refresh_provenance(root: Path, descriptor: dict[str, Any]) -> None:
    with np.load(root / descriptor["train"], allow_pickle=False) as train:
        train_probe = train[descriptor["probe_key"]]
        train_x = train[descriptor["x_key"]]
        train_y = train[descriptor["y_key"]]
    with np.load(root / descriptor["test"], allow_pickle=False) as test:
        test_probe = test[descriptor["probe_key"]]
        test_x = test[descriptor["x_key"]]
        test_y = test[descriptor["y_key"]]
    train_total_energy, train_mode_energies = _probe_energies(train_probe)
    test_total_energy, test_mode_energies = _probe_energies(test_probe)

    source_object_id: str | None = None
    source_objects: dict[str, dict[str, str]] = {}
    if descriptor["truth"] == "object_truth":
        source_object_id = "shared_object"
        truth_path = (
            root / descriptor["test"]
            if descriptor["truth_location"] == "embedded_test"
            else root / descriptor["reference"]
        )
        with np.load(truth_path, allow_pickle=False) as truth_archive:
            truth_array = truth_archive[descriptor["truth_key"]]
        source_objects[source_object_id] = {"sha256": _array_sha256(truth_array)}

    record: dict[str, Any] = {
        "kind": descriptor["kind"],
        "format": descriptor["format"],
        "scale_contract_version": descriptor["scale_contract_version"],
        "measurement_domain": descriptor["measurement_domain"],
        "truth": descriptor["truth"],
        "truth_location": descriptor["truth_location"],
        "measurement_key": descriptor["measurement_key"],
        "probe_key": descriptor["probe_key"],
        "x_key": descriptor["x_key"],
        "y_key": descriptor["y_key"],
        "truth_key": descriptor.get("truth_key"),
        "coords_convention": descriptor["coords_convention"],
        "detector_shape": descriptor["detector_shape"],
        "grouping_max_C": descriptor["grouping_max_C"],
        "probe_modes": descriptor["probe_modes"],
        "source_object_id": source_object_id,
        "coordinate_set_id": "shared_coordinates",
        "probe_geometry_id": "shared_probe_geometry",
        "dose_family_id": (
            "single_dataset_family"
            if descriptor["measurement_domain"] == "count_intensity"
            else None
        ),
        "base_dataset_id": (
            descriptor["_id"]
            if descriptor["measurement_domain"] == "count_intensity"
            else None
        ),
        "dose_multiplier": (
            1.0 if descriptor["measurement_domain"] == "count_intensity" else None
        ),
        "files": {
            "train": descriptor["train_sha256"],
            "test": descriptor["test_sha256"],
        },
        "probe": {
            "source": descriptor["probe"]["source"],
            "calibration": descriptor["probe"]["calibration"],
            "gauge": descriptor["probe"]["gauge"],
            "mask_policy": descriptor["probe"]["mask_policy"],
            "train_sha256": _array_sha256(_canonical_probe(train_probe)),
            "test_sha256": _array_sha256(_canonical_probe(test_probe)),
            "train_l2_norm": _probe_l2_norm(train_probe),
            "test_l2_norm": _probe_l2_norm(test_probe),
            "train_total_energy": train_total_energy,
            "test_total_energy": test_total_energy,
            "train_mode_energies": train_mode_energies,
            "test_mode_energies": test_mode_energies,
        },
    }
    if "reference_sha256" in descriptor:
        record["files"]["reference"] = descriptor["reference_sha256"]
    if "dose" in descriptor:
        record["dose"] = copy.deepcopy(descriptor["dose"])
    provenance = {
        "schema_version": 1,
        "materializer_id": "standalone_dataset_descriptor_v1",
        "materializer_version": 1,
        "generator_commit": "a" * 40,
        "expected_dataset_ids": [descriptor["_id"]],
        "seeds": {
            "object": 7,
            "train_coordinates": 11,
            "test_coordinates": 13,
            "measurements": {descriptor["_id"]: {"train": 17, "test": 19}},
        },
        "source_objects": source_objects,
        "coordinate_sets": {
            "shared_coordinates": {
                "train_x_sha256": _array_sha256(train_x),
                "train_y_sha256": _array_sha256(train_y),
                "test_x_sha256": _array_sha256(test_x),
                "test_y_sha256": _array_sha256(test_y),
            }
        },
        "probe_geometries": {
            "shared_probe_geometry": {"sha256": _probe_geometry_sha256(train_probe)}
        },
        "datasets": {descriptor["_id"]: record},
    }
    path = root / descriptor["provenance"]
    path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    descriptor["provenance_sha256"] = _file_sha256(path)


def _bundle(
    root: Path,
    *,
    dataset_id: str = "synthetic_ci",
    kind: str = "synthetic",
    truth: str = "object_truth",
    domain: str = "count_intensity",
    count_value: int = 250_000,
    probe_layout: str = "2d",
    truth_location: str | None = None,
    sample_count: int = 25,
    coordinates: tuple[np.ndarray, np.ndarray] | None = None,
    probe_array: np.ndarray | None = None,
    raw_probe_geometry: np.ndarray | None = None,
    detector_size: int = 2,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    samples = sample_count
    if domain == "count_intensity":
        measurement: np.ndarray = np.full(
            (samples, detector_size, detector_size), count_value, dtype=np.uint32
        )
        scale_contract = "ci_intensity_v2"
        calibration = "count_amplitude"
        gauge = "physical_count_amplitude"
    else:
        measurement = np.full((samples, detector_size, detector_size), 0.5, dtype=np.float32)
        scale_contract = "legacy_v1"
        calibration = "legacy_normalized"
        gauge = "legacy_normalized"
    probe_2d = (
        np.array([[1 + 1j, 2 + 0j], [3 - 1j, 4 + 2j]], dtype=np.complex64)
        if probe_array is None
        else probe_array
    )
    if probe_layout == "trailing":
        probe = probe_2d[..., None]
    elif probe_layout == "modes":
        probe = np.stack([probe_2d, probe_2d * 2], axis=0)
    else:
        probe = probe_2d
    probe_modes = int(_canonical_probe(probe).shape[0])

    train = root / "train.npz"
    test = root / "test.npz"
    if truth_location is None:
        truth_location = (
            "none"
            if truth == "none"
            else "external_npz"
            if truth == "reference_reconstruction"
            else "embedded_test"
        )
    _write_npz(
        train,
        measurement=measurement,
        probe=probe,
        samples=samples,
        coordinates=coordinates,
        raw_probe_geometry=raw_probe_geometry,
    )
    _write_npz(
        test,
        measurement=measurement.copy(),
        probe=probe.copy(),
        samples=samples,
        coordinates=coordinates,
        raw_probe_geometry=raw_probe_geometry,
    )

    descriptor: dict[str, Any] = {
        "_id": dataset_id,
        "kind": kind,
        "format": "npz_mmap",
        "scale_contract_version": scale_contract,
        "measurement_domain": domain,
        "truth": truth,
        "measurement_key": "diff3d",
        "probe_key": "probeGuess",
        "x_key": "xcoords",
        "y_key": "ycoords",
        "coords_convention": "xy_pixels",
        "detector_shape": [detector_size, detector_size],
        "grouping_max_C": 4,
        "probe_modes": probe_modes,
        "train": "train.npz",
        "test": "test.npz",
        "provenance": "provenance.json",
        "train_sha256": _file_sha256(train),
        "test_sha256": _file_sha256(test),
        "provenance_sha256": "0" * 64,
        "probe": {
            "source": "tiny_fixture",
            "calibration": calibration,
            "gauge": gauge,
            "mask_policy": "model_config",
            "sha256": _array_sha256(_canonical_probe(probe)),
        },
    }
    if truth == "none":
        descriptor["truth_location"] = "none"
    elif truth_location == "external_npz":
        reference = root / "reference.npz"
        np.savez(reference, object=np.ones((5, 5), dtype=np.complex64))
        descriptor.update(
            truth_key="object",
            truth_location="external_npz",
            reference="reference.npz",
            reference_sha256=_file_sha256(reference),
        )
    else:
        descriptor.update(truth_key="objectGuess", truth_location="embedded_test")
    if domain == "count_intensity":
        descriptor["dose"] = {"train": _dose(measurement), "test": _dose(measurement)}
    _refresh_provenance(root, descriptor)
    return descriptor


def _task10_bundle(
    root: Path,
    *,
    probe_modes: int = 1,
    probe_magnitude: float = 1.0,
    randomized_probe: bool = False,
) -> dict[str, dict[str, Any]]:
    """Two-dataset Task10 fixture: ``ci_3p5m`` plus its ``legacy_amp`` twin."""
    rng = np.random.default_rng(2718 + probe_modes)
    if probe_modes == 1 and not randomized_probe:
        base_probe: np.ndarray = np.tile(
            np.array([[1 + 1j, 2 + 0j], [3 - 1j, 4 + 2j]], dtype=np.complex64),
            (32, 32),
        )
    else:
        base_probe = (
            rng.normal(size=(probe_modes, 64, 64))
            + 1j * rng.normal(size=(probe_modes, 64, 64))
        ).astype(np.complex64)
    base_probe *= np.float32(probe_magnitude)
    raw_geometry = _canonical_probe(base_probe).copy()
    base = _bundle(
        root,
        dataset_id="ci_3p5m",
        count_value=864,
        probe_array=base_probe,
        raw_probe_geometry=raw_geometry,
        detector_size=64,
    )
    provenance_path = root / base["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["materializer_id"] = "task10_ci_compatibility_v1"
    base_record = provenance["datasets"]["ci_3p5m"]
    base_record.update(
        dose_family_id="calibrated_family",
        base_dataset_id="ci_3p5m",
        dose_multiplier=1.0,
        target_counts_mean=864.0,
    )
    descriptors = {"ci_3p5m": base}

    legacy_train = root / "legacy_amp_train.npz"
    legacy_test = root / "legacy_amp_test.npz"
    legacy_measurement = np.full((25, 64, 64), 0.5, dtype=np.float32)
    legacy_probe = (base_probe / _probe_l2_norm(base_probe)).astype(np.complex64)
    _write_npz(
        legacy_train,
        measurement=legacy_measurement,
        probe=legacy_probe,
        raw_probe_geometry=raw_geometry,
    )
    _write_npz(
        legacy_test,
        measurement=legacy_measurement.copy(),
        probe=legacy_probe.copy(),
        raw_probe_geometry=raw_geometry,
    )
    legacy = copy.deepcopy(base)
    legacy.update(
        _id="legacy_amp",
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
        train=legacy_train.name,
        test=legacy_test.name,
        train_sha256=_file_sha256(legacy_train),
        test_sha256=_file_sha256(legacy_test),
    )
    legacy["probe"] = {
        "source": base["probe"]["source"],
        "calibration": "legacy_normalized",
        "gauge": "legacy_normalized",
        "mask_policy": base["probe"]["mask_policy"],
        "sha256": _array_sha256(_canonical_probe(legacy_probe)),
    }
    del legacy["dose"]
    descriptors["legacy_amp"] = legacy

    legacy_total_energy, legacy_mode_energies = _probe_energies(legacy_probe)
    legacy_record = copy.deepcopy(base_record)
    legacy_record.update(
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
        dose_family_id=None,
        base_dataset_id=None,
        dose_multiplier=None,
    )
    legacy_record["files"] = {
        "train": legacy["train_sha256"],
        "test": legacy["test_sha256"],
    }
    legacy_record["probe"].update(
        calibration="legacy_normalized",
        gauge="legacy_normalized",
        train_sha256=legacy["probe"]["sha256"],
        test_sha256=legacy["probe"]["sha256"],
        train_l2_norm=_probe_l2_norm(legacy_probe),
        test_l2_norm=_probe_l2_norm(legacy_probe),
        train_total_energy=legacy_total_energy,
        test_total_energy=legacy_total_energy,
        train_mode_energies=legacy_mode_energies,
        test_mode_energies=legacy_mode_energies,
    )
    del legacy_record["dose"]
    del legacy_record["target_counts_mean"]
    provenance["datasets"]["legacy_amp"] = legacy_record
    provenance["probe_geometries"]["shared_probe_geometry"] = {
        "array_key": "probeGeometry",
        "sha256": _array_sha256(raw_geometry),
    }

    dataset_ids = sorted(descriptors)
    provenance["expected_dataset_ids"] = dataset_ids
    provenance["seeds"]["measurements"] = {
        dataset_id: {"train": 17, "test": 19 + index}
        for index, dataset_id in enumerate(dataset_ids)
    }
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash
        descriptor.pop("_id")
    return descriptors


def _generic_dose_family_bundle(
    root: Path,
    *,
    kind: str = "experimental",
    truth: str = "none",
) -> dict[str, dict[str, Any]]:
    """Three calibrated generic count datasets outside the Task10 policy."""
    base_probe = np.array([[1 + 1j, 2 + 0j], [3 - 1j, 4 + 2j]], dtype=np.complex64)
    base = _bundle(
        root,
        dataset_id="dose_1x",
        kind=kind,
        truth=truth,
        count_value=200,
        probe_array=base_probe,
    )
    provenance_path = root / base["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    base_record = provenance["datasets"].pop("dose_1x")
    base_record.update(
        dose_family_id="generic_calibrated_family",
        base_dataset_id="dose_1x",
        dose_multiplier=1.0,
    )
    provenance["datasets"]["dose_1x"] = base_record
    descriptors = {"dose_1x": base}
    base_train_probe_hash = base["probe"]["sha256"]

    for dataset_id, multiplier in (("dose_2x", 2.0), ("dose_4x", 4.0)):
        test_path = root / f"{dataset_id}_test.npz"
        test_measurement = _integer_measurement_with_mean(25, 200.0 * multiplier)
        test_probe = (base_probe * np.sqrt(multiplier)).astype(np.complex64)
        _write_npz(test_path, measurement=test_measurement, probe=test_probe)

        descriptor = copy.deepcopy(base)
        descriptor["_id"] = dataset_id
        descriptor["test"] = test_path.name
        descriptor["test_sha256"] = _file_sha256(test_path)
        descriptor["probe"] = {
            key: value for key, value in base["probe"].items() if key != "sha256"
        }
        descriptor["probe"].update(
            train_sha256=base_train_probe_hash,
            test_sha256=_array_sha256(_canonical_probe(test_probe)),
        )
        descriptor["dose"] = {
            "train": copy.deepcopy(base["dose"]["train"]),
            "test": _dose(test_measurement),
        }
        descriptors[dataset_id] = descriptor

        total_energy, mode_energies = _probe_energies(test_probe)
        record = copy.deepcopy(base_record)
        record.update(
            dose_family_id="generic_calibrated_family",
            base_dataset_id="dose_1x",
            dose_multiplier=multiplier,
        )
        record["files"]["test"] = descriptor["test_sha256"]
        record["probe"].update(
            test_sha256=descriptor["probe"]["test_sha256"],
            test_l2_norm=_probe_l2_norm(test_probe),
            test_total_energy=total_energy,
            test_mode_energies=mode_energies,
        )
        record["dose"] = copy.deepcopy(descriptor["dose"])
        provenance["datasets"][dataset_id] = record

    provenance["expected_dataset_ids"] = sorted(descriptors)
    provenance["seeds"]["measurements"] = {
        dataset_id: {"train": 17, "test": 19 + index}
        for index, dataset_id in enumerate(sorted(descriptors))
    }
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash
        descriptor.pop("_id")
    return descriptors


def _rewrite_generic_dose_test_split(
    root: Path,
    descriptors: dict[str, dict[str, Any]],
    dataset_id: str,
    *,
    measurement: np.ndarray,
    probe: np.ndarray,
) -> None:
    """Rewrite one generic dose test split and refresh all checked claims."""
    descriptor = descriptors[dataset_id]
    test_path = root / descriptor["test"]
    with np.load(test_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["diff3d"] = measurement
    payload["probeGuess"] = probe
    np.savez(test_path, **payload)
    descriptor["test_sha256"] = _file_sha256(test_path)
    test_probe_hash = _array_sha256(_canonical_probe(probe))
    descriptor["probe"]["test_sha256"] = test_probe_hash
    descriptor["dose"]["test"] = _dose(measurement)

    provenance_path = root / descriptor["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    record = provenance["datasets"][dataset_id]
    total_energy, mode_energies = _probe_energies(probe)
    record["files"]["test"] = descriptor["test_sha256"]
    record["probe"].update(
        test_sha256=test_probe_hash,
        test_l2_norm=_probe_l2_norm(probe),
        test_total_energy=total_energy,
        test_mode_energies=mode_energies,
    )
    record["dose"]["test"] = copy.deepcopy(descriptor["dose"]["test"])
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for item in descriptors.values():
        item["provenance_sha256"] = provenance_hash


def _rewrite_task10_measurement(
    root: Path,
    descriptors: dict[str, dict[str, Any]],
    split: str,
    measurement: np.ndarray,
) -> None:
    """Rewrite one Task10 CI split and refresh all checked dose claims."""
    descriptor = descriptors["ci_3p5m"]
    path = root / descriptor[split]
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["diff3d"] = measurement
    np.savez(path, **payload)
    descriptor[f"{split}_sha256"] = _file_sha256(path)
    descriptor["dose"][split] = _dose(measurement)

    provenance_path = root / descriptor["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    record = provenance["datasets"]["ci_3p5m"]
    record["files"][split] = descriptor[f"{split}_sha256"]
    record["dose"][split] = copy.deepcopy(descriptor["dose"][split])
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for item in descriptors.values():
        item["provenance_sha256"] = provenance_hash


def _rewrite_split(
    root: Path,
    descriptor: dict[str, Any],
    split: str,
    transform: Callable[[dict[str, np.ndarray]], None],
) -> None:
    path = root / descriptor[split]
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    transform(payload)
    np.savez(path, **payload)
    descriptor[f"{split}_sha256"] = _file_sha256(path)
    provenance_path = root / descriptor["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["datasets"][descriptor["_id"]]["files"][split] = descriptor[
        f"{split}_sha256"
    ]
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    descriptor["provenance_sha256"] = _file_sha256(provenance_path)


def _mutate_provenance(
    root: Path,
    descriptor: dict[str, Any],
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    provenance_path = root / descriptor["provenance"]
    provenance: dict[str, Any] = json.loads(provenance_path.read_text(encoding="utf-8"))
    mutation(provenance)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    descriptor["provenance_sha256"] = _file_sha256(provenance_path)


def _rewrite_bundle_test_split(
    root: Path,
    descriptors: dict[str, dict[str, Any]],
    dataset_id: str,
    *,
    measurement: np.ndarray,
    probe: np.ndarray,
) -> None:
    descriptor = descriptors[dataset_id]
    test_path = root / descriptor["test"]
    with np.load(test_path, allow_pickle=False) as archive:
        raw_geometry = archive["probeGeometry"]
        coordinates = (archive["xcoords"], archive["ycoords"])
        object_guess = archive["objectGuess"]
    _write_npz(
        test_path,
        measurement=measurement,
        probe=probe,
        coordinates=coordinates,
        object_guess=object_guess,
        raw_probe_geometry=raw_geometry,
    )
    descriptor["test_sha256"] = _file_sha256(test_path)
    test_probe_hash = _array_sha256(_canonical_probe(probe))
    probe_table = descriptor["probe"]
    if probe_table.get("sha256") == test_probe_hash:
        pass
    elif "sha256" in probe_table:
        probe_table["train_sha256"] = probe_table.pop("sha256")
        probe_table["test_sha256"] = test_probe_hash
    elif probe_table["train_sha256"] == test_probe_hash:
        probe_table["sha256"] = test_probe_hash
        del probe_table["train_sha256"]
        del probe_table["test_sha256"]
    else:
        probe_table["test_sha256"] = test_probe_hash
    descriptor["dose"]["test"] = _dose(measurement)

    provenance_path = root / descriptor["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    record = provenance["datasets"][dataset_id]
    total_energy, mode_energies = _probe_energies(probe)
    record["files"]["test"] = descriptor["test_sha256"]
    record["probe"].update(
        test_sha256=test_probe_hash,
        test_l2_norm=_probe_l2_norm(probe),
        test_total_energy=total_energy,
        test_mode_energies=mode_energies,
    )
    record["dose"]["test"] = copy.deepcopy(descriptor["dose"]["test"])
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for item in descriptors.values():
        item["provenance_sha256"] = provenance_hash


def _rewrite_task10_count_mean(
    root: Path,
    descriptors: dict[str, dict[str, Any]],
    split: str,
    count_mean: float,
) -> None:
    """Rewrite one ``ci_3p5m`` split's measured counts to a chosen mean."""
    if split == "test":
        with np.load(
            root / descriptors["ci_3p5m"]["test"], allow_pickle=False
        ) as archive:
            probe = archive["probeGuess"]
        _rewrite_bundle_test_split(
            root,
            descriptors,
            "ci_3p5m",
            measurement=_integer_measurement_with_mean(
                25, count_mean, detector_size=64
            ),
            probe=probe,
        )
        return

    train_path = root / descriptors["ci_3p5m"]["train"]
    with np.load(train_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    measurement = _integer_measurement_with_mean(25, count_mean, detector_size=64)
    payload["diff3d"] = measurement
    np.savez(train_path, **payload)
    train_hash = _file_sha256(train_path)
    train_dose = _dose(measurement)

    provenance_path = root / descriptors["ci_3p5m"]["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    descriptors["ci_3p5m"]["train_sha256"] = train_hash
    descriptors["ci_3p5m"]["dose"]["train"] = copy.deepcopy(train_dose)
    record = provenance["datasets"]["ci_3p5m"]
    record["files"]["train"] = train_hash
    record["dose"]["train"] = copy.deepcopy(train_dose)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash


def _replace_single_descriptor_probes(
    root: Path, descriptor: dict[str, Any], probe: np.ndarray
) -> None:
    for split in ("train", "test"):
        path = root / descriptor[split]
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["probeGuess"] = probe
        np.savez(path, **payload)
        descriptor[f"{split}_sha256"] = _file_sha256(path)
    canonical = _canonical_probe(probe)
    descriptor["probe"] = {
        key: value
        for key, value in descriptor["probe"].items()
        if key not in {"train_sha256", "test_sha256"}
    }
    descriptor["probe"]["sha256"] = _array_sha256(canonical)
    descriptor["probe_modes"] = int(canonical.shape[0])

    provenance_path = root / descriptor["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    record = provenance["datasets"][descriptor["_id"]]
    total_energy, mode_energies = _probe_energies(probe)
    record["probe_modes"] = descriptor["probe_modes"]
    record["files"]["train"] = descriptor["train_sha256"]
    record["files"]["test"] = descriptor["test_sha256"]
    record["probe"].update(
        train_sha256=descriptor["probe"]["sha256"],
        test_sha256=descriptor["probe"]["sha256"],
        train_l2_norm=_probe_l2_norm(probe),
        test_l2_norm=_probe_l2_norm(probe),
        train_total_energy=total_energy,
        test_total_energy=total_energy,
        train_mode_energies=mode_energies,
        test_mode_energies=mode_energies,
    )
    if total_energy > 0:
        provenance["probe_geometries"]["shared_probe_geometry"]["sha256"] = (
            _probe_geometry_sha256(probe)
        )
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    descriptor["provenance_sha256"] = _file_sha256(provenance_path)


def _requirements(
    api: ModuleType,
    *,
    domain: str = "count_intensity",
    requested_c: int = 1,
    required: tuple[str, ...] = (),
) -> Any:
    if domain == "count_intensity":
        scale = "ci_intensity_v2"
        calibration = "count_amplitude"
        gauge = "physical_count_amplitude"
    else:
        scale = "legacy_v1"
        calibration = "legacy_normalized"
        gauge = "legacy_normalized"
    return api.DatasetCompatibilityRequirements(
        scale_contract_version=scale,
        measurement_domain=domain,
        probe_calibration=calibration,
        probe_gauge=gauge,
        requested_C=requested_c,
        required_capabilities=frozenset(required),
    )
