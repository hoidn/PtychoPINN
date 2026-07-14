from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.studies.ablation_dataset_fixtures import (
    _api,
    _array_sha256,
    _canonical_probe,
    _file_sha256,
    _integer_measurement_with_mean,
    _probe_energies,
    _probe_geometry_sha256,
    _probe_l2_norm,
    _rewrite_bundle_test_split,
    _rewrite_task10_count_mean,
    _rewrite_task10_measurement,
    _task10_bundle,
)


def test_validates_complete_two_dataset_bundle(tmp_path: Path) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)

    validated = api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    assert set(validated) == {"ci_3p5m", "legacy_amp"}
    assert validated["ci_3p5m"].capabilities.supports_count_metrics is True
    assert validated["ci_3p5m"].capabilities.has_physical_probe is True
    assert validated["ci_3p5m"].capabilities.supports_dose_sweep is False
    assert validated["legacy_amp"].capabilities.supports_dose_sweep is False
    assert validated["ci_3p5m"].descriptor.probe.sha256 is not None


@pytest.mark.parametrize("probe_modes", [1, 2, 4])
@pytest.mark.parametrize("probe_magnitude", [1.0, 1e-8])
def test_task10_raw_probe_geometry_supports_randomized_multimode_families(
    tmp_path: Path, probe_modes: int, probe_magnitude: float
) -> None:
    api = _api()
    descriptors = _task10_bundle(
        tmp_path,
        probe_modes=probe_modes,
        probe_magnitude=probe_magnitude,
        randomized_probe=True,
    )

    validated = api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    assert api.TASK10_RAW_PROBE_GEOMETRY_KEY == "probeGeometry"
    assert all(
        item.descriptor.probe_modes == probe_modes for item in validated.values()
    )


@pytest.mark.parametrize("probe_magnitude", [1.0, 1e-8])
@pytest.mark.parametrize(
    ("probe_modes", "case"),
    [
        (1, "phase_rotation"),
        (2, "phase_rotation"),
        (4, "phase_rotation"),
        (2, "per_mode_scaling"),
        (4, "per_mode_scaling"),
    ],
)
def test_task10_rejects_non_positive_scalar_probe_calibration(
    tmp_path: Path, probe_magnitude: float, probe_modes: int, case: str
) -> None:
    api = _api()
    descriptors = _task10_bundle(
        tmp_path,
        probe_modes=probe_modes,
        probe_magnitude=probe_magnitude,
        randomized_probe=True,
    )
    with np.load(
        tmp_path / descriptors["ci_3p5m"]["test"], allow_pickle=False
    ) as archive:
        raw_geometry = archive["probeGeometry"]
    probe = raw_geometry * np.complex64(np.sqrt(2.0))
    if case == "phase_rotation":
        probe *= np.complex64(np.exp(0.25j))
    else:
        probe[1] *= np.complex64(1.1)
    _rewrite_bundle_test_split(
        tmp_path,
        descriptors,
        "ci_3p5m",
        measurement=_integer_measurement_with_mean(25, 864.0, detector_size=64),
        probe=probe,
    )

    with pytest.raises(
        api.DatasetError, match="positive-real scalar|phase|raw geometry"
    ):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


@pytest.mark.parametrize("case", ["missing_array", "shape_drift", "wrong_digest"])
def test_task10_binds_exact_embedded_raw_probe_geometry(
    tmp_path: Path, case: str
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if case in {"missing_array", "shape_drift"}:
        descriptor = descriptors["ci_3p5m"]
        test_path = tmp_path / descriptor["test"]
        with np.load(test_path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        if case == "missing_array":
            del payload["probeGeometry"]
        else:
            payload["probeGeometry"] = payload["probeGeometry"][0]
        np.savez(test_path, **payload)
        descriptor["test_sha256"] = _file_sha256(test_path)
        provenance["datasets"]["ci_3p5m"]["files"]["test"] = descriptor["test_sha256"]
    else:
        provenance["probe_geometries"]["shared_probe_geometry"]["sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(
        api.DatasetError, match="probeGeometry|raw probe geometry|digest"
    ):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


@pytest.mark.parametrize("case", ["raw_zero", "calibrated_zero"])
def test_task10_rejects_zero_norm_raw_or_calibrated_probe(
    tmp_path: Path, case: str
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path, probe_modes=2, randomized_probe=True)
    descriptor = descriptors["ci_3p5m"]
    if case == "calibrated_zero":
        _rewrite_bundle_test_split(
            tmp_path,
            descriptors,
            "ci_3p5m",
            measurement=_integer_measurement_with_mean(25, 864.0, detector_size=64),
            probe=np.zeros((2, 64, 64), dtype=np.complex64),
        )
    else:
        test_path = tmp_path / descriptor["test"]
        with np.load(test_path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["probeGeometry"] = np.zeros((2, 64, 64), dtype=np.complex64)
        np.savez(test_path, **payload)
        descriptor["test_sha256"] = _file_sha256(test_path)
        provenance_path = tmp_path / descriptor["provenance"]
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["datasets"]["ci_3p5m"]["files"]["test"] = descriptor["test_sha256"]
        provenance_path.write_text(
            json.dumps(provenance, sort_keys=True), encoding="utf-8"
        )
        provenance_hash = _file_sha256(provenance_path)
        for item in descriptors.values():
            item["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="positive|zero|energy|norm"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


def test_probe_geometry_digest_is_invariant_to_positive_calibration_scale() -> None:
    api = _api()
    probe = np.array([[1 + 1j, 2 + 0j], [3 - 1j, 4 + 2j]], dtype=np.complex64)[None]
    calibrated = (probe * np.sqrt(2.0)).astype(np.complex64)

    assert api.probe_geometry_sha256(probe) == api.probe_geometry_sha256(calibrated)


def test_bundle_rejects_sibling_forgery_and_expected_id_mismatch(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["datasets"]["ci_3p5m"]["files"]["test"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="ci_3p5m|sibling|file hashes"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    descriptors = _task10_bundle(tmp_path / "ids")
    provenance_path = tmp_path / "ids" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["expected_dataset_ids"].append("fabricated")
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="expected_dataset_ids"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path / "ids")


def test_task10_materializer_requires_the_exact_complete_claim_bundle(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    assert api.TASK10_CI_COMPATIBILITY_MATERIALIZER_ID == "task10_ci_compatibility_v1"

    del descriptors["legacy_amp"]
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    del provenance["datasets"]["legacy_amp"]
    del provenance["seeds"]["measurements"]["legacy_amp"]
    provenance["expected_dataset_ids"] = sorted(descriptors)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="Task10|exactly|legacy_amp"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


def test_generic_checked_bundle_cannot_claim_task10_dose_sweep(tmp_path: Path) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["materializer_id"] = "future_generic_study_v1"
    del provenance["datasets"]["ci_3p5m"]["target_counts_mean"]
    with np.load(
        tmp_path / descriptors["ci_3p5m"]["train"], allow_pickle=False
    ) as archive:
        generic_geometry_hash = _probe_geometry_sha256(archive["probeGuess"])
    provenance["probe_geometries"]["shared_probe_geometry"] = {
        "sha256": generic_geometry_hash
    }
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    validated = api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    assert not any(item.capabilities.supports_dose_sweep for item in validated.values())


@pytest.mark.parametrize(
    ("dataset_id", "field", "value"),
    [
        ("ci_3p5m", "dose_multiplier", 2.0),
        ("ci_3p5m", "base_dataset_id", "some_other_base"),
        ("legacy_amp", "dose_family_id", "calibrated_family"),
    ],
)
def test_task10_materializer_rejects_wrong_dataset_roles(
    tmp_path: Path, dataset_id: str, field: str, value: Any
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["datasets"][dataset_id][field] = value
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(
        api.DatasetError,
        match="Task10|role|multiplier|base|dose family/twin identity",
    ):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


@pytest.mark.parametrize("case", ["object", "coordinates", "probe_geometry"])
def test_task10_rejects_legacy_twin_content_or_lineage_changes(
    tmp_path: Path, case: str
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    legacy = descriptors["legacy_amp"]
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    record = provenance["datasets"]["legacy_amp"]

    if case == "object":
        test_path = tmp_path / legacy["test"]
        with np.load(test_path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["objectGuess"] = payload["objectGuess"] * np.complex64(2.0)
        np.savez(test_path, **payload)
        legacy["test_sha256"] = _file_sha256(test_path)
        record["files"]["test"] = legacy["test_sha256"]
        record["source_object_id"] = "legacy_object"
        provenance["source_objects"]["legacy_object"] = {
            "sha256": _array_sha256(payload["objectGuess"])
        }
    elif case == "coordinates":
        for split in ("train", "test"):
            path = tmp_path / legacy[split]
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["xcoords"] = payload["xcoords"] + 1.0
            np.savez(path, **payload)
            legacy[f"{split}_sha256"] = _file_sha256(path)
            record["files"][split] = legacy[f"{split}_sha256"]
        record["coordinate_set_id"] = "legacy_coordinates"
        provenance["coordinate_sets"]["legacy_coordinates"] = {
            "train_x_sha256": _array_sha256(np.linspace(1.0, 3.0, 3, dtype=np.float64)),
            "train_y_sha256": _array_sha256(np.linspace(3.0, 5.0, 3, dtype=np.float64)),
            "test_x_sha256": _array_sha256(np.linspace(1.0, 3.0, 3, dtype=np.float64)),
            "test_y_sha256": _array_sha256(np.linspace(3.0, 5.0, 3, dtype=np.float64)),
        }
    else:
        with np.load(tmp_path / legacy["train"], allow_pickle=False) as archive:
            changed_probe = np.flip(archive["probeGuess"], axis=-1)
        changed_probe /= np.float32(_probe_l2_norm(changed_probe))
        for split in ("train", "test"):
            path = tmp_path / legacy[split]
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["probeGuess"] = changed_probe
            payload["probeGeometry"] = _canonical_probe(changed_probe)
            np.savez(path, **payload)
            legacy[f"{split}_sha256"] = _file_sha256(path)
            record["files"][split] = legacy[f"{split}_sha256"]
        probe_hash = _array_sha256(_canonical_probe(changed_probe))
        legacy["probe"]["sha256"] = probe_hash
        total_energy, mode_energies = _probe_energies(changed_probe)
        record["probe"].update(
            train_sha256=probe_hash,
            test_sha256=probe_hash,
            train_l2_norm=_probe_l2_norm(changed_probe),
            test_l2_norm=_probe_l2_norm(changed_probe),
            train_total_energy=total_energy,
            test_total_energy=total_energy,
            train_mode_energies=mode_energies,
            test_mode_energies=mode_energies,
        )
        record["probe_geometry_id"] = "legacy_probe_geometry"
        provenance["probe_geometries"]["legacy_probe_geometry"] = {
            "array_key": "probeGeometry",
            "sha256": _array_sha256(_canonical_probe(changed_probe)),
        }

    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="legacy|Task10|twin|lineage"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


def test_task10_source_object_digest_binds_every_train_object_guess(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    rewritten_hashes: dict[str, str] = {}
    for descriptor in descriptors.values():
        train_name = descriptor["train"]
        if train_name not in rewritten_hashes:
            train_path = tmp_path / train_name
            with np.load(train_path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["objectGuess"] = payload["objectGuess"] * np.complex64(2.0)
            np.savez(train_path, **payload)
            rewritten_hashes[train_name] = _file_sha256(train_path)
        descriptor["train_sha256"] = rewritten_hashes[train_name]

    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for dataset_id, descriptor in descriptors.items():
        provenance["datasets"][dataset_id]["files"]["train"] = descriptor[
            "train_sha256"
        ]
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="source object|objectGuess|latent"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


def test_task10_source_object_digest_binds_initialization_separate_from_truth_key(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    descriptor = descriptors["ci_3p5m"]
    test_path = tmp_path / descriptor["test"]
    with np.load(test_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["truthObject"] = payload["objectGuess"].copy()
    payload["objectGuess"] = payload["objectGuess"] * np.complex64(2.0)
    np.savez(test_path, **payload)
    descriptor["truth_key"] = "truthObject"
    descriptor["test_sha256"] = _file_sha256(test_path)

    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    record = provenance["datasets"]["ci_3p5m"]
    record["truth_key"] = "truthObject"
    record["files"]["test"] = descriptor["test_sha256"]
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for item in descriptors.values():
        item["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="source object|objectGuess|latent"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize(
    ("count_mean", "passes"),
    [
        (846.72, True),
        (881.28, True),
        (846.71, False),
        (881.29, False),
        (100.0, False),
    ],
)
def test_task10_train_and_test_means_use_exact_864_denominator_tolerance(
    tmp_path: Path, split: str, count_mean: float, passes: bool
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    _rewrite_task10_count_mean(tmp_path, descriptors, split, count_mean)

    if passes:
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)
    else:
        with pytest.raises(api.DatasetError, match="864|counts_mean"):
            api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


@pytest.mark.parametrize("split", ["train", "test"])
def test_task10_requires_each_split_photon_floor_strictly_above_one_million(
    tmp_path: Path, split: str
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    measurement = np.full((25, 64, 64), 890, dtype=np.uint32)
    measurement[0] = 244  # 999,424 photons: below the strict one-million floor.
    _rewrite_task10_measurement(tmp_path, descriptors, split, measurement)

    with pytest.raises(api.DatasetError, match="photons_per_image_min|one million"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


@pytest.mark.parametrize("split", ["train", "test"])
def test_task10_rejects_photon_floor_equal_to_one_million(
    tmp_path: Path, split: str
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    measurement = np.full((25, 64, 64), 890, dtype=np.uint32)
    measurement[0] = 244
    measurement[0].flat[:576] = 245  # Exactly 1,000,000 photons in frame zero.
    measurement.flat[64 * 64 : 64 * 64 + 16_960] -= 1
    _rewrite_task10_measurement(tmp_path, descriptors, split, measurement)

    with pytest.raises(api.DatasetError, match="photons_per_image_min|one million"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


@pytest.mark.parametrize("split", ["train", "test"])
def test_task10_requires_zero_saturation_for_each_split(
    tmp_path: Path, split: str
) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    measurement = np.full((25, 64, 64), 864, dtype=np.uint16)
    measurement.flat[0] = np.iinfo(np.uint16).max
    measurement.flat[1 : 1 + (np.iinfo(np.uint16).max - 864)] -= 1
    _rewrite_task10_measurement(tmp_path, descriptors, split, measurement)

    with pytest.raises(api.DatasetError, match="saturation_fraction|saturation"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


@pytest.mark.parametrize(
    "case", ["missing_count_target", "wrong_count_target", "legacy_target"]
)
def test_task10_target_count_claim_is_role_closed(tmp_path: Path, case: str) -> None:
    api = _api()
    descriptors = _task10_bundle(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if case == "missing_count_target":
        del provenance["datasets"]["ci_3p5m"]["target_counts_mean"]
    elif case == "wrong_count_target":
        provenance["datasets"]["ci_3p5m"]["target_counts_mean"] = 432.0
    else:
        provenance["datasets"]["legacy_amp"]["target_counts_mean"] = 864.0
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="target_counts_mean|unknown field"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


def test_checked_task10_single_record_requires_bundle_api(tmp_path: Path) -> None:
    from scripts.studies.ablation import datasets

    descriptors = _task10_bundle(tmp_path)
    descriptor = descriptors["ci_3p5m"]

    with pytest.raises(datasets.DatasetError, match="load_checked_dataset_bundle"):
        datasets.load_checked_dataset("ci_3p5m", descriptor, repo_root=tmp_path)
