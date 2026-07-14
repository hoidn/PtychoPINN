from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.studies.ablation_dataset_fixtures import (
    _api,
    _file_sha256,
    _generic_dose_family_bundle,
    _integer_measurement_with_mean,
    _rewrite_generic_dose_test_split,
)


def test_public_api_exports_typed_compatibility_contract() -> None:
    api = _api()

    assert set(api.__all__) == {
        "DOSE_SCALE_REL_TOL",
        "DatasetCompatibilityRequirements",
        "TASK10_CI_COMPATIBILITY_MATERIALIZER_ID",
        "TASK10_RAW_PROBE_GEOMETRY_KEY",
        "DatasetCapabilities",
        "DatasetCompatibilityError",
        "DatasetDescriptor",
        "DatasetError",
        "DatasetPathDeclarations",
        "DoseDescriptor",
        "DoseStatistics",
        "GridLinesReferenceRecipe",
        "GroupingCapability",
        "LadderDatasetRecipe",
        "MaterializedReferenceDataset",
        "ProbeDescriptor",
        "ValidatedDataset",
        "ValidatedDatasetBundle",
        "array_sha256",
        "canonicalize_probe",
        "file_sha256",
        "load_checked_dataset",
        "load_checked_dataset_bundle",
        "load_standalone_dataset",
        "parse_checked_dataset_descriptor",
        "parse_grid_lines_reference_recipe",
        "parse_ladder_dataset",
        "parse_standalone_dataset_descriptor",
        "preflight_standalone_dataset",
        "probe_geometry_sha256",
        "probe_l2_norm",
        "require_compatible_dataset",
        "revalidate_dataset_bundle",
        "validate_dataset_compatibility",
        "validate_ladder_npz_pair",
        "validate_reference_npz_pair",
    }


def test_public_facade_exports_only_supported_preflight_boundaries() -> None:
    from scripts.studies.ablation import datasets

    assert "preflight_dataset" not in datasets.__all__
    assert {
        "ValidatedDatasetBundle",
        "preflight_standalone_dataset",
        "revalidate_dataset_bundle",
    } <= set(datasets.__all__)


def test_cold_facade_import_is_quiet_and_fast() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import time; start=time.perf_counter(); "
                "import scripts.studies.ablation.datasets; "
                "print(time.perf_counter()-start)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    elapsed = float(result.stdout.strip())
    assert result.stderr == ""
    assert elapsed < 2.0


def test_generic_complete_calibrated_dose_family_derives_capability(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _generic_dose_family_bundle(tmp_path)

    validated = api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    assert all(item.descriptor.kind == "experimental" for item in validated.values())
    assert all(item.descriptor.truth == "none" for item in validated.values())
    assert all(item.capabilities.supports_dose_sweep for item in validated.values())


def test_generic_synthetic_object_truth_family_derives_capability(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _generic_dose_family_bundle(
        tmp_path, kind="synthetic", truth="object_truth"
    )

    validated = api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    assert all(item.capabilities.has_object_truth for item in validated.values())
    assert all(item.capabilities.supports_dose_sweep for item in validated.values())


def test_generic_incomplete_calibrated_dose_family_stays_incapable(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _generic_dose_family_bundle(tmp_path)
    del descriptors["dose_4x"]
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    del provenance["datasets"]["dose_4x"]
    del provenance["seeds"]["measurements"]["dose_4x"]
    provenance["expected_dataset_ids"] = sorted(descriptors)
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = hashlib.sha256(
            provenance_path.read_bytes()
        ).hexdigest()

    validated = api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    assert not any(item.capabilities.supports_dose_sweep for item in validated.values())


def test_generic_invalid_calibrated_dose_family_rejects_bad_multiplier(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _generic_dose_family_bundle(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["datasets"]["dose_2x"]["dose_multiplier"] = 4.0
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    provenance_hash = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="dose family|co-scaled|multiplier"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


def test_generic_dose_family_rejects_near_unit_base_multiplier(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _generic_dose_family_bundle(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["datasets"]["dose_1x"]["dose_multiplier"] = 1.01
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="base|multiplier 1|dose family"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


def test_generic_near_required_multipliers_do_not_complete_family(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _generic_dose_family_bundle(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["datasets"]["dose_2x"]["dose_multiplier"] = 2.02
    provenance["datasets"]["dose_4x"]["dose_multiplier"] = 4.04
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    provenance_hash = _file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_hash

    validated = api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    assert not any(item.capabilities.supports_dose_sweep for item in validated.values())


def test_generic_dose_family_rejects_direct_count_probe_scaling_mismatch(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _generic_dose_family_bundle(tmp_path)
    with np.load(tmp_path / descriptors["dose_1x"]["test"], allow_pickle=False) as base:
        base_probe = base["probeGuess"]
    _rewrite_generic_dose_test_split(
        tmp_path,
        descriptors,
        "dose_2x",
        measurement=_integer_measurement_with_mean(25, 394.0),
        probe=(base_probe * np.sqrt(2.03)).astype(np.complex64),
    )

    with pytest.raises(api.DatasetError, match="count|probe|co-scaled"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


def test_generic_complete_dose_family_rejects_test_object_drift(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptors = _generic_dose_family_bundle(tmp_path)
    descriptor = descriptors["dose_2x"]
    test_path = tmp_path / descriptor["test"]
    with np.load(test_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["objectGuess"] = payload["objectGuess"] * np.complex64(2.0)
    np.savez(test_path, **payload)
    descriptor["test_sha256"] = _file_sha256(test_path)

    provenance_path = tmp_path / descriptor["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["datasets"]["dose_2x"]["files"]["test"] = descriptor["test_sha256"]
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    provenance_hash = _file_sha256(provenance_path)
    for item in descriptors.values():
        item["provenance_sha256"] = provenance_hash

    with pytest.raises(api.DatasetError, match="dose family|object|initialization"):
        api.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)
