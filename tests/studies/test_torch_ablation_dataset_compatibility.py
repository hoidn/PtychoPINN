from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.studies.ablation_dataset_fixtures import (
    _api,
    _bundle,
    _requirements,
)


@pytest.mark.parametrize(
    ("domain", "requested_c", "required", "match"),
    [
        ("normalized_amplitude", 1, (), "scale_contract_version"),
        ("count_intensity", 5, (), "grouping"),
        ("count_intensity", 0, (), "requested_C"),
        ("count_intensity", 1, ("has_reference",), "has_reference"),
    ],
)
def test_compatibility_rejects_effective_requirement_mismatches(
    tmp_path: Path,
    domain: str,
    requested_c: int,
    required: tuple[str, ...],
    match: str,
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    with pytest.raises(api.DatasetCompatibilityError, match=match):
        api.validate_dataset_compatibility(
            validated,
            _requirements(
                api, domain=domain, requested_c=requested_c, required=required
            ),
        )


@pytest.mark.parametrize(
    "profile_id", ["profile-alpha", "renamed-profile", "arbitrary-A"]
)
def test_compatibility_is_profile_id_neutral(tmp_path: Path, profile_id: str) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    resolved_profiles = {profile_id: _requirements(api, requested_c=4)}

    assert (
        api.validate_dataset_compatibility(validated, resolved_profiles[profile_id])
        is validated
    )


def test_compatibility_accepts_effective_pairs_and_rejects_truth_capabilities(
    tmp_path: Path,
) -> None:
    api = _api()
    count = _bundle(tmp_path / "count")
    count_validated = api.load_checked_dataset(
        count.pop("_id"), count, repo_root=tmp_path / "count"
    )
    assert (
        api.validate_dataset_compatibility(
            count_validated,
            _requirements(
                api,
                requested_c=4,
                required=("has_object_truth", "supports_count_metrics"),
            ),
        )
        is count_validated
    )

    legacy = _bundle(tmp_path / "legacy", domain="normalized_amplitude")
    legacy_validated = api.load_checked_dataset(
        legacy.pop("_id"), legacy, repo_root=tmp_path / "legacy"
    )
    assert (
        api.validate_dataset_compatibility(
            legacy_validated,
            _requirements(api, domain="normalized_amplitude"),
        )
        is legacy_validated
    )
    with pytest.raises(api.DatasetCompatibilityError, match="scale_contract_version"):
        api.validate_dataset_compatibility(legacy_validated, _requirements(api))

    for truth in ("reference_reconstruction", "none"):
        root = tmp_path / truth
        descriptor = _bundle(root, kind="experimental", truth=truth)
        validated = api.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=root
        )
        with pytest.raises(api.DatasetCompatibilityError, match="has_object_truth"):
            api.validate_dataset_compatibility(
                validated,
                _requirements(api, required=("has_object_truth",)),
            )


def test_synthetic_count_photon_floor_is_compatibility_only(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path, count_value=1)
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    with pytest.raises(api.DatasetCompatibilityError, match="1,000,000"):
        api.validate_dataset_compatibility(validated, _requirements(api))


@pytest.mark.parametrize(
    "field",
    [
        "has_reference",
        "has_physical_probe",
        "supports_count_metrics",
        "supports_grouping_C",
    ],
)
def test_compatibility_rejects_forged_capabilities(tmp_path: Path, field: str) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    if field == "supports_grouping_C":
        forged = replace(
            validated.capabilities,
            supports_grouping_C=api.GroupingCapability(max_C=99),
        )
    else:
        forged = replace(
            validated.capabilities,
            **{field: not getattr(validated.capabilities, field)},
        )

    with pytest.raises(TypeError):
        replace(validated, capabilities=forged)


def test_compatibility_repreflights_forged_descriptor(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    invalid_descriptor = replace(
        validated.descriptor,
        probe=replace(validated.descriptor.probe, gauge="legacy_normalized"),
    )

    with pytest.raises(TypeError):
        replace(validated, descriptor=invalid_descriptor)
    with pytest.raises(api.DatasetError, match="provenance|probe"):
        api.preflight_dataset(invalid_descriptor)


def test_validated_dataset_constructor_is_private(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    with pytest.raises(TypeError):
        api.ValidatedDataset(
            descriptor=validated.descriptor,
            capabilities=validated.capabilities,
        )


@pytest.mark.parametrize(
    ("scale", "domain", "calibration", "gauge"),
    [
        (
            "ci_intensity_v2",
            "count_intensity",
            "legacy_normalized",
            "legacy_normalized",
        ),
        (
            "legacy_v1",
            "normalized_amplitude",
            "count_amplitude",
            "physical_count_amplitude",
        ),
    ],
)
def test_requirements_reject_contradictory_complete_pairs(
    scale: str, domain: str, calibration: str, gauge: str
) -> None:
    api = _api()

    with pytest.raises(
        api.DatasetCompatibilityError, match="complete pair|calibration"
    ):
        api.DatasetCompatibilityRequirements(
            scale_contract_version=scale,
            measurement_domain=domain,
            probe_calibration=calibration,
            probe_gauge=gauge,
            requested_C=1,
        )


def test_experimental_count_does_not_invent_synthetic_photon_floor(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptor = _bundle(
        tmp_path,
        kind="experimental",
        truth="none",
        count_value=1,
    )
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert (
        api.validate_dataset_compatibility(validated, _requirements(api)) is validated
    )


def test_compatibility_performs_no_file_or_npz_io(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from scripts.studies.ablation import datasets

    descriptor = _bundle(tmp_path)
    validated = datasets.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("compatibility must be pure after preflight")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(np, "load", forbidden)

    assert (
        datasets.validate_dataset_compatibility(
            validated, _requirements(datasets, requested_c=4)
        )
        is validated
    )


def test_explicit_revalidation_is_the_only_post_preflight_io_boundary(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from scripts.studies.ablation import dataset_content, datasets

    descriptor = _bundle(tmp_path)
    dataset_id = descriptor.pop("_id")
    validated = datasets.load_checked_dataset(
        dataset_id, descriptor, repo_root=tmp_path
    )
    snapshots: list[Path] = []
    real_after_hash = dataset_content.SnapshotReader._after_hash

    def counted_snapshot(reader: Any, handle: Any) -> None:
        snapshots.append(reader.path)
        real_after_hash(reader, handle)

    monkeypatch.setattr(dataset_content.SnapshotReader, "_after_hash", counted_snapshot)

    refreshed = datasets.revalidate_dataset_bundle(validated.bundle)

    assert refreshed[dataset_id].descriptor == validated.descriptor
    assert len(snapshots) == 3
    assert len(set(snapshots)) == 3


@pytest.mark.parametrize("artifact", ["train", "provenance", "reference"])
def test_explicit_revalidation_detects_post_preflight_artifact_changes(
    tmp_path: Path, artifact: str
) -> None:
    from scripts.studies.ablation import datasets

    descriptor = _bundle(
        tmp_path,
        kind="experimental" if artifact == "reference" else "synthetic",
        truth="reference_reconstruction" if artifact == "reference" else "object_truth",
    )
    validated = datasets.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    requirements = _requirements(datasets)
    assert datasets.validate_dataset_compatibility(validated, requirements) is validated
    path = getattr(validated.descriptor, artifact)
    assert path is not None
    with path.open("ab") as output:
        output.write(b"post-preflight-change")

    assert datasets.validate_dataset_compatibility(validated, requirements) is validated
    with pytest.raises(datasets.DatasetError, match="sha256|snapshot|mismatch"):
        datasets.revalidate_dataset_bundle(validated.bundle)


def test_compatibility_rejects_copied_in_memory_capability_forgery_without_io(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from scripts.studies.ablation import datasets

    descriptor = _bundle(tmp_path)
    validated = datasets.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    forged = copy.copy(validated)
    object.__setattr__(
        forged,
        "capabilities",
        replace(forged.capabilities, has_reference=True),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("seal verification must not perform I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(np, "load", forbidden)

    with pytest.raises(datasets.DatasetCompatibilityError, match="forged|seal"):
        datasets.validate_dataset_compatibility(forged, _requirements(datasets))


def test_copied_bundle_descriptor_forgery_fails_before_revalidation_io(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from scripts.studies.ablation import datasets

    descriptor = _bundle(tmp_path)
    validated = datasets.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    forged_bundle = copy.copy(validated.bundle)
    object.__setattr__(
        forged_bundle,
        "_descriptors",
        (replace(validated.descriptor, grouping_max_C=99),),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("bundle seal failure must precede I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(np, "load", forbidden)

    with pytest.raises(datasets.DatasetCompatibilityError, match="bundle seal"):
        datasets.revalidate_dataset_bundle(forged_bundle)


def test_required_capabilities_rejects_non_string_members() -> None:
    from scripts.studies.ablation import datasets

    with pytest.raises(datasets.DatasetError, match="required_capabilities.*str"):
        datasets.DatasetCompatibilityRequirements(
            scale_contract_version="ci_intensity_v2",
            measurement_domain="count_intensity",
            probe_calibration="count_amplitude",
            probe_gauge="physical_count_amplitude",
            requested_C=1,
            required_capabilities=frozenset({"unknown", 1}),  # type: ignore[arg-type]
        )
