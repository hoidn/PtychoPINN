"""Claim-grade Task10 two-dataset lineage, parity, and calibrated-count policy.

Validates the revised (2026-07-10) family: one count-intensity dataset
``ci_3p5m`` calibrated to a mean of ~3.5M measured photons per image (864
counts/pixel at N=64) plus its normalized-amplitude twin ``legacy_amp`` built
from the identical latent object, scan coordinates, and uncalibrated probe
geometry. The former calibrated dose sweep and its multiplier machinery are
removed; per-dataset dose statistics remain mandatory for count data.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

import numpy as np

from .dataset_content import (
    _ContentValidation,
    array_sha256,
    validate_positive_real_scalar_multiple,
)
from .dataset_provenance import ProvenanceDatasetV1, ProvenanceV1
from .dataset_schema import (
    DOSE_SCALE_REL_TOL,
    TASK10_RAW_PROBE_GEOMETRY_KEY,
    DatasetError,
    _TASK10_CI_DATASET_ID,
    _TASK10_CI_TARGET_COUNTS_MEAN,
    _TASK10_DATASET_IDS,
    _TASK10_LEGACY_DATASET_ID,
    _REQUIRED_DOSE_MULTIPLIERS,
)


_TASK10_MIN_PHOTONS_PER_IMAGE = 1_000_000.0


def validate_task10_policy(
    provenance: ProvenanceV1,
    contents: Mapping[str, _ContentValidation],
) -> None:
    """Validate the exact Task10 ``ci_3p5m`` + ``legacy_amp`` family."""
    records = provenance.dataset_map()
    if set(contents) != _TASK10_DATASET_IDS or set(records) != _TASK10_DATASET_IDS:
        raise DatasetError(
            "Task10 CI compatibility provenance requires exactly the two-dataset "
            f"bundle {sorted(_TASK10_DATASET_IDS)}"
        )
    _validate_count_role(
        contents[_TASK10_CI_DATASET_ID], records[_TASK10_CI_DATASET_ID]
    )
    _validate_legacy_role(
        contents[_TASK10_LEGACY_DATASET_ID], records[_TASK10_LEGACY_DATASET_ID]
    )
    _validate_shared_lineage(provenance, contents, records)
    _validate_legacy_parity(contents)
    _validate_raw_geometry_family(provenance, contents, records[_TASK10_CI_DATASET_ID])


def _validate_count_role(
    content: _ContentValidation, record: ProvenanceDatasetV1
) -> None:
    descriptor = content.descriptor
    if (
        descriptor.kind,
        descriptor.truth,
        descriptor.truth_location,
        descriptor.scale_contract_version,
        descriptor.measurement_domain,
    ) != (
        "synthetic",
        "object_truth",
        "embedded_test",
        "ci_intensity_v2",
        "count_intensity",
    ):
        raise DatasetError(
            f"Task10 dataset {_TASK10_CI_DATASET_ID!r} does not have its "
            "required CI role"
        )
    if record.base_dataset_id != _TASK10_CI_DATASET_ID or record.dose_multiplier != 1.0:
        raise DatasetError(
            f"Task10 dataset {_TASK10_CI_DATASET_ID!r} must be its own dose "
            "base with exact multiplier 1"
        )
    if record.target_counts_mean != _TASK10_CI_TARGET_COUNTS_MEAN:
        raise DatasetError(
            f"Task10 dataset {_TASK10_CI_DATASET_ID!r} target_counts_mean must "
            f"be exactly {_TASK10_CI_TARGET_COUNTS_MEAN:g}"
        )
    _validate_target_means(content)


def _validate_target_means(content: _ContentValidation) -> None:
    descriptor = content.descriptor
    assert descriptor.dose is not None
    for split, statistics in (
        ("train", descriptor.dose.train),
        ("test", descriptor.dose.test),
    ):
        if not dose_scale_matches(
            statistics.counts_mean, _TASK10_CI_TARGET_COUNTS_MEAN
        ):
            raise DatasetError(
                f"Task10 dataset {_TASK10_CI_DATASET_ID!r} measured {split} "
                f"counts_mean is not within {DOSE_SCALE_REL_TOL:.0%} of "
                f"{_TASK10_CI_TARGET_COUNTS_MEAN:g}"
            )
        if statistics.photons_per_image_min <= _TASK10_MIN_PHOTONS_PER_IMAGE:
            raise DatasetError(
                f"Task10 dataset {_TASK10_CI_DATASET_ID!r} {split} "
                "photons_per_image_min must be strictly above one million"
            )
        if statistics.saturation_fraction != 0.0:
            raise DatasetError(
                f"Task10 dataset {_TASK10_CI_DATASET_ID!r} {split} "
                "saturation_fraction must be exactly zero"
            )


def _validate_legacy_role(
    legacy: _ContentValidation, record: ProvenanceDatasetV1
) -> None:
    descriptor = legacy.descriptor
    if (
        descriptor.kind,
        descriptor.truth,
        descriptor.truth_location,
        descriptor.scale_contract_version,
        descriptor.measurement_domain,
        record.dose_family_id,
        record.base_dataset_id,
        record.dose_multiplier,
    ) != (
        "synthetic",
        "object_truth",
        "embedded_test",
        "legacy_v1",
        "normalized_amplitude",
        None,
        None,
        None,
    ):
        raise DatasetError("Task10 dataset 'legacy_amp' does not have its legacy role")


def _validate_shared_lineage(
    provenance: ProvenanceV1,
    contents: Mapping[str, _ContentValidation],
    records: Mapping[str, ProvenanceDatasetV1],
) -> None:
    base_record = records[_TASK10_CI_DATASET_ID]
    legacy_record = records[_TASK10_LEGACY_DATASET_ID]
    base_lineage = (
        base_record.source_object_id,
        base_record.coordinate_set_id,
        base_record.probe_geometry_id,
    )
    legacy_lineage = (
        legacy_record.source_object_id,
        legacy_record.coordinate_set_id,
        legacy_record.probe_geometry_id,
    )
    if legacy_lineage != base_lineage:
        raise DatasetError(
            "Task10 legacy twin must share source object, coordinates, and "
            f"uncalibrated probe geometry lineage with {_TASK10_CI_DATASET_ID}"
        )
    assert base_record.source_object_id is not None
    source_digest = provenance.source_object_map()[base_record.source_object_id].sha256
    base = contents[_TASK10_CI_DATASET_ID]
    if source_digest != base.truth_sha256:
        raise DatasetError("Task10 shared source-object digest is inconsistent")
    for dataset_id, content in contents.items():
        for split, split_content in (("train", content.train), ("test", content.test)):
            if split_content.initialization_sha256 != source_digest:
                raise DatasetError(
                    f"Task10 dataset {dataset_id!r} {split} objectGuess does not "
                    "match the provenance source object/latent digest"
                )
    coordinates = provenance.coordinate_set_map()[base_record.coordinate_set_id]
    expected_coordinates = (
        base.train.x_sha256,
        base.train.y_sha256,
        base.test.x_sha256,
        base.test.y_sha256,
    )
    actual_coordinates = (
        coordinates.train_x_sha256,
        coordinates.train_y_sha256,
        coordinates.test_x_sha256,
        coordinates.test_y_sha256,
    )
    if actual_coordinates != expected_coordinates:
        raise DatasetError("Task10 shared coordinate digests are inconsistent")


def _validate_legacy_parity(contents: Mapping[str, _ContentValidation]) -> None:
    base = contents[_TASK10_CI_DATASET_ID]
    legacy = contents[_TASK10_LEGACY_DATASET_ID]
    if (
        legacy.truth_sha256,
        legacy.train.initialization_sha256,
        legacy.test.initialization_sha256,
    ) != (
        base.truth_sha256,
        base.train.initialization_sha256,
        base.test.initialization_sha256,
    ):
        raise DatasetError(
            "Task10 legacy twin truth/object content differs from CI base"
        )
    if (
        legacy.train.x_sha256,
        legacy.train.y_sha256,
        legacy.test.x_sha256,
        legacy.test.y_sha256,
    ) != (
        base.train.x_sha256,
        base.train.y_sha256,
        base.test.x_sha256,
        base.test.y_sha256,
    ):
        raise DatasetError("Task10 legacy twin coordinate content differs from CI base")
    if (
        legacy.train.measurement_sha256 == base.train.measurement_sha256
        or legacy.test.measurement_sha256 == base.test.measurement_sha256
    ):
        raise DatasetError(
            "Task10 legacy twin measurements must differ from count-intensity data"
        )


def _validate_raw_geometry_family(
    provenance: ProvenanceV1,
    contents: Mapping[str, _ContentValidation],
    base_record: ProvenanceDatasetV1,
) -> None:
    geometry = provenance.probe_geometry_map()[base_record.probe_geometry_id]
    if geometry.array_key != TASK10_RAW_PROBE_GEOMETRY_KEY:
        raise DatasetError("Task10 raw probe geometry array key is inconsistent")
    base = contents[_TASK10_CI_DATASET_ID]
    if base.train.raw_probe_geometry_sha256 != geometry.sha256:
        raise DatasetError("Task10 shared raw probe geometry digest is inconsistent")
    for dataset_id, content in contents.items():
        for split, split_content in (("train", content.train), ("test", content.test)):
            raw = split_content.raw_probe_geometry
            if raw is None:
                raise DatasetError(
                    f"Task10 dataset {dataset_id!r} {split} is missing "
                    f"{TASK10_RAW_PROBE_GEOMETRY_KEY}"
                )
            if split_content.raw_probe_geometry_sha256 != geometry.sha256:
                raise DatasetError(
                    f"Task10 dataset {dataset_id!r} {split} raw probe geometry "
                    "does not match the family digest"
                )
            validate_positive_real_probe_scale(
                split_content.probe, raw, dataset_id=dataset_id, split=split
            )


def validate_positive_real_probe_scale(
    probe: np.ndarray,
    raw_geometry: np.ndarray,
    *,
    dataset_id: str,
    split: str,
) -> None:
    validate_positive_real_scalar_multiple(
        probe,
        raw_geometry,
        label=f"Task10 dataset {dataset_id!r} {split} probe/raw geometry",
    )


def validate_dose_families(
    contents: Mapping[str, _ContentValidation],
    records: Mapping[str, ProvenanceDatasetV1],
) -> frozenset[str]:
    """Validate generic calibrated count families and return complete members."""
    families: dict[str, list[str]] = {}
    for dataset_id, content in contents.items():
        if content.descriptor.measurement_domain == "count_intensity":
            family_id = records[dataset_id].dose_family_id
            assert family_id is not None
            families.setdefault(family_id, []).append(dataset_id)
    complete: set[str] = set()
    for family_id, member_ids in families.items():
        complete.update(_validate_dose_family(family_id, member_ids, contents, records))
    return frozenset(complete)


def _validate_dose_family(
    family_id: str,
    member_ids: list[str],
    contents: Mapping[str, _ContentValidation],
    records: Mapping[str, ProvenanceDatasetV1],
) -> set[str]:
    base_ids = {records[item].base_dataset_id for item in member_ids}
    if len(base_ids) != 1:
        raise DatasetError(
            f"dose family {family_id!r} has inconsistent base_dataset_id"
        )
    base_id = next(iter(base_ids))
    if base_id is None or base_id not in contents or base_id not in member_ids:
        raise DatasetError(f"dose family {family_id!r} base dataset is missing")
    base_record = records[base_id]
    if base_record.dose_family_id != family_id or not multiplier_identity_matches(
        base_record.dose_multiplier or 0.0, 1.0
    ):
        raise DatasetError(f"dose family {family_id!r} base must have multiplier 1")
    base = contents[base_id]
    assert base.descriptor.dose is not None
    if base.descriptor.dose.test.counts_mean <= 0:
        raise DatasetError(f"dose family {family_id!r} base count mean must be positive")
    multipliers = [
        _validate_dose_member(
            family_id,
            member_id,
            base,
            contents[member_id],
            base_record,
            records[member_id],
        )
        for member_id in member_ids
    ]
    complete = all(
        any(multiplier_identity_matches(value, required) for value in multipliers)
        for required in _REQUIRED_DOSE_MULTIPLIERS
    )
    return set(member_ids) if complete else set()


def _validate_dose_member(
    family_id: str,
    member_id: str,
    base: _ContentValidation,
    member: _ContentValidation,
    base_record: ProvenanceDatasetV1,
    record: ProvenanceDatasetV1,
) -> float:
    if (
        record.source_object_id,
        record.coordinate_set_id,
        record.probe_geometry_id,
    ) != (
        base_record.source_object_id,
        base_record.coordinate_set_id,
        base_record.probe_geometry_id,
    ):
        raise DatasetError(
            f"dose family {family_id!r} members must share object, coordinates, "
            "and uncalibrated probe geometry lineage"
        )
    assert base.descriptor.dose is not None and member.descriptor.dose is not None
    if (
        member.descriptor.train_sha256 != base.descriptor.train_sha256
        or array_sha256(member.train.probe) != array_sha256(base.train.probe)
        or member.descriptor.dose.train != base.descriptor.dose.train
    ):
        raise DatasetError(
            f"dose family {family_id!r} twins must reuse the base train artifact"
        )
    if (
        member.train.initialization_sha256,
        member.test.initialization_sha256,
    ) != (
        base.train.initialization_sha256,
        base.test.initialization_sha256,
    ):
        raise DatasetError(
            f"dose family {family_id!r} members must share train/test object "
            "initialization content"
        )
    multiplier = record.dose_multiplier
    assert multiplier is not None
    count_ratio = (
        member.descriptor.dose.test.counts_mean / base.descriptor.dose.test.counts_mean
    )
    energy_ratio = member.test.probe_total_energy / base.test.probe_total_energy
    if not dose_scale_matches(count_ratio, multiplier) or not dose_scale_matches(
        energy_ratio, multiplier
    ):
        raise DatasetError(
            f"dose family {family_id!r} test counts and physical probe are not "
            "co-scaled to the declared multiplier"
        )
    if not dose_scale_matches(count_ratio, energy_ratio):
        raise DatasetError(
            f"dose family {family_id!r} measured test count and physical probe "
            "ratios are not directly co-scaled"
        )
    _validate_mode_energy_ratios(family_id, member, base, multiplier)
    return multiplier


def _validate_mode_energy_ratios(
    family_id: str,
    member: _ContentValidation,
    base: _ContentValidation,
    multiplier: float,
) -> None:
    if len(member.test.probe_mode_energies) != len(base.test.probe_mode_energies):
        raise DatasetError(f"dose family {family_id!r} probe mode count changed")
    for member_energy, base_energy in zip(
        member.test.probe_mode_energies,
        base.test.probe_mode_energies,
        strict=True,
    ):
        if not dose_scale_matches(member_energy / base_energy, multiplier):
            raise DatasetError(
                f"dose family {family_id!r} per-mode probe energy is not co-scaled"
            )


def dose_scale_matches(actual: float, expected: float) -> bool:
    """Use the exact expected-denominator 2% boundary for measured means."""
    actual_decimal = Decimal(str(actual))
    expected_decimal = Decimal(str(expected))
    tolerance = Decimal(str(DOSE_SCALE_REL_TOL)) * abs(expected_decimal)
    return abs(actual_decimal - expected_decimal) <= tolerance


def multiplier_identity_matches(actual: float, expected: float) -> bool:
    """Compare declared multiplier identities without measurement tolerance."""
    return Decimal(str(actual)) == Decimal(str(expected))
