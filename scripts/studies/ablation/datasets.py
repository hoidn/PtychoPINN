"""Stable public facade for immutable dataset descriptor validation.

Schema-only parsers resolve declared path identity without opening artifacts.
Loaders additionally perform artifact, provenance, Task10, and grouping
preflight exactly once. Compatibility checks consume sealed in-memory results
and perform no filesystem access. Use :func:`revalidate_dataset_bundle` at an
execution boundary when the artifacts must be checked again.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .dataset_content import (
    ArtifactCache,
    _ContentValidation,
    _validate_runtime_grouping,
    analyze_descriptor_content,
    array_sha256,
    canonicalize_probe,
    file_sha256,
    probe_geometry_sha256,
    probe_l2_norm,
    validate_positive_real_scalar_multiple,
)
from .dataset_provenance import (
    ProvenanceV1,
    ProvenanceV2,
    canonical_array_sha256,
    load_provenance,
    validate_provenance_claims,
)
from .dataset_reference import (
    GridLinesReferenceRecipe,
    LadderDatasetRecipe,
    MaterializedReferenceDataset,
    parse_grid_lines_reference_recipe,
    parse_ladder_dataset,
    validate_ladder_npz_pair,
    validate_reference_npz_pair,
)
from .dataset_schema import (
    DOSE_SCALE_REL_TOL,
    TASK10_CI_COMPATIBILITY_MATERIALIZER_ID,
    TASK10_RAW_PROBE_GEOMETRY_KEY,
    DatasetCapabilities,
    DatasetCompatibilityError,
    DatasetCompatibilityRequirements,
    DatasetDescriptor,
    DatasetError,
    DatasetPathDeclarations,
    DoseDescriptor,
    DoseStatistics,
    GroupingCapability,
    ProbeDescriptor,
    _identifier,
    _validate_descriptor_object,
    parse_checked_dataset_descriptor,
    parse_standalone_dataset_descriptor,
)
from .dataset_task10 import validate_dose_families, validate_task10_policy


_CI_COMPATIBILITY_TARGET_PHOTONS = 3_538_944.0
_CI_COMPATIBILITY_TARGET_REL_TOL = 0.03
_CI_COMPATIBILITY_MIN_PHOTONS = 1_000_000.0


@dataclass(frozen=True, init=False)
class ValidatedDataset:
    """Opaque sealed validation result for one dataset in a validated bundle."""

    descriptor: DatasetDescriptor
    capabilities: DatasetCapabilities
    bundle: ValidatedDatasetBundle
    _content_identity: tuple[object, ...]
    _seal: str

    @classmethod
    def _create(
        cls,
        descriptor: DatasetDescriptor,
        capabilities: DatasetCapabilities,
        bundle: ValidatedDatasetBundle,
        content_identity: tuple[object, ...],
    ) -> ValidatedDataset:
        instance = object.__new__(cls)
        object.__setattr__(instance, "descriptor", descriptor)
        object.__setattr__(instance, "capabilities", capabilities)
        object.__setattr__(instance, "bundle", bundle)
        object.__setattr__(instance, "_content_identity", content_identity)
        object.__setattr__(
            instance,
            "_seal",
            _dataset_seal(descriptor, capabilities, content_identity),
        )
        return instance

    def _assert_sealed(self) -> None:
        if self._seal != _dataset_seal(
            self.descriptor, self.capabilities, self._content_identity
        ):
            raise DatasetCompatibilityError("validated dataset seal is forged or stale")
        self.bundle._assert_sealed()
        if self.bundle.get(self.descriptor.id) is not self:
            raise DatasetCompatibilityError(
                "validated dataset is detached from its sealed bundle"
            )


@dataclass(frozen=True, init=False)
class ValidatedDatasetBundle(Mapping[str, ValidatedDataset]):
    """Opaque immutable mapping returned by complete bundle preflight."""

    _datasets: Mapping[str, ValidatedDataset]
    _descriptors: tuple[DatasetDescriptor, ...]
    _content_identities: tuple[tuple[str, tuple[object, ...]], ...]
    materialization_profile: str | None
    _seal: str

    @classmethod
    def _create(
        cls,
        descriptors: Mapping[str, DatasetDescriptor],
        contents: Mapping[str, _ContentValidation],
        capabilities: Mapping[str, DatasetCapabilities],
        materialization_profile: str | None,
    ) -> ValidatedDatasetBundle:
        instance = object.__new__(cls)
        ordered_ids = tuple(sorted(descriptors))
        ordered_descriptors = tuple(descriptors[key] for key in ordered_ids)
        identities = tuple(
            (key, _content_identity(contents[key])) for key in ordered_ids
        )
        object.__setattr__(instance, "_descriptors", ordered_descriptors)
        object.__setattr__(instance, "_content_identities", identities)
        if materialization_profile not in {None, "claim_grade", "fixture"}:
            raise DatasetError("validated bundle materialization profile is invalid")
        object.__setattr__(instance, "materialization_profile", materialization_profile)
        object.__setattr__(instance, "_datasets", MappingProxyType({}))
        datasets = {
            key: ValidatedDataset._create(
                descriptors[key],
                capabilities[key],
                instance,
                dict(identities)[key],
            )
            for key in ordered_ids
        }
        object.__setattr__(instance, "_datasets", MappingProxyType(datasets))
        object.__setattr__(instance, "_seal", _bundle_seal(instance))
        return instance

    def __getitem__(self, key: str) -> ValidatedDataset:
        return self._datasets[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._datasets)

    def __len__(self) -> int:
        return len(self._datasets)

    def _assert_sealed(self) -> None:
        if self.materialization_profile not in {None, "claim_grade", "fixture"}:
            raise DatasetCompatibilityError(
                "validated dataset bundle materialization profile is forged"
            )
        if self._seal != _bundle_seal(self):
            raise DatasetCompatibilityError(
                "validated dataset bundle seal is forged or stale"
            )


def load_checked_dataset(
    dataset_id: str,
    value: Mapping[str, Any],
    *,
    repo_root: Path,
) -> ValidatedDataset:
    """Load one generic checked descriptor.

    Claim-grade Task10 records require :func:`load_checked_dataset_bundle` so
    sibling-derived capabilities cannot be inferred from a partial family.
    """
    descriptor = parse_checked_dataset_descriptor(
        dataset_id,
        value,
        repo_root=repo_root,
    )
    bundle = _preflight_descriptors(
        {descriptor.id: descriptor}, reject_single_task10=True
    )
    return bundle[descriptor.id]


def load_checked_dataset_bundle(
    values: Mapping[str, Mapping[str, Any]], *, repo_root: Path
) -> ValidatedDatasetBundle:
    """Parse and preflight a complete checked-manifest descriptor mapping."""
    if not isinstance(values, Mapping) or not values:
        raise DatasetError("checked dataset bundle must be a nonempty mapping")
    descriptors = {
        _identifier(dataset_id, "dataset id"): parse_checked_dataset_descriptor(
            dataset_id,
            value,
            repo_root=repo_root,
        )
        for dataset_id, value in values.items()
    }
    return _preflight_descriptors(descriptors)


def load_standalone_dataset(path: Path) -> ValidatedDataset:
    """Load and preflight standalone ``[schema]`` plus ``[dataset]`` TOML."""
    descriptor_path = Path(path).resolve()
    try:
        with descriptor_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DatasetError(
            f"cannot load standalone dataset descriptor {descriptor_path}: {exc}"
        ) from exc
    descriptor = parse_standalone_dataset_descriptor(
        raw,
        descriptor_path=descriptor_path,
    )
    return _preflight_descriptors({descriptor.id: descriptor})[descriptor.id]


def preflight_standalone_dataset(
    descriptor: DatasetDescriptor,
) -> ValidatedDataset:
    """Preflight an already parsed standalone descriptor only."""
    _validate_descriptor_object(descriptor)
    if descriptor.path_origin != "descriptor_directory":
        raise DatasetError(
            "preflight_standalone_dataset requires descriptor-directory origin"
        )
    return _preflight_descriptors({descriptor.id: descriptor})[descriptor.id]


def preflight_dataset(descriptor: DatasetDescriptor) -> ValidatedDataset:
    """Internal compatibility shim; not part of the public facade."""
    _validate_descriptor_object(descriptor)
    return _preflight_descriptors({descriptor.id: descriptor})[descriptor.id]


def revalidate_dataset_bundle(
    validated: ValidatedDatasetBundle,
) -> ValidatedDatasetBundle:
    """Explicitly repeat all I/O and content validation at an execution boundary."""
    if not isinstance(validated, ValidatedDatasetBundle):
        raise DatasetError("revalidation requires a ValidatedDatasetBundle")
    validated._assert_sealed()
    return _preflight_descriptors(
        {descriptor.id: descriptor for descriptor in validated._descriptors}
    )


def validate_dataset_compatibility(
    validated: ValidatedDataset,
    requirements: DatasetCompatibilityRequirements,
) -> ValidatedDataset:
    """Purely enforce typed effective requirements on a sealed preflight result."""
    if not isinstance(validated, ValidatedDataset):
        raise DatasetCompatibilityError("validated must be a ValidatedDataset")
    validated._assert_sealed()
    requirements = _validated_requirements(requirements)
    descriptor = validated.descriptor
    capabilities = validated.capabilities
    _validate_profile_compatibility(descriptor, requirements)
    if not capabilities.supports_grouping_C.supports(requirements.requested_C):
        raise DatasetCompatibilityError(
            f"requested grouping C={requirements.requested_C} exceeds grouping_max_C="
            f"{capabilities.supports_grouping_C.max_C}"
        )
    _validate_required_capabilities(capabilities, requirements)
    _validate_synthetic_count_floor(descriptor)
    return validated


def require_compatible_dataset(
    validated: ValidatedDataset,
    requirements: DatasetCompatibilityRequirements,
) -> ValidatedDataset:
    """Compatibility alias retained for Task 4 callers."""
    return validate_dataset_compatibility(validated, requirements)


def _preflight_descriptors(
    descriptors: Mapping[str, DatasetDescriptor],
    *,
    reject_single_task10: bool = False,
) -> ValidatedDatasetBundle:
    if not descriptors:
        raise DatasetError("descriptor bundle must not be empty")
    if set(descriptors) != {item.id for item in descriptors.values()}:
        raise DatasetError("descriptor bundle keys must exactly match descriptor ids")
    for descriptor in descriptors.values():
        _validate_descriptor_object(descriptor)
    cache = ArtifactCache()
    provenance = _load_shared_provenance(descriptors, cache)
    supplied_ids = set(descriptors)
    if (
        reject_single_task10
        and provenance.materializer_id == TASK10_CI_COMPATIBILITY_MATERIALIZER_ID
    ):
        raise DatasetError(
            "Task10 checked records require load_checked_dataset_bundle with the complete family"
        )
    if set(provenance.expected_dataset_ids) != supplied_ids:
        raise DatasetError(
            "provenance.expected_dataset_ids must exactly match supplied descriptor ids"
        )
    is_v2 = isinstance(provenance, ProvenanceV2)
    is_task10 = provenance.materializer_id == TASK10_CI_COMPATIBILITY_MATERIALIZER_ID
    contents = {
        dataset_id: analyze_descriptor_content(
            descriptor,
            cache,
            requires_raw_probe_geometry=is_task10 or is_v2,
        )
        for dataset_id, descriptor in descriptors.items()
    }
    if is_v2:
        _validate_v2_provenance(provenance, contents)
    else:
        validate_provenance_claims(provenance, contents)
        _validate_dose_identity_references(contents, provenance)
    dose_sweep_ids: frozenset[str] = frozenset()
    if is_v2:
        pass
    elif is_task10:
        validate_task10_policy(provenance, contents)
    else:
        dose_sweep_ids = validate_dose_families(contents, provenance.dataset_map())
    grouping = {
        dataset_id: _validate_runtime_grouping(content, cache)
        for dataset_id, content in contents.items()
    }
    physical = _physical_probe_ids(contents, provenance)
    capabilities = {
        dataset_id: _derive_capabilities(
            content.descriptor,
            has_physical_probe=dataset_id in physical,
            supports_dose_sweep=dataset_id in dose_sweep_ids,
            grouping_max_C=grouping[dataset_id],
        )
        for dataset_id, content in contents.items()
    }
    materialization_profile = (
        provenance.materialization_profile
        if isinstance(provenance, ProvenanceV2)
        else None
    )
    return ValidatedDatasetBundle._create(
        descriptors,
        contents,
        capabilities,
        materialization_profile,
    )


def _load_shared_provenance(
    descriptors: Mapping[str, DatasetDescriptor],
    cache: ArtifactCache,
) -> ProvenanceV1 | ProvenanceV2:
    paths = {descriptor.provenance for descriptor in descriptors.values()}
    hashes = {descriptor.provenance_sha256 for descriptor in descriptors.values()}
    if len(paths) != 1 or len(hashes) != 1:
        raise DatasetError(
            "bundle descriptors must share one provenance path and provenance hash"
        )
    return load_provenance(next(iter(paths)), next(iter(hashes)), cache)


def _validate_v2_provenance(
    provenance: ProvenanceV2,
    contents: Mapping[str, _ContentValidation],
) -> None:
    """Recompute every staged v2 array hash, then enforce twin relationships."""
    records = provenance.dataset_map()
    coordinates = provenance.coordinate_set_map()["shared_scan"]
    raw_geometry = provenance.probe_geometry_map()["raw_probe"]
    sources = provenance.source_object_map()
    _validate_v2_materialization_profile(provenance, contents)
    observed: dict[tuple[str, str], dict[str, str]] = {}

    for dataset_id, content in contents.items():
        descriptor = content.descriptor
        record = records[dataset_id]
        if (
            record.scale_contract_version != descriptor.scale_contract_version
            or record.measurement_domain != descriptor.measurement_domain
        ):
            raise DatasetError(
                f"provenance dataset {dataset_id!r} profile/domain does not agree with descriptor"
            )
        for split, path, expected_file_hash in (
            ("train", descriptor.train, descriptor.train_sha256),
            ("test", descriptor.test, descriptor.test_sha256),
        ):
            claim = record.split_map()[split]
            expected_relative_path = _v2_bundle_relative_split_path(
                descriptor, split, path
            )
            if (
                claim.path != expected_relative_path
                or claim.file_sha256 != expected_file_hash
            ):
                raise DatasetError(
                    f"provenance dataset {dataset_id!r} {split} split path or "
                    "file_sha256 does not match the normalized bundle-relative path"
                )
            try:
                with np.load(path, allow_pickle=False) as archive:
                    truth = np.asarray(archive[descriptor.truth_key])
                    xcoords = np.asarray(archive[descriptor.x_key])
                    ycoords = np.asarray(archive[descriptor.y_key])
                    raw_probe_array = np.asarray(archive[raw_geometry.array_key])
                    stored_probe_array = np.asarray(archive[descriptor.probe_key])
                    measurement = np.asarray(archive[descriptor.measurement_key])
                    _require_v2_array_dtype(
                        truth, np.dtype(np.complex64), descriptor.truth_key
                    )
                    _require_v2_array_dtype(
                        xcoords, np.dtype(np.float32), descriptor.x_key
                    )
                    _require_v2_array_dtype(
                        ycoords, np.dtype(np.float32), descriptor.y_key
                    )
                    _require_v2_array_dtype(
                        raw_probe_array,
                        np.dtype(np.complex64),
                        raw_geometry.array_key,
                    )
                    _require_v2_array_dtype(
                        stored_probe_array,
                        np.dtype(np.complex64),
                        descriptor.probe_key,
                    )
                    truth = np.ascontiguousarray(truth)
                    xcoords = np.ascontiguousarray(xcoords)
                    ycoords = np.ascontiguousarray(ycoords)
                    raw_probe = canonicalize_probe(
                        raw_probe_array,
                        descriptor.detector_shape,
                    )
                    stored_probe = canonicalize_probe(
                        stored_probe_array,
                        descriptor.detector_shape,
                    )
            except (OSError, KeyError, ValueError) as exc:
                raise DatasetError(
                    f"cannot recompute v2 provenance arrays for {dataset_id} {split}: {exc}"
                ) from exc
            hashes = {
                "truth": canonical_array_sha256(truth),
                "x": canonical_array_sha256(xcoords),
                "y": canonical_array_sha256(ycoords),
                "raw_probe": canonical_array_sha256(raw_probe),
                "stored_probe": canonical_array_sha256(stored_probe),
            }
            claims = {
                "truth": claim.truth_sha256,
                "x": claim.xcoords_sha256,
                "y": claim.ycoords_sha256,
                "raw_probe": claim.raw_probe_sha256,
                "stored_probe": claim.stored_probe_sha256,
            }
            if hashes != claims:
                raise DatasetError(
                    f"provenance dataset {dataset_id!r} {split} array sha256 mismatch"
                )
            split_coordinates = getattr(coordinates, split)
            if (
                len(xcoords) != split_coordinates.count
                or xcoords.shape != split_coordinates.shape
                or ycoords.shape != split_coordinates.shape
                or hashes["x"] != split_coordinates.x_sha256
                or hashes["y"] != split_coordinates.y_sha256
            ):
                raise DatasetError(
                    f"provenance dataset {dataset_id!r} {split} coordinate claim mismatch"
                )
            if (
                raw_probe.shape != raw_geometry.shape
                or hashes["raw_probe"] != raw_geometry.sha256
            ):
                raise DatasetError(
                    f"provenance dataset {dataset_id!r} {split} raw probe claim mismatch"
                )
            source = sources[record.family]
            if truth.shape != source.shape or hashes["truth"] != source.sha256:
                raise DatasetError(
                    f"provenance dataset {dataset_id!r} {split} source truth mismatch"
                )
            stored_norm = probe_l2_norm(stored_probe)
            if not math.isclose(stored_norm, claim.stored_probe_l2_norm, rel_tol=1e-12):
                raise DatasetError(
                    f"provenance dataset {dataset_id!r} {split} stored probe norm mismatch"
                )
            if not np.allclose(
                stored_probe,
                raw_probe * np.float32(claim.probe_scale),
                rtol=2e-6,
                atol=1e-7,
            ):
                raise DatasetError(
                    f"provenance dataset {dataset_id!r} {split} probe_scale mismatch"
                )
            if descriptor.measurement_domain == "count_intensity":
                _validate_v2_ci_physical_policy(
                    dataset_id,
                    split,
                    measurement,
                    raw_probe,
                    stored_probe,
                    claim.probe_scale,
                )
            declared_dose = (
                (descriptor.dose.train if split == "train" else descriptor.dose.test)
                if descriptor.dose is not None
                else None
            )
            if claim.dose != declared_dose:
                raise DatasetError(
                    f"provenance dataset {dataset_id!r} {split} dose mismatch"
                )
            observed[(dataset_id, split)] = hashes

    for family in ("deadleaves", "lines"):
        ci_id = f"{family}_ci_3p5m"
        legacy_id = f"{family}_legacy_amp"
        for split in ("train", "test"):
            ci = observed[(ci_id, split)]
            legacy = observed[(legacy_id, split)]
            if (
                ci["truth"] != legacy["truth"]
                or ci["x"] != legacy["x"]
                or ci["y"] != legacy["y"]
            ):
                raise DatasetError(
                    f"provenance {family} CI/legacy twin relationship mismatch"
                )
    if sources["deadleaves"].sha256 == sources["lines"].sha256:
        raise DatasetError(
            "provenance source truth digests must differ across families"
        )
    reference = observed[("deadleaves_ci_3p5m", "train")]
    for hashes in observed.values():
        if hashes["raw_probe"] != reference["raw_probe"]:
            raise DatasetError(
                "provenance cross-family raw probe relationship mismatch"
            )
    for split in ("train", "test"):
        reference = observed[("deadleaves_ci_3p5m", split)]
        for dataset_id in records:
            hashes = observed[(dataset_id, split)]
            if hashes["x"] != reference["x"] or hashes["y"] != reference["y"]:
                raise DatasetError(
                    "provenance cross-family coordinate relationship mismatch"
                )


def _validate_v2_materialization_profile(
    provenance: ProvenanceV2,
    contents: Mapping[str, _ContentValidation],
) -> None:
    if provenance.materialization_profile == "fixture":
        return
    sources = provenance.source_object_map()
    coordinates = provenance.coordinate_set_map()["shared_scan"]
    lines_parameters = dict(sources["lines"].parameters)
    detector_shapes = {
        content.descriptor.detector_shape for content in contents.values()
    }
    if (
        detector_shapes != {(64, 64)}
        or sources["deadleaves"].shape != (320, 320)
        or sources["lines"].shape != (320, 320)
        or lines_parameters["canvas_size"] != 640
        or lines_parameters["object_resolution"] != 320
        or lines_parameters["crop_start"] != 160
        or lines_parameters["crop_stop"] != 480
        or coordinates.train.count != 5000
        or coordinates.test.count != 1250
    ):
        raise DatasetError(
            "claim_grade CI compatibility provenance requires detector size 64, "
            "object resolution 320, lines canvas/crop 640/[160:480], and exactly "
            "5000/1250 train/test positions"
        )


def _v2_bundle_relative_split_path(
    descriptor: DatasetDescriptor,
    split: str,
    resolved_path: Path,
) -> str:
    bundle_root = descriptor.provenance.parent.resolve()
    try:
        relative = resolved_path.relative_to(bundle_root)
    except ValueError as exc:
        raise DatasetError(
            f"v2 provenance {split} path must resolve under the bundle root"
        ) from exc
    normalized = relative.as_posix()
    if (
        normalized in {"", "."}
        or Path(normalized).is_absolute()
        or ".." in Path(normalized).parts
    ):
        raise DatasetError(
            f"v2 provenance {split} path must be normalized and bundle-relative"
        )
    return normalized


def _require_v2_array_dtype(
    array: np.ndarray, expected: np.dtype[Any], array_key: str
) -> None:
    if array.dtype != expected:
        raise DatasetError(
            f"v2 provenance array {array_key!r} must have exact on-disk dtype "
            f"{expected}, got {array.dtype}"
        )


def _validate_v2_ci_physical_policy(
    dataset_id: str,
    split: str,
    measurement: np.ndarray,
    raw_probe: np.ndarray,
    stored_probe: np.ndarray,
    declared_probe_scale: float,
) -> None:
    label = f"v2 CI dataset {dataset_id!r} {split}"
    if not np.issubdtype(measurement.dtype, np.integer):
        raise DatasetError(f"{label} counts must have an integer dtype")
    if np.any(measurement < 0):
        raise DatasetError(f"{label} counts must be nonnegative")
    dtype_max = int(np.iinfo(measurement.dtype).max)
    max_count = int(measurement.max())
    saturation_fraction = float(
        np.count_nonzero(measurement == dtype_max) / measurement.size
    )
    if saturation_fraction != 0.0:
        raise DatasetError(f"{label} saturation_fraction must equal zero")
    if max_count >= dtype_max:
        raise DatasetError(f"{label} max count must be below dtype max")
    photons = measurement.astype(np.float64).sum(axis=(-2, -1))
    mean_photons = float(photons.mean())
    if not math.isclose(
        mean_photons,
        _CI_COMPATIBILITY_TARGET_PHOTONS,
        rel_tol=_CI_COMPATIBILITY_TARGET_REL_TOL,
    ):
        raise DatasetError(
            f"{label} mean photons/image {mean_photons:g} is not within "
            f"{_CI_COMPATIBILITY_TARGET_REL_TOL:.0%} of target "
            f"{_CI_COMPATIBILITY_TARGET_PHOTONS:g}"
        )
    weakest = float(photons.min())
    if weakest < _CI_COMPATIBILITY_MIN_PHOTONS:
        raise DatasetError(
            f"{label} weakest frame must contain at least one million photons"
        )
    validate_positive_real_scalar_multiple(
        stored_probe,
        raw_probe,
        label=f"{label} physical probe calibration",
    )
    observed_scale = probe_l2_norm(stored_probe) / probe_l2_norm(raw_probe)
    if not math.isclose(observed_scale, declared_probe_scale, rel_tol=2e-6):
        raise DatasetError(f"{label} physical probe scale claim mismatch")


def _validate_dose_identity_references(
    contents: Mapping[str, _ContentValidation],
    provenance: ProvenanceV1,
) -> None:
    """Require every count record's base_dataset_id to name a bundled dataset.

    The calibrated dose sweep was removed (design revision 2026-07-10), but
    ``base_dataset_id`` remains a mandatory provenance claim on count records
    and must not dangle outside the supplied bundle.
    """
    records = provenance.dataset_map()
    for dataset_id, content in contents.items():
        if content.descriptor.measurement_domain != "count_intensity":
            continue
        base_id = records[dataset_id].base_dataset_id
        if base_id not in contents:
            raise DatasetError(
                f"provenance dataset {dataset_id!r} base_dataset_id {base_id!r} "
                "does not reference a dataset in this bundle"
            )


def _physical_probe_ids(
    contents: Mapping[str, _ContentValidation], provenance: ProvenanceV1 | ProvenanceV2
) -> frozenset[str]:
    if isinstance(provenance, ProvenanceV2):
        return frozenset(
            dataset_id
            for dataset_id, content in contents.items()
            if content.descriptor.measurement_domain == "count_intensity"
            and content.train.probe_total_energy > 0
            and content.test.probe_total_energy > 0
        )
    records = provenance.dataset_map()
    geometries = provenance.probe_geometry_map()
    return frozenset(
        dataset_id
        for dataset_id, content in contents.items()
        if content.descriptor.measurement_domain == "count_intensity"
        and records[dataset_id].probe_geometry_id in geometries
        and content.train.probe_total_energy > 0
        and content.test.probe_total_energy > 0
    )


def _derive_capabilities(
    descriptor: DatasetDescriptor,
    *,
    has_physical_probe: bool,
    supports_dose_sweep: bool,
    grouping_max_C: int,
) -> DatasetCapabilities:
    return DatasetCapabilities(
        has_object_truth=descriptor.truth == "object_truth",
        has_reference=descriptor.truth == "reference_reconstruction",
        has_physical_probe=has_physical_probe,
        supports_count_metrics=descriptor.measurement_domain == "count_intensity",
        supports_dose_sweep=supports_dose_sweep,
        supports_grouping_C=GroupingCapability(max_C=grouping_max_C),
    )


def _validated_requirements(
    requirements: DatasetCompatibilityRequirements,
) -> DatasetCompatibilityRequirements:
    if not isinstance(requirements, DatasetCompatibilityRequirements):
        raise DatasetCompatibilityError(
            "requirements must be DatasetCompatibilityRequirements"
        )
    checked = DatasetCompatibilityRequirements(
        scale_contract_version=requirements.scale_contract_version,
        measurement_domain=requirements.measurement_domain,
        probe_calibration=requirements.probe_calibration,
        probe_gauge=requirements.probe_gauge,
        requested_C=requirements.requested_C,
        required_capabilities=requirements.required_capabilities,
    )
    if checked != requirements:
        raise DatasetCompatibilityError("requirements are forged or malformed")
    return checked


def _validate_profile_compatibility(
    descriptor: DatasetDescriptor,
    requirements: DatasetCompatibilityRequirements,
) -> None:
    claims = (
        (
            "scale_contract_version",
            descriptor.scale_contract_version,
            requirements.scale_contract_version,
        ),
        (
            "measurement_domain",
            descriptor.measurement_domain,
            requirements.measurement_domain,
        ),
        (
            "probe calibration",
            descriptor.probe.calibration,
            requirements.probe_calibration,
        ),
        ("probe gauge", descriptor.probe.gauge, requirements.probe_gauge),
    )
    for label, actual, required in claims:
        if actual != required:
            raise DatasetCompatibilityError(
                f"dataset {label} does not match effective requirements"
            )


def _validate_required_capabilities(
    capabilities: DatasetCapabilities,
    requirements: DatasetCompatibilityRequirements,
) -> None:
    for name in requirements.required_capabilities:
        value = getattr(capabilities, name)
        supported = (
            value.max_C >= requirements.requested_C
            if isinstance(value, GroupingCapability)
            else value
        )
        if not supported:
            raise DatasetCompatibilityError(f"dataset lacks required capability {name}")


def _validate_synthetic_count_floor(descriptor: DatasetDescriptor) -> None:
    if (
        descriptor.kind != "synthetic"
        or descriptor.measurement_domain != "count_intensity"
    ):
        return
    assert descriptor.dose is not None
    minimum = min(
        descriptor.dose.train.photons_per_image_min,
        descriptor.dose.test.photons_per_image_min,
    )
    if minimum < 1_000_000.0:
        raise DatasetCompatibilityError(
            "synthetic count compatibility requires at least 1,000,000 measured "
            f"photons per image; observed minimum is {minimum}"
        )


def _content_identity(content: _ContentValidation) -> tuple[object, ...]:
    return (
        content.descriptor.train_sha256,
        content.descriptor.test_sha256,
        content.descriptor.reference_sha256,
        content.descriptor.provenance_sha256,
        _split_identity(content.train),
        _split_identity(content.test),
        content.truth_sha256,
    )


def _split_identity(content: Any) -> tuple[object, ...]:
    return (
        array_sha256(content.probe),
        content.probe_l2_norm,
        content.probe_total_energy,
        content.probe_mode_energies,
        content.x_sha256,
        content.y_sha256,
        content.measurement_sha256,
        content.initialization_sha256,
        content.raw_probe_geometry_sha256,
        content.truth_sha256,
        content.truth_dtype,
        content.truth_is_complex,
        content.dose,
    )


def _dataset_seal(
    descriptor: DatasetDescriptor,
    capabilities: DatasetCapabilities,
    content_identity: tuple[object, ...],
) -> str:
    return hashlib.sha256(
        repr((descriptor, capabilities, content_identity)).encode("utf-8")
    ).hexdigest()


def _bundle_seal(bundle: ValidatedDatasetBundle) -> str:
    dataset_seals = tuple(
        (key, item._seal) for key, item in sorted(bundle._datasets.items())
    )
    return hashlib.sha256(
        repr(
            (
                bundle._descriptors,
                bundle._content_identities,
                bundle.materialization_profile,
                dataset_seals,
            )
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DOSE_SCALE_REL_TOL",
    "TASK10_CI_COMPATIBILITY_MATERIALIZER_ID",
    "TASK10_RAW_PROBE_GEOMETRY_KEY",
    "DatasetCapabilities",
    "DatasetCompatibilityError",
    "DatasetCompatibilityRequirements",
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
]
