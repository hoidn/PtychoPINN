"""Descriptor-v1 schema, immutable value objects, and path-origin validation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

DatasetKind: TypeAlias = Literal["synthetic", "experimental"]

ScaleContract: TypeAlias = Literal["ci_intensity_v2", "legacy_v1"]

MeasurementDomain: TypeAlias = Literal["count_intensity", "normalized_amplitude"]

TruthRole: TypeAlias = Literal["object_truth", "reference_reconstruction", "none"]

TruthLocation: TypeAlias = Literal["embedded_test", "external_npz", "none"]

PathOrigin: TypeAlias = Literal["repository_root", "descriptor_directory"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_GIT_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_PROFILE_PAIRS = {
    ("ci_intensity_v2", "count_intensity"),
    ("legacy_v1", "normalized_amplitude"),
}

DOSE_SCALE_REL_TOL = 0.02

TASK10_CI_COMPATIBILITY_MATERIALIZER_ID = "task10_ci_compatibility_v1"

TASK10_RAW_PROBE_GEOMETRY_KEY = "probeGeometry"

_REQUIRED_DOSE_MULTIPLIERS = (1.0, 2.0, 4.0)

_TASK10_CI_DATASET_ID = "ci_3p5m"

_TASK10_LEGACY_DATASET_ID = "legacy_amp"

_TASK10_DATASET_IDS = frozenset({_TASK10_CI_DATASET_ID, _TASK10_LEGACY_DATASET_ID})

# 864 counts/pixel at N=64 is the calibration detail behind the governing
# 3,538,944 (~3.5M) mean-measured-photons-per-image target (design revision
# 2026-07-10: dose sweep removed; one CI dataset plus its legacy twin).
_TASK10_CI_TARGET_COUNTS_MEAN = 864.0

_PROBE_SCALAR_PHASE_REL_TOL = 2e-5

_PROBE_SCALAR_RESIDUAL_REL_TOL = 2e-5

_RUNTIME_GROUPING_K = 6

_RUNTIME_GROUPING_GRID_SIZE = (2, 2)

_CAPABILITY_NAMES = frozenset(
    {
        "has_object_truth",
        "has_reference",
        "has_physical_probe",
        "supports_count_metrics",
        "supports_dose_sweep",
        "supports_grouping_C",
    }
)

_DATASET_FIELDS = frozenset(
    {
        "id",
        "kind",
        "format",
        "scale_contract_version",
        "measurement_domain",
        "truth",
        "truth_location",
        "truth_key",
        "measurement_key",
        "probe_key",
        "x_key",
        "y_key",
        "coords_convention",
        "detector_shape",
        "grouping_max_C",
        "probe_modes",
        "train",
        "test",
        "reference",
        "provenance",
        "train_sha256",
        "test_sha256",
        "reference_sha256",
        "provenance_sha256",
        "probe",
        "dose",
    }
)

_COMMON_REQUIRED = frozenset(
    {
        "kind",
        "format",
        "scale_contract_version",
        "measurement_domain",
        "truth",
        "truth_location",
        "measurement_key",
        "probe_key",
        "x_key",
        "y_key",
        "coords_convention",
        "detector_shape",
        "grouping_max_C",
        "probe_modes",
        "train",
        "test",
        "provenance",
        "train_sha256",
        "test_sha256",
        "provenance_sha256",
        "probe",
    }
)

_PROBE_BASE_FIELDS = frozenset({"source", "calibration", "gauge", "mask_policy"})

_PROBE_HASH_FIELDS = frozenset({"sha256", "train_sha256", "test_sha256"})

_PROBE_FIELDS = _PROBE_BASE_FIELDS | _PROBE_HASH_FIELDS

_DOSE_FIELDS = frozenset(
    {
        "counts_mean",
        "photons_per_image_min",
        "photons_per_image_mean",
        "max_observed_count",
        "dtype_max",
        "saturation_fraction",
    }
)


class DatasetError(ValueError):
    """Raised when descriptor schema or immutable content is invalid."""


class DatasetCompatibilityError(DatasetError):
    """Raised when validated data cannot support a requested study arm."""


@dataclass(frozen=True)
class ProbeDescriptor:
    source: str
    calibration: Literal["count_amplitude", "legacy_normalized"]
    gauge: Literal["physical_count_amplitude", "legacy_normalized"]
    mask_policy: Literal["model_config", "pre_masked"]
    sha256: str | None
    train_sha256: str | None
    test_sha256: str | None


@dataclass(frozen=True)
class DoseStatistics:
    counts_mean: float
    photons_per_image_min: float
    photons_per_image_mean: float
    max_observed_count: int
    dtype_max: int
    saturation_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "counts_mean": self.counts_mean,
            "photons_per_image_min": self.photons_per_image_min,
            "photons_per_image_mean": self.photons_per_image_mean,
            "max_observed_count": self.max_observed_count,
            "dtype_max": self.dtype_max,
            "saturation_fraction": self.saturation_fraction,
        }


@dataclass(frozen=True)
class DoseDescriptor:
    train: DoseStatistics
    test: DoseStatistics


@dataclass(frozen=True)
class DatasetPathDeclarations:
    """Exact descriptor path strings retained for origin revalidation."""

    train: str
    test: str
    provenance: str
    reference: str | None


@dataclass(frozen=True)
class DatasetDescriptor:
    schema_version: int
    id: str
    kind: DatasetKind
    format: Literal["npz_mmap"]
    scale_contract_version: ScaleContract
    measurement_domain: MeasurementDomain
    truth: TruthRole
    truth_location: TruthLocation
    truth_key: str | None
    measurement_key: str
    probe_key: str
    x_key: str
    y_key: str
    coords_convention: Literal["xy_pixels"]
    detector_shape: tuple[int, int]
    grouping_max_C: int
    probe_modes: int
    train: Path
    test: Path
    reference: Path | None
    provenance: Path
    train_sha256: str
    test_sha256: str
    reference_sha256: str | None
    provenance_sha256: str
    probe: ProbeDescriptor
    dose: DoseDescriptor | None
    path_origin: PathOrigin
    path_base: Path
    descriptor_path: Path | None
    path_declarations: DatasetPathDeclarations


@dataclass(frozen=True)
class GroupingCapability:
    """Inclusive maximum supported grouping count."""

    max_C: int

    def supports(self, requested_C: int) -> bool:
        return 1 <= requested_C <= self.max_C


@dataclass(frozen=True)
class DatasetCapabilities:
    has_object_truth: bool
    has_reference: bool
    has_physical_probe: bool
    supports_count_metrics: bool
    supports_dose_sweep: bool
    supports_grouping_C: GroupingCapability


@dataclass(frozen=True)
class DatasetCompatibilityRequirements:
    """Architecture- and profile-id-neutral effective dataset requirements."""

    scale_contract_version: ScaleContract
    measurement_domain: MeasurementDomain
    probe_calibration: Literal["count_amplitude", "legacy_normalized"]
    probe_gauge: Literal["physical_count_amplitude", "legacy_normalized"]
    requested_C: int
    required_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _enum(
            self.scale_contract_version,
            {"ci_intensity_v2", "legacy_v1"},
            "requirements.scale_contract_version",
        )
        _enum(
            self.measurement_domain,
            {"count_intensity", "normalized_amplitude"},
            "requirements.measurement_domain",
        )
        if (self.scale_contract_version, self.measurement_domain) not in _PROFILE_PAIRS:
            raise DatasetCompatibilityError(
                "requirements scale_contract_version/measurement_domain is not supported"
            )
        _enum(
            self.probe_calibration,
            {"count_amplitude", "legacy_normalized"},
            "requirements.probe_calibration",
        )
        _enum(
            self.probe_gauge,
            {"physical_count_amplitude", "legacy_normalized"},
            "requirements.probe_gauge",
        )
        expected_probe_pair = (
            ("count_amplitude", "physical_count_amplitude")
            if self.measurement_domain == "count_intensity"
            else ("legacy_normalized", "legacy_normalized")
        )
        if (self.probe_calibration, self.probe_gauge) != expected_probe_pair:
            raise DatasetCompatibilityError(
                "requirements must use a supported complete profile/domain/probe "
                "calibration and gauge pair"
            )
        if type(self.requested_C) is not int or self.requested_C <= 0:
            raise DatasetCompatibilityError("requested_C must be a positive integer")
        if not isinstance(self.required_capabilities, frozenset):
            raise DatasetCompatibilityError(
                "required_capabilities must be a frozenset of capability names"
            )
        if any(not isinstance(value, str) for value in self.required_capabilities):
            raise DatasetCompatibilityError(
                "required_capabilities members must all be str capability names"
            )
        unknown = self.required_capabilities - _CAPABILITY_NAMES
        if unknown:
            raise DatasetCompatibilityError(
                f"unknown required capabilities: {sorted(unknown)}"
            )


def _validate_descriptor_object(descriptor: DatasetDescriptor) -> None:
    if not isinstance(descriptor, DatasetDescriptor):
        raise DatasetError("descriptor must be a DatasetDescriptor")
    _validate_descriptor_profile(descriptor)
    _validate_descriptor_truth(descriptor)
    _validate_descriptor_keys_and_shape(descriptor)
    _validate_descriptor_paths(descriptor)
    _validate_descriptor_hashes_and_probe(descriptor)
    _validate_descriptor_dose(descriptor)


def _validate_descriptor_profile(descriptor: DatasetDescriptor) -> None:
    if type(descriptor.schema_version) is not int or descriptor.schema_version != 1:
        raise DatasetError("schema_version must be non-bool integer 1")
    _identifier(descriptor.id, "id")
    _enum(descriptor.kind, {"synthetic", "experimental"}, "kind")
    _enum(descriptor.format, {"npz_mmap"}, "format")
    _enum(
        descriptor.scale_contract_version,
        {"ci_intensity_v2", "legacy_v1"},
        "scale_contract_version",
    )
    _enum(
        descriptor.measurement_domain,
        {"count_intensity", "normalized_amplitude"},
        "measurement_domain",
    )
    if (
        descriptor.scale_contract_version,
        descriptor.measurement_domain,
    ) not in _PROFILE_PAIRS:
        raise DatasetError("descriptor profile/domain is not a supported pair")


def _validate_descriptor_truth(descriptor: DatasetDescriptor) -> None:
    _enum(
        descriptor.truth,
        {"object_truth", "reference_reconstruction", "none"},
        "truth",
    )
    _enum(
        descriptor.truth_location,
        {"embedded_test", "external_npz", "none"},
        "truth_location",
    )
    if descriptor.kind == "synthetic" and descriptor.truth != "object_truth":
        raise DatasetError("synthetic datasets require truth=object_truth")
    if descriptor.truth == "none":
        if descriptor.truth_location != "none" or descriptor.truth_key is not None:
            raise DatasetError(
                "truth=none requires no truth_key and truth_location=none"
            )
    else:
        if descriptor.truth_location == "none" or descriptor.truth_key is None:
            raise DatasetError(
                "present truth requires truth_key and a content location"
            )
        _nonempty(descriptor.truth_key, "truth_key")


def _validate_descriptor_keys_and_shape(descriptor: DatasetDescriptor) -> None:
    for field in ("measurement_key", "probe_key", "x_key", "y_key"):
        _nonempty(getattr(descriptor, field), field)
    _enum(descriptor.coords_convention, {"xy_pixels"}, "coords_convention")
    if not isinstance(descriptor.detector_shape, tuple):
        raise DatasetError("detector_shape must be an exact tuple in DatasetDescriptor")
    _detector_shape(descriptor.detector_shape)
    _positive_int(descriptor.grouping_max_C, "grouping_max_C")
    _positive_int(descriptor.probe_modes, "probe_modes")


def _validate_descriptor_paths(descriptor: DatasetDescriptor) -> None:
    for field in ("train", "test", "provenance"):
        path = getattr(descriptor, field)
        if not isinstance(path, Path) or not path.is_absolute():
            raise DatasetError(f"{field} must be an absolute resolved Path")
    _enum(
        descriptor.path_origin,
        {"repository_root", "descriptor_directory"},
        "path_origin",
    )
    if (
        not isinstance(descriptor.path_base, Path)
        or not descriptor.path_base.is_absolute()
    ):
        raise DatasetError("path_base must be an absolute Path")
    if descriptor.path_origin == "repository_root":
        if descriptor.descriptor_path is not None:
            raise DatasetError(
                "descriptor_path must be None for repository-root identity"
            )
    elif (
        not isinstance(descriptor.descriptor_path, Path)
        or not descriptor.descriptor_path.is_absolute()
    ):
        raise DatasetError(
            "descriptor_path must be an absolute Path for descriptor-directory identity"
        )
    _validate_path_identity(descriptor)


def _validate_descriptor_hashes_and_probe(descriptor: DatasetDescriptor) -> None:
    if descriptor.truth_location == "external_npz":
        if (
            not isinstance(descriptor.reference, Path)
            or not descriptor.reference.is_absolute()
            or descriptor.reference_sha256 is None
        ):
            raise DatasetError("external_npz truth requires reference path/hash")
    elif descriptor.reference is not None or descriptor.reference_sha256 is not None:
        raise DatasetError(
            "reference path/hash is forbidden unless truth is external_npz"
        )
    for field in ("train_sha256", "test_sha256", "provenance_sha256"):
        _hash(getattr(descriptor, field), field)
    if descriptor.reference_sha256 is not None:
        _hash(descriptor.reference_sha256, "reference_sha256")
    _validate_probe_descriptor_object(descriptor.probe, descriptor.measurement_domain)


def _validate_descriptor_dose(descriptor: DatasetDescriptor) -> None:
    if descriptor.measurement_domain == "count_intensity":
        if not isinstance(descriptor.dose, DoseDescriptor):
            raise DatasetError("count descriptor requires dose")
        if not isinstance(descriptor.dose.train, DoseStatistics) or not isinstance(
            descriptor.dose.test, DoseStatistics
        ):
            raise DatasetError("dose train/test must be DoseStatistics")
        _parse_dose_statistics(descriptor.dose.train.to_dict(), "dose.train")
        _parse_dose_statistics(descriptor.dose.test.to_dict(), "dose.test")
    elif descriptor.dose is not None:
        raise DatasetError("normalized amplitude descriptor forbids dose")


def _validate_path_identity(descriptor: DatasetDescriptor) -> None:
    declarations = descriptor.path_declarations
    if not isinstance(declarations, DatasetPathDeclarations):
        raise DatasetError("path_declarations must be DatasetPathDeclarations")
    declared = {
        "train": declarations.train,
        "test": declarations.test,
        "provenance": declarations.provenance,
        "reference": declarations.reference,
    }
    for field in ("train", "test", "provenance"):
        _nonempty(declared[field], f"path_declarations.{field}")
    if descriptor.reference is None:
        if declarations.reference is not None:
            raise DatasetError(
                "path_declarations.reference is forbidden without a reference path"
            )
    else:
        _nonempty(declarations.reference, "path_declarations.reference")

    if descriptor.path_origin == "descriptor_directory":
        assert descriptor.descriptor_path is not None
        if (
            descriptor.descriptor_path.parent.resolve()
            != descriptor.path_base.resolve()
        ):
            raise DatasetError(
                "descriptor_path must retain its original descriptor directory"
            )

    fields = ("train", "test", "provenance") + (
        ("reference",) if descriptor.reference is not None else ()
    )
    for field in fields:
        declaration = declared[field]
        assert declaration is not None
        expected = _resolve_path(
            declaration,
            field,
            path_origin=descriptor.path_origin,
            path_base=descriptor.path_base,
            repository_bundle_provenance=declarations.provenance,
        )
        actual = getattr(descriptor, field)
        if actual != expected:
            raise DatasetError(
                f"{field} path identity does not match its declared origin"
            )


def _validate_probe_descriptor_object(
    probe: ProbeDescriptor, domain: MeasurementDomain
) -> None:
    if not isinstance(probe, ProbeDescriptor):
        raise DatasetError("probe must be ProbeDescriptor")
    _nonempty(probe.source, "probe.source")
    _enum(
        probe.calibration,
        {"count_amplitude", "legacy_normalized"},
        "probe.calibration",
    )
    _enum(
        probe.gauge,
        {"physical_count_amplitude", "legacy_normalized"},
        "probe.gauge",
    )
    expected_pair = (
        ("count_amplitude", "physical_count_amplitude")
        if domain == "count_intensity"
        else ("legacy_normalized", "legacy_normalized")
    )
    if (probe.calibration, probe.gauge) != expected_pair:
        raise DatasetError("probe calibration/gauge does not match measurement domain")
    _enum(probe.mask_policy, {"model_config", "pre_masked"}, "probe.mask_policy")
    if probe.sha256 is not None:
        _hash(probe.sha256, "probe.sha256")
        if probe.train_sha256 is not None or probe.test_sha256 is not None:
            raise DatasetError("probe hash form mixes shorthand and split hashes")
    else:
        if probe.train_sha256 is None or probe.test_sha256 is None:
            raise DatasetError(
                "probe split hash form requires train_sha256 and test_sha256"
            )
        _hash(probe.train_sha256, "probe.train_sha256")
        _hash(probe.test_sha256, "probe.test_sha256")
        if probe.train_sha256 == probe.test_sha256:
            raise DatasetError("identical probe split hashes must use sha256 shorthand")


def _parse_descriptor(
    dataset_id: str,
    raw_value: Mapping[str, Any],
    *,
    path_origin: PathOrigin,
    path_base: Path,
    descriptor_path: Path | None,
    id_is_external: bool,
) -> DatasetDescriptor:
    value = _mapping(raw_value, f"dataset {dataset_id}")
    _closed(value, _DATASET_FIELDS, f"dataset {dataset_id}")
    required = set(_COMMON_REQUIRED)
    if not id_is_external:
        required.add("id")
    _required(value, required, f"dataset {dataset_id}")
    if id_is_external and "id" in value:
        raise DatasetError(
            "checked dataset id is supplied externally; field id is forbidden"
        )
    if not id_is_external and value["id"] != dataset_id:
        raise DatasetError("dataset.id does not match parsed standalone identity")

    kind = cast(
        DatasetKind, _enum(value["kind"], {"synthetic", "experimental"}, "kind")
    )
    storage_format = cast(
        Literal["npz_mmap"], _enum(value["format"], {"npz_mmap"}, "format")
    )
    scale = cast(
        ScaleContract,
        _enum(
            value["scale_contract_version"],
            {"ci_intensity_v2", "legacy_v1"},
            "scale_contract_version",
        ),
    )
    domain = cast(
        MeasurementDomain,
        _enum(
            value["measurement_domain"],
            {"count_intensity", "normalized_amplitude"},
            "measurement_domain",
        ),
    )
    if (scale, domain) not in _PROFILE_PAIRS:
        raise DatasetError(
            "scale_contract_version and measurement_domain must be a supported pair: "
            "CI/count or legacy/normalized amplitude"
        )
    truth = cast(
        TruthRole,
        _enum(
            value["truth"],
            {"object_truth", "reference_reconstruction", "none"},
            "truth",
        ),
    )
    location = cast(
        TruthLocation,
        _enum(
            value["truth_location"],
            {"embedded_test", "external_npz", "none"},
            "truth_location",
        ),
    )
    if kind == "synthetic" and truth != "object_truth":
        raise DatasetError("synthetic datasets require truth=object_truth")
    _validate_truth_fields(value, truth, location)

    detector_shape = _detector_shape(value["detector_shape"])
    probe = _parse_probe(value["probe"], domain)
    dose = _parse_dose(value.get("dose"), domain)
    path_declarations = DatasetPathDeclarations(
        train=_nonempty(value["train"], "train"),
        test=_nonempty(value["test"], "test"),
        provenance=_nonempty(value["provenance"], "provenance"),
        reference=(
            _nonempty(value["reference"], "reference") if "reference" in value else None
        ),
    )

    def resolver(field: str) -> Path:
        return _resolve_path(
            value[field],
            field,
            path_origin=path_origin,
            path_base=path_base,
            repository_bundle_provenance=path_declarations.provenance,
        )

    reference = resolver("reference") if "reference" in value else None
    reference_sha = (
        _hash(value["reference_sha256"], "reference_sha256")
        if "reference_sha256" in value
        else None
    )

    return DatasetDescriptor(
        schema_version=1,
        id=dataset_id,
        kind=kind,
        format=storage_format,
        scale_contract_version=scale,
        measurement_domain=domain,
        truth=truth,
        truth_location=location,
        truth_key=_nonempty(value["truth_key"], "truth_key")
        if "truth_key" in value
        else None,
        measurement_key=_nonempty(value["measurement_key"], "measurement_key"),
        probe_key=_nonempty(value["probe_key"], "probe_key"),
        x_key=_nonempty(value["x_key"], "x_key"),
        y_key=_nonempty(value["y_key"], "y_key"),
        coords_convention=cast(
            Literal["xy_pixels"],
            _enum(value["coords_convention"], {"xy_pixels"}, "coords_convention"),
        ),
        detector_shape=detector_shape,
        grouping_max_C=_positive_int(value["grouping_max_C"], "grouping_max_C"),
        probe_modes=_positive_int(value["probe_modes"], "probe_modes"),
        train=resolver("train"),
        test=resolver("test"),
        reference=reference,
        provenance=resolver("provenance"),
        train_sha256=_hash(value["train_sha256"], "train_sha256"),
        test_sha256=_hash(value["test_sha256"], "test_sha256"),
        reference_sha256=reference_sha,
        provenance_sha256=_hash(value["provenance_sha256"], "provenance_sha256"),
        probe=probe,
        dose=dose,
        path_origin=path_origin,
        path_base=path_base.resolve(),
        descriptor_path=descriptor_path,
        path_declarations=path_declarations,
    )


def parse_checked_dataset_descriptor(
    dataset_id: str,
    value: Mapping[str, Any],
    *,
    repo_root: Path,
) -> DatasetDescriptor:
    """Parse one checked descriptor without opening any declared artifact."""
    identifier = _identifier(dataset_id, "dataset id")
    return _parse_descriptor(
        identifier,
        value,
        path_origin="repository_root",
        path_base=Path(repo_root).resolve(),
        descriptor_path=None,
        id_is_external=True,
    )


def parse_standalone_dataset_descriptor(
    value: Mapping[str, Any],
    *,
    descriptor_path: Path,
) -> DatasetDescriptor:
    """Parse a complete standalone descriptor document without artifact I/O."""
    path = Path(descriptor_path).resolve()
    root = _mapping(value, "descriptor")
    _closed(root, {"schema", "dataset"}, "descriptor", noun="table")
    _required(root, {"schema", "dataset"}, "descriptor")
    schema = _mapping(root["schema"], "schema")
    _closed(schema, {"version"}, "schema")
    _required(schema, {"version"}, "schema")
    if type(schema["version"]) is not int or schema["version"] != 1:
        raise DatasetError("schema.version must be integer 1")
    dataset = _mapping(root["dataset"], "dataset")
    if "id" not in dataset:
        raise DatasetError("dataset.id is required for standalone descriptors")
    identifier = _identifier(dataset["id"], "dataset.id")
    return _parse_descriptor(
        identifier,
        dataset,
        path_origin="descriptor_directory",
        path_base=path.parent,
        descriptor_path=path,
        id_is_external=False,
    )


def _validate_truth_fields(
    value: Mapping[str, Any], truth: TruthRole, location: TruthLocation
) -> None:
    has_key = "truth_key" in value
    has_reference = "reference" in value or "reference_sha256" in value
    if truth == "none":
        if location != "none":
            raise DatasetError("truth=none requires truth_location=none")
        if has_key or has_reference:
            raise DatasetError(
                "truth_key/reference fields are forbidden when truth=none"
            )
        return
    if not has_key:
        raise DatasetError("truth_key is required when truth is present")
    if location == "embedded_test":
        if has_reference:
            raise DatasetError(
                "reference fields are forbidden for truth_location=embedded_test"
            )
    elif location == "external_npz":
        missing = {"reference", "reference_sha256"} - set(value)
        if missing:
            raise DatasetError(
                f"external_npz truth is missing required fields {sorted(missing)}"
            )
    else:
        raise DatasetError(
            "present truth requires embedded_test or external_npz truth_location"
        )


def _parse_probe(value: Any, domain: MeasurementDomain) -> ProbeDescriptor:
    table = _mapping(value, "probe")
    _closed(table, _PROBE_FIELDS, "probe")
    _required(table, _PROBE_BASE_FIELDS, "probe")
    hash_fields = set(table) & _PROBE_HASH_FIELDS
    if hash_fields == {"sha256"}:
        sha256 = _hash(table["sha256"], "probe.sha256")
        train_sha256 = test_sha256 = None
    elif hash_fields == {"train_sha256", "test_sha256"}:
        sha256 = None
        train_sha256 = _hash(table["train_sha256"], "probe.train_sha256")
        test_sha256 = _hash(table["test_sha256"], "probe.test_sha256")
        if train_sha256 == test_sha256:
            raise DatasetError(
                "probe split hashes are identical; use sha256 shorthand instead"
            )
    else:
        raise DatasetError(
            "probe hash identity must use exactly sha256 shorthand or both "
            "train_sha256 and test_sha256"
        )
    calibration = cast(
        Literal["count_amplitude", "legacy_normalized"],
        _enum(
            table["calibration"],
            {"count_amplitude", "legacy_normalized"},
            "probe.calibration",
        ),
    )
    gauge = cast(
        Literal["physical_count_amplitude", "legacy_normalized"],
        _enum(
            table["gauge"],
            {"physical_count_amplitude", "legacy_normalized"},
            "probe.gauge",
        ),
    )
    expected = (
        ("count_amplitude", "physical_count_amplitude")
        if domain == "count_intensity"
        else ("legacy_normalized", "legacy_normalized")
    )
    if (calibration, gauge) != expected:
        label = (
            "CI/count" if domain == "count_intensity" else "legacy/normalized amplitude"
        )
        raise DatasetError(f"{label} requires probe calibration/gauge pair {expected}")
    return ProbeDescriptor(
        source=_nonempty(table["source"], "probe.source"),
        calibration=calibration,
        gauge=gauge,
        mask_policy=cast(
            Literal["model_config", "pre_masked"],
            _enum(
                table["mask_policy"],
                {"model_config", "pre_masked"},
                "probe.mask_policy",
            ),
        ),
        sha256=sha256,
        train_sha256=train_sha256,
        test_sha256=test_sha256,
    )


def _parse_dose(value: Any, domain: MeasurementDomain) -> DoseDescriptor | None:
    if domain == "normalized_amplitude":
        if value is not None:
            raise DatasetError("dose is forbidden for normalized-amplitude datasets")
        return None
    if value is None:
        raise DatasetError("dose.train and dose.test are required for count datasets")
    table = _mapping(value, "dose")
    _closed(table, {"train", "test"}, "dose", noun="table")
    _required(table, {"train", "test"}, "dose")
    return DoseDescriptor(
        train=_parse_dose_statistics(table["train"], "dose.train"),
        test=_parse_dose_statistics(table["test"], "dose.test"),
    )


def _parse_dose_statistics(value: Any, path: str) -> DoseStatistics:
    table = _mapping(value, path)
    _closed(table, _DOSE_FIELDS, path)
    _required(table, _DOSE_FIELDS, path)
    saturation = _nonnegative_number(
        table["saturation_fraction"], f"{path}.saturation_fraction"
    )
    if saturation > 1.0:
        raise DatasetError(f"{path}.saturation_fraction must be in [0,1]")
    return DoseStatistics(
        counts_mean=_nonnegative_number(table["counts_mean"], f"{path}.counts_mean"),
        photons_per_image_min=_nonnegative_number(
            table["photons_per_image_min"], f"{path}.photons_per_image_min"
        ),
        photons_per_image_mean=_nonnegative_number(
            table["photons_per_image_mean"], f"{path}.photons_per_image_mean"
        ),
        max_observed_count=_nonnegative_int(
            table["max_observed_count"], f"{path}.max_observed_count"
        ),
        dtype_max=_nonnegative_int(table["dtype_max"], f"{path}.dtype_max"),
        saturation_fraction=saturation,
    )


def _resolve_path(
    value: Any,
    field: str,
    *,
    path_origin: PathOrigin,
    path_base: Path,
    repository_bundle_provenance: str | None = None,
) -> Path:
    text = _nonempty(value, field)
    raw = Path(text)
    if path_origin == "repository_root":
        if raw.is_absolute() or ".." in raw.parts:
            raise DatasetError(
                f"{field} must be repository-root-relative without '..' escape"
            )
        repository_root = path_base.resolve()
        lexical_path = repository_root / raw
        resolved = lexical_path.resolve()
        try:
            resolved.relative_to(repository_root)
            return resolved
        except ValueError:
            pass
        if repository_bundle_provenance is None:
            raise DatasetError(f"{field} must resolve under the repository root")
        provenance_raw = Path(repository_bundle_provenance)
        if provenance_raw.is_absolute() or ".." in provenance_raw.parts:
            raise DatasetError(
                "provenance must be repository-root-relative without '..' escape"
            )
        bundle_relative = provenance_raw.parent
        bundle_lexical = repository_root / bundle_relative
        if bundle_relative == Path(".") or not bundle_lexical.is_symlink():
            raise DatasetError(f"{field} must resolve under the repository root")
        try:
            path_within_bundle = raw.relative_to(bundle_relative)
        except ValueError as exc:
            raise DatasetError(
                f"{field} must remain under the configured bundle root"
            ) from exc
        nested = bundle_lexical
        for part in path_within_bundle.parts:
            nested /= part
            if nested.is_symlink():
                raise DatasetError(
                    f"{field} nested symlink is forbidden within relocated bundle root"
                )
        resolved_bundle_root = bundle_lexical.resolve()
        try:
            resolved.relative_to(resolved_bundle_root)
        except ValueError as exc:
            raise DatasetError(
                f"{field} must resolve under the relocated bundle root"
            ) from exc
        return resolved
    if raw.is_absolute():
        return raw.resolve()
    if ".." in raw.parts:
        raise DatasetError(
            f"{field} relative path must not escape the descriptor directory"
        )
    resolved = (path_base / raw).resolve()
    try:
        resolved.relative_to(path_base.resolve())
    except ValueError as exc:
        raise DatasetError(
            f"{field} relative path must resolve under the descriptor directory"
        ) from exc
    return resolved


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetError(f"{path} must be a table/mapping")
    if any(not isinstance(key, str) for key in value):
        raise DatasetError(f"{path} field names must be strings")
    return value


def _closed(
    value: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    path: str,
    *,
    noun: str = "field",
) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        raise DatasetError(f"{path} has unknown {noun}(s): {sorted(unknown)}")


def _required(
    value: Mapping[str, Any], required: set[str] | frozenset[str], path: str
) -> None:
    missing = set(required) - set(value)
    if missing:
        raise DatasetError(f"{path} is missing required fields: {sorted(missing)}")


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise DatasetError(f"{path} must be a nonempty stable id")
    return value


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{path} must be a nonempty string/path")
    return value


def _enum(value: Any, accepted: set[str], path: str) -> str:
    if not isinstance(value, str) or value not in accepted:
        raise DatasetError(f"{path} must be one of {sorted(accepted)}")
    return value


def _hash(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise DatasetError(f"{path} must be a lowercase SHA-256 hash")
    return value


def _positive_int(value: Any, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise DatasetError(
            f"{path} must be a positive integer (bool is not an integer)"
        )
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise DatasetError(
            f"{path} must be a nonnegative integer (bool is not an integer)"
        )
    return value


def _nonnegative_number(value: Any, path: str) -> float:
    if type(value) not in (int, float):
        raise DatasetError(
            f"{path} must be a finite nonnegative number (bool is not numeric)"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise DatasetError(f"{path} must be a finite nonnegative number")
    return number


def _positive_number(value: Any, path: str) -> float:
    number = _nonnegative_number(value, path)
    if number <= 0:
        raise DatasetError(f"{path} must be finite and positive")
    return number


def _detector_shape(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise DatasetError("detector_shape must contain two positive equal integers")
    if (
        any(type(item) is not int or item <= 0 for item in value)
        or value[0] != value[1]
    ):
        raise DatasetError("detector_shape must contain two positive equal integers")
    return int(value[0]), int(value[1])
