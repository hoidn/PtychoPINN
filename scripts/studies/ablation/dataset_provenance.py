"""Closed typed provenance parsers and claim checks for generic v1 and CI v2/v3."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Generic, Literal, TypeVar, cast

from .dataset_content import (
    ArtifactCache,
    _ContentValidation,
    deterministic_float_equal,
    probe_geometry_sha256,
    validate_positive_real_scalar_multiple,
)

from .dataset_schema import (
    TASK10_CI_COMPATIBILITY_MATERIALIZER_ID,
    TASK10_RAW_PROBE_GEOMETRY_KEY,
    DatasetError,
    DoseStatistics,
    MeasurementDomain,
    _closed,
    _detector_shape,
    _enum,
    _hash,
    _identifier,
    _mapping,
    _nonempty,
    _nonnegative_int,
    _parse_dose_statistics,
    _positive_int,
    _positive_number,
    _required,
)

_GIT_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "materializer_id",
        "materializer_version",
        "generator_commit",
        "expected_dataset_ids",
        "seeds",
        "source_objects",
        "coordinate_sets",
        "probe_geometries",
        "datasets",
    }
)
_V2_ROOT_FIELDS = _ROOT_FIELDS | {"materialization_profile"}
_DATASET_FIELDS = frozenset(
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
        "truth_key",
        "coords_convention",
        "detector_shape",
        "grouping_max_C",
        "probe_modes",
        "source_object_id",
        "coordinate_set_id",
        "probe_geometry_id",
        "dose_family_id",
        "base_dataset_id",
        "dose_multiplier",
        "files",
        "probe",
    }
)
_PROBE_FIELDS = frozenset(
    {
        "source",
        "calibration",
        "gauge",
        "mask_policy",
        "train_sha256",
        "test_sha256",
        "train_l2_norm",
        "test_l2_norm",
        "train_total_energy",
        "test_total_energy",
        "train_mode_energies",
        "test_mode_energies",
    }
)
_COORDINATE_FIELDS = frozenset(
    {
        "train_x_sha256",
        "train_y_sha256",
        "test_x_sha256",
        "test_y_sha256",
    }
)


@dataclass(frozen=True)
class SplitSeedsV1:
    train: int
    test: int


@dataclass(frozen=True)
class SeedsV1:
    object: int
    train_coordinates: int
    test_coordinates: int
    measurements: tuple[tuple[str, SplitSeedsV1], ...]


@dataclass(frozen=True)
class SourceObjectV1:
    sha256: str


@dataclass(frozen=True)
class CoordinateSetV1:
    train_x_sha256: str
    train_y_sha256: str
    test_x_sha256: str
    test_y_sha256: str


@dataclass(frozen=True)
class ProbeGeometryV1:
    sha256: str
    array_key: str | None


@dataclass(frozen=True)
class FilesV1:
    train: str
    test: str
    reference: str | None


@dataclass(frozen=True)
class ProbeRecordV1:
    source: str
    calibration: Literal["count_amplitude", "legacy_normalized"]
    gauge: Literal["physical_count_amplitude", "legacy_normalized"]
    mask_policy: Literal["model_config", "pre_masked"]
    train_sha256: str
    test_sha256: str
    train_l2_norm: float
    test_l2_norm: float
    train_total_energy: float
    test_total_energy: float
    train_mode_energies: tuple[float, ...]
    test_mode_energies: tuple[float, ...]


@dataclass(frozen=True)
class ProvenanceDatasetV1:
    id: str
    kind: Literal["synthetic", "experimental"]
    format: Literal["npz_mmap"]
    scale_contract_version: Literal["ci_intensity_v2", "legacy_v1"]
    measurement_domain: MeasurementDomain
    truth: Literal["object_truth", "reference_reconstruction", "none"]
    truth_location: Literal["embedded_test", "external_npz", "none"]
    measurement_key: str
    probe_key: str
    x_key: str
    y_key: str
    truth_key: str | None
    coords_convention: Literal["xy_pixels"]
    detector_shape: tuple[int, int]
    grouping_max_C: int
    probe_modes: int
    source_object_id: str | None
    coordinate_set_id: str
    probe_geometry_id: str
    dose_family_id: str | None
    base_dataset_id: str | None
    dose_multiplier: float | None
    target_counts_mean: float | None
    files: FilesV1
    probe: ProbeRecordV1
    dose_train: DoseStatistics | None
    dose_test: DoseStatistics | None


@dataclass(frozen=True)
class ProvenanceV1:
    schema_version: Literal[1]
    materializer_id: str
    materializer_version: int
    generator_commit: str
    expected_dataset_ids: tuple[str, ...]
    seeds: SeedsV1
    source_objects: tuple[tuple[str, SourceObjectV1], ...]
    coordinate_sets: tuple[tuple[str, CoordinateSetV1], ...]
    probe_geometries: tuple[tuple[str, ProbeGeometryV1], ...]
    datasets: tuple[tuple[str, ProvenanceDatasetV1], ...]

    def source_object_map(self) -> dict[str, SourceObjectV1]:
        return dict(self.source_objects)

    def coordinate_set_map(self) -> dict[str, CoordinateSetV1]:
        return dict(self.coordinate_sets)

    def probe_geometry_map(self) -> dict[str, ProbeGeometryV1]:
        return dict(self.probe_geometries)

    def dataset_map(self) -> dict[str, ProvenanceDatasetV1]:
        return dict(self.datasets)


CI_COMPATIBILITY_PROVENANCE_V2: Final = "ci_compatibility_provenance_v2"
CI_COMPATIBILITY_PROVENANCE_V3: Final = "ci_compatibility_provenance_v3"
CI_COMPATIBILITY_DATASET_IDS = (
    "deadleaves_ci_3p5m",
    "deadleaves_legacy_amp",
    "lines_ci_3p5m",
    "lines_legacy_amp",
)


def canonical_array_sha256(array: Any, dtype: Any | None = None) -> str:
    """Hash canonical bytes; dtype and shape are separate compatibility claims."""
    import numpy as np

    value = np.asarray(array, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class SourceObjectV2:
    generator: Literal["create_dead_leaves", "grid_lines_set_phi"]
    parameters: tuple[tuple[str, Any], ...]
    dtype: Literal["complex64"]
    shape: tuple[int, int]
    sha256: str


@dataclass(frozen=True)
class SourceObjectV3:
    generator: Literal["create_dead_leaves", "grid_lines_rectangular_v1"]
    parameters: tuple[tuple[str, Any], ...]
    dtype: Literal["complex64"]
    shape: tuple[int, int]
    sha256: str


_SchemaVersionT = TypeVar("_SchemaVersionT", bound=str)
_SourceObjectT = TypeVar("_SourceObjectT", SourceObjectV2, SourceObjectV3)


@dataclass(frozen=True)
class CoordinateSplitV2:
    count: int
    dtype: Literal["float32"]
    shape: tuple[int]
    x_sha256: str
    y_sha256: str


@dataclass(frozen=True)
class CoordinateSetV2:
    train: CoordinateSplitV2
    test: CoordinateSplitV2


@dataclass(frozen=True)
class ProbeGeometryV2:
    array_key: Literal["probeGeometry"]
    dtype: Literal["complex64"]
    shape: tuple[int, int, int]
    sha256: str


@dataclass(frozen=True)
class DatasetSplitV2:
    path: str
    file_sha256: str
    truth_sha256: str
    xcoords_sha256: str
    ycoords_sha256: str
    raw_probe_sha256: str
    stored_probe_sha256: str
    probe_scale: float
    stored_probe_l2_norm: float
    dose: DoseStatistics | None


@dataclass(frozen=True)
class ProvenanceDatasetV2:
    id: str
    family: Literal["deadleaves", "lines"]
    scale_contract_version: Literal["ci_intensity_v2", "legacy_v1"]
    measurement_domain: MeasurementDomain
    splits: tuple[tuple[str, DatasetSplitV2], ...]

    def split_map(self) -> dict[str, DatasetSplitV2]:
        return dict(self.splits)


@dataclass(frozen=True)
class ProvenanceV2(Generic[_SchemaVersionT, _SourceObjectT]):
    schema_version: _SchemaVersionT
    materializer_id: str
    materializer_version: int
    generator_commit: str
    materialization_profile: Literal["claim_grade", "fixture"]
    expected_dataset_ids: tuple[str, ...]
    seeds: SeedsV1
    source_objects: tuple[tuple[str, _SourceObjectT], ...]
    coordinate_sets: tuple[tuple[str, CoordinateSetV2], ...]
    probe_geometries: tuple[tuple[str, ProbeGeometryV2], ...]
    datasets: tuple[tuple[str, ProvenanceDatasetV2], ...]

    def source_object_map(self) -> dict[str, _SourceObjectT]:
        return dict(self.source_objects)

    def coordinate_set_map(self) -> dict[str, CoordinateSetV2]:
        return dict(self.coordinate_sets)

    def probe_geometry_map(self) -> dict[str, ProbeGeometryV2]:
        return dict(self.probe_geometries)

    def dataset_map(self) -> dict[str, ProvenanceDatasetV2]:
        return dict(self.datasets)


class ProvenanceV3(
    ProvenanceV2[Literal["ci_compatibility_provenance_v3"], SourceObjectV3]
):
    pass


def load_provenance_v1(
    path: Path, expected_sha256: str, cache: ArtifactCache
) -> ProvenanceV1:
    payload = cache.json(path, expected_sha256, "provenance_sha256")
    return parse_provenance_v1(payload)


def parse_provenance_v1(value: Any) -> ProvenanceV1:
    """Parse exactly one closed provenance-v1 representation into frozen records."""
    root = _mapping(value, "provenance")
    _closed(root, _ROOT_FIELDS, "provenance")
    _required(root, _ROOT_FIELDS, "provenance")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise DatasetError("provenance.schema_version must be integer 1")
    materializer_id = _identifier(root["materializer_id"], "provenance.materializer_id")
    materializer_version = _positive_int(
        root["materializer_version"], "provenance.materializer_version"
    )
    generator_commit = root["generator_commit"]
    if not isinstance(generator_commit, str) or not _GIT_COMMIT_RE.fullmatch(
        generator_commit
    ):
        raise DatasetError(
            "provenance.generator_commit must be a full lowercase 40- or 64-hex commit"
        )
    expected_ids = _parse_expected_ids(root["expected_dataset_ids"])
    is_task10 = materializer_id == TASK10_CI_COMPATIBILITY_MATERIALIZER_ID
    source_objects = _parse_source_objects(root["source_objects"])
    coordinate_sets = _parse_coordinate_sets(root["coordinate_sets"])
    probe_geometries = _parse_probe_geometries(
        root["probe_geometries"], require_raw_geometry=is_task10
    )
    records = _parse_dataset_records(root["datasets"], is_task10=is_task10)
    record_ids = tuple(key for key, _ in records)
    if record_ids != expected_ids:
        raise DatasetError(
            "provenance.datasets must exactly match sorted expected_dataset_ids"
        )
    seeds = _parse_seeds(root["seeds"], set(expected_ids))
    _validate_record_references(
        records, source_objects, coordinate_sets, probe_geometries
    )
    return ProvenanceV1(
        schema_version=1,
        materializer_id=materializer_id,
        materializer_version=materializer_version,
        generator_commit=generator_commit,
        expected_dataset_ids=expected_ids,
        seeds=seeds,
        source_objects=source_objects,
        coordinate_sets=coordinate_sets,
        probe_geometries=probe_geometries,
        datasets=records,
    )


def parse_provenance_v2(value: Any) -> ProvenanceV2:
    """Parse the fully closed CI compatibility two-family provenance schema."""
    root = _mapping(value, "provenance")
    _closed(root, _V2_ROOT_FIELDS, "provenance")
    _required(root, _V2_ROOT_FIELDS, "provenance")
    if root["schema_version"] != CI_COMPATIBILITY_PROVENANCE_V2:
        raise DatasetError(
            f"provenance.schema_version must be {CI_COMPATIBILITY_PROVENANCE_V2!r}"
        )
    materializer_id = _identifier(root["materializer_id"], "provenance.materializer_id")
    materializer_version = _positive_int(
        root["materializer_version"], "provenance.materializer_version"
    )
    commit = root["generator_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise DatasetError(
            "provenance.generator_commit must be a full lowercase 40-hex commit"
        )
    materialization_profile = cast(
        Literal["claim_grade", "fixture"],
        _enum(
            root["materialization_profile"],
            {"claim_grade", "fixture"},
            "provenance.materialization_profile",
        ),
    )
    expected_ids = _parse_expected_ids(root["expected_dataset_ids"])
    if expected_ids != CI_COMPATIBILITY_DATASET_IDS:
        raise DatasetError(
            "provenance.expected_dataset_ids must be exactly the four compatibility ids"
        )
    seeds = _parse_seeds(root["seeds"], set(expected_ids))
    sources = _parse_source_objects_v2(root["source_objects"], seeds.object)
    coordinates = _parse_coordinate_sets_v2(root["coordinate_sets"])
    geometries = _parse_probe_geometries_v2(root["probe_geometries"])
    datasets = _parse_datasets_v2(root["datasets"])
    if tuple(key for key, _ in datasets) != expected_ids:
        raise DatasetError(
            "provenance.datasets must exactly match expected_dataset_ids"
        )
    return ProvenanceV2(
        schema_version=CI_COMPATIBILITY_PROVENANCE_V2,
        materializer_id=materializer_id,
        materializer_version=materializer_version,
        generator_commit=commit,
        materialization_profile=materialization_profile,
        expected_dataset_ids=expected_ids,
        seeds=seeds,
        source_objects=sources,
        coordinate_sets=coordinates,
        probe_geometries=geometries,
        datasets=datasets,
    )


def parse_provenance_v3(value: Any) -> ProvenanceV3:
    """Parse bounded-lines CI compatibility provenance without widening v2."""
    root = _mapping(value, "provenance")
    _closed(root, _V2_ROOT_FIELDS, "provenance")
    _required(root, _V2_ROOT_FIELDS, "provenance")
    if root["schema_version"] != CI_COMPATIBILITY_PROVENANCE_V3:
        raise DatasetError(
            f"provenance.schema_version must be {CI_COMPATIBILITY_PROVENANCE_V3!r}"
        )
    materializer_id = _identifier(root["materializer_id"], "provenance.materializer_id")
    materializer_version = _positive_int(
        root["materializer_version"], "provenance.materializer_version"
    )
    if materializer_version != 3:
        raise DatasetError("provenance.materializer_version must be integer 3 for v3")
    commit = root["generator_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise DatasetError(
            "provenance.generator_commit must be a full lowercase 40-hex commit"
        )
    materialization_profile = cast(
        Literal["claim_grade", "fixture"],
        _enum(
            root["materialization_profile"],
            {"claim_grade", "fixture"},
            "provenance.materialization_profile",
        ),
    )
    expected_ids = _parse_expected_ids(root["expected_dataset_ids"])
    if expected_ids != CI_COMPATIBILITY_DATASET_IDS:
        raise DatasetError(
            "provenance.expected_dataset_ids must be exactly the four compatibility ids"
        )
    seeds = _parse_seeds(root["seeds"], set(expected_ids))
    sources = _parse_source_objects_v3(root["source_objects"], seeds.object)
    coordinates = _parse_coordinate_sets_v2(root["coordinate_sets"])
    geometries = _parse_probe_geometries_v2(root["probe_geometries"])
    datasets = _parse_datasets_v2(root["datasets"])
    if tuple(key for key, _ in datasets) != expected_ids:
        raise DatasetError(
            "provenance.datasets must exactly match expected_dataset_ids"
        )
    return ProvenanceV3(
        schema_version=CI_COMPATIBILITY_PROVENANCE_V3,
        materializer_id=materializer_id,
        materializer_version=materializer_version,
        generator_commit=commit,
        materialization_profile=materialization_profile,
        expected_dataset_ids=expected_ids,
        seeds=seeds,
        source_objects=sources,
        coordinate_sets=coordinates,
        probe_geometries=geometries,
        datasets=datasets,
    )


def load_provenance(
    path: Path, expected_sha256: str, cache: ArtifactCache
) -> ProvenanceV1 | ProvenanceV2 | ProvenanceV3:
    payload = cache.json(path, expected_sha256, "provenance_sha256")
    if (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == CI_COMPATIBILITY_PROVENANCE_V3
    ):
        return parse_provenance_v3(payload)
    if (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == CI_COMPATIBILITY_PROVENANCE_V2
    ):
        return parse_provenance_v2(payload)
    return parse_provenance_v1(payload)


def _exact_float(value: Any, path: str, *, positive: bool = False) -> float:
    if type(value) is not float or not __import__("math").isfinite(value):
        raise DatasetError(f"{path} must be a finite float")
    if positive and value <= 0:
        raise DatasetError(f"{path} must be a positive float")
    return value


def _shape(value: Any, length: int, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise DatasetError(f"{path} must be a {length}-element shape list")
    return tuple(_positive_int(item, path) for item in value)


def _parse_source_objects_v2(
    value: Any, object_seed: int
) -> tuple[tuple[str, SourceObjectV2], ...]:
    table = _mapping(value, "provenance.source_objects")
    if set(table) != {"deadleaves", "lines"}:
        raise DatasetError(
            "provenance.source_objects must contain exactly deadleaves and lines"
        )
    result = []
    for family in ("deadleaves", "lines"):
        path = f"provenance.source_objects.{family}"
        item = _mapping(table[family], path)
        fields = {"generator", "parameters", "dtype", "shape", "sha256"}
        _closed(item, fields, path)
        _required(item, fields, path)
        generator = _enum(
            item["generator"],
            {"create_dead_leaves", "grid_lines_set_phi"},
            f"{path}.generator",
        )
        expected_generator = (
            "create_dead_leaves" if family == "deadleaves" else "grid_lines_set_phi"
        )
        if generator != expected_generator:
            raise DatasetError(f"{path}.generator does not match object family")
        parameters = _parse_source_parameters_v2(
            family, item["parameters"], object_seed
        )
        if item["dtype"] != "complex64":
            raise DatasetError(f"{path}.dtype must be complex64")
        result.append(
            (
                family,
                SourceObjectV2(
                    generator=cast(Any, generator),
                    parameters=tuple(parameters.items()),
                    dtype="complex64",
                    shape=cast(
                        tuple[int, int], _shape(item["shape"], 2, f"{path}.shape")
                    ),
                    sha256=_hash(item["sha256"], f"{path}.sha256"),
                ),
            )
        )
    return tuple(result)


def _parse_source_objects_v3(
    value: Any, object_seed: int
) -> tuple[tuple[str, SourceObjectV3], ...]:
    table = _mapping(value, "provenance.source_objects")
    if set(table) != {"deadleaves", "lines"}:
        raise DatasetError(
            "provenance.source_objects must contain exactly deadleaves and lines"
        )
    result = []
    for family in ("deadleaves", "lines"):
        path = f"provenance.source_objects.{family}"
        item = _mapping(table[family], path)
        fields = {"generator", "parameters", "dtype", "shape", "sha256"}
        _closed(item, fields, path)
        _required(item, fields, path)
        expected_generator = (
            "create_dead_leaves"
            if family == "deadleaves"
            else "grid_lines_rectangular_v1"
        )
        if item["generator"] != expected_generator:
            raise DatasetError(f"{path}.generator does not match object family")
        parameters = (
            _parse_source_parameters_v2(family, item["parameters"], object_seed)
            if family == "deadleaves"
            else _parse_lines_parameters_v3(item["parameters"], object_seed)
        )
        if item["dtype"] != "complex64":
            raise DatasetError(f"{path}.dtype must be complex64")
        result.append(
            (
                family,
                SourceObjectV3(
                    generator=cast(Any, expected_generator),
                    parameters=tuple(parameters.items()),
                    dtype="complex64",
                    shape=cast(
                        tuple[int, int], _shape(item["shape"], 2, f"{path}.shape")
                    ),
                    sha256=_hash(item["sha256"], f"{path}.sha256"),
                ),
            )
        )
    return tuple(result)


def _parse_source_parameters_v2(
    family: str, value: Any, object_seed: int
) -> dict[str, Any]:
    path = f"provenance.source_objects.{family}.parameters"
    table = _mapping(value, path)
    if family == "deadleaves":
        fields = {
            "max_iters",
            "r_min_frac",
            "r_max_frac",
            "r_sigma",
            "phase_max",
            "seed",
        }
        _closed(table, fields, path)
        _required(table, fields, path)
        parsed = {
            "max_iters": _positive_int(table["max_iters"], f"{path}.max_iters"),
            "r_min_frac": _exact_float(
                table["r_min_frac"], f"{path}.r_min_frac", positive=True
            ),
            "r_max_frac": _exact_float(
                table["r_max_frac"], f"{path}.r_max_frac", positive=True
            ),
            "r_sigma": _exact_float(table["r_sigma"], f"{path}.r_sigma", positive=True),
            "phase_max": _exact_float(
                table["phase_max"], f"{path}.phase_max", positive=True
            ),
            "seed": _nonnegative_int(table["seed"], f"{path}.seed"),
        }
        if not parsed["r_min_frac"] < parsed["r_max_frac"] < 1:
            raise DatasetError(
                f"{path} radius fractions must satisfy 0 < min < max < 1"
            )
        expected = {
            "max_iters": 700,
            "r_min_frac": 0.02,
            "r_max_frac": 0.18,
            "r_sigma": 3.0,
            "phase_max": 0.5,
            "seed": object_seed,
        }
    else:
        fields = {
            "canvas_size",
            "object_resolution",
            "crop_start",
            "crop_stop",
            "nlines",
            "set_phi",
            "seed",
        }
        _closed(table, fields, path)
        _required(table, fields, path)
        parsed = {
            key: _positive_int(table[key], f"{path}.{key}")
            for key in (
                "canvas_size",
                "object_resolution",
                "crop_start",
                "crop_stop",
                "nlines",
            )
        }
        if table["set_phi"] is not True:
            raise DatasetError(f"{path}.set_phi must be true")
        parsed["set_phi"] = True
        parsed["seed"] = _nonnegative_int(table["seed"], f"{path}.seed")
        expected = dict(parsed)
        if (
            parsed["canvas_size"] != 2 * parsed["object_resolution"]
            or parsed["crop_start"] != parsed["object_resolution"] // 2
            or parsed["crop_stop"] != parsed["crop_start"] + parsed["object_resolution"]
            or parsed["nlines"] != 400
            or parsed["seed"] != object_seed
        ):
            raise DatasetError(
                f"{path} does not declare the canonical lines construction"
            )
    if parsed != expected:
        raise DatasetError(f"{path} does not match claim-grade parameters")
    return parsed


def _parse_lines_parameters_v3(value: Any, object_seed: int) -> dict[str, Any]:
    path = "provenance.source_objects.lines.parameters"
    table = _mapping(value, path)
    fields = {
        "canvas_size",
        "object_resolution",
        "crop_start",
        "crop_stop",
        "nlines",
        "mapping",
        "amplitude_min",
        "amplitude_max",
        "phase_min",
        "phase_max",
        "seed",
    }
    _closed(table, fields, path)
    _required(table, fields, path)
    parsed: dict[str, Any] = {
        key: _positive_int(table[key], f"{path}.{key}")
        for key in (
            "canvas_size",
            "object_resolution",
            "crop_start",
            "crop_stop",
            "nlines",
        )
    }
    parsed.update(
        mapping=table["mapping"],
        amplitude_min=_exact_float(table["amplitude_min"], f"{path}.amplitude_min"),
        amplitude_max=_exact_float(table["amplitude_max"], f"{path}.amplitude_max"),
        phase_min=_exact_float(table["phase_min"], f"{path}.phase_min"),
        phase_max=_exact_float(table["phase_max"], f"{path}.phase_max"),
        seed=_nonnegative_int(table["seed"], f"{path}.seed"),
    )
    expected = {
        "canvas_size": 2 * parsed["object_resolution"],
        "object_resolution": parsed["object_resolution"],
        "crop_start": parsed["object_resolution"] // 2,
        "crop_stop": parsed["object_resolution"] // 2 + parsed["object_resolution"],
        "nlines": 400,
        "mapping": "rectangular_v1",
        "amplitude_min": 0.3,
        "amplitude_max": 1.0,
        "phase_min": -0.5,
        "phase_max": 0.5,
        "seed": object_seed,
    }
    if parsed != expected:
        raise DatasetError(f"{path} does not match bounded rectangular parameters")
    return parsed


def _parse_coordinate_sets_v2(value: Any) -> tuple[tuple[str, CoordinateSetV2], ...]:
    table = _mapping(value, "provenance.coordinate_sets")
    if set(table) != {"shared_scan"}:
        raise DatasetError(
            "provenance.coordinate_sets must contain exactly shared_scan"
        )
    shared = _mapping(table["shared_scan"], "provenance.coordinate_sets.shared_scan")
    _closed(shared, {"train", "test"}, "provenance.coordinate_sets.shared_scan")
    _required(shared, {"train", "test"}, "provenance.coordinate_sets.shared_scan")
    splits = {}
    for split in ("train", "test"):
        path = f"provenance.coordinate_sets.shared_scan.{split}"
        item = _mapping(shared[split], path)
        fields = {"count", "dtype", "shape", "x_sha256", "y_sha256"}
        _closed(item, fields, path)
        _required(item, fields, path)
        count = _positive_int(item["count"], f"{path}.count")
        if item["dtype"] != "float32" or item["shape"] != [count]:
            raise DatasetError(f"{path} dtype/shape must be float32/[count]")
        splits[split] = CoordinateSplitV2(
            count,
            "float32",
            (count,),
            _hash(item["x_sha256"], f"{path}.x_sha256"),
            _hash(item["y_sha256"], f"{path}.y_sha256"),
        )
    return (("shared_scan", CoordinateSetV2(splits["train"], splits["test"])),)


def _parse_probe_geometries_v2(value: Any) -> tuple[tuple[str, ProbeGeometryV2], ...]:
    table = _mapping(value, "provenance.probe_geometries")
    if set(table) != {"raw_probe"}:
        raise DatasetError("provenance.probe_geometries must contain exactly raw_probe")
    path = "provenance.probe_geometries.raw_probe"
    item = _mapping(table["raw_probe"], path)
    fields = {"array_key", "dtype", "shape", "sha256"}
    _closed(item, fields, path)
    _required(item, fields, path)
    if item["array_key"] != "probeGeometry" or item["dtype"] != "complex64":
        raise DatasetError(f"{path} must declare probeGeometry complex64")
    return (
        (
            "raw_probe",
            ProbeGeometryV2(
                "probeGeometry",
                "complex64",
                cast(tuple[int, int, int], _shape(item["shape"], 3, f"{path}.shape")),
                _hash(item["sha256"], f"{path}.sha256"),
            ),
        ),
    )


def _parse_datasets_v2(value: Any) -> tuple[tuple[str, ProvenanceDatasetV2], ...]:
    table = _mapping(value, "provenance.datasets")
    if set(table) != set(CI_COMPATIBILITY_DATASET_IDS):
        raise DatasetError(
            "provenance.datasets must contain exactly the four compatibility ids"
        )
    result = []
    for dataset_id in CI_COMPATIBILITY_DATASET_IDS:
        path = f"provenance.datasets.{dataset_id}"
        item = _mapping(table[dataset_id], path)
        fields = {"family", "scale_contract_version", "measurement_domain", "splits"}
        _closed(item, fields, path)
        _required(item, fields, path)
        family = _enum(item["family"], {"deadleaves", "lines"}, f"{path}.family")
        scale = _enum(
            item["scale_contract_version"],
            {"ci_intensity_v2", "legacy_v1"},
            f"{path}.scale_contract_version",
        )
        domain = _enum(
            item["measurement_domain"],
            {"count_intensity", "normalized_amplitude"},
            f"{path}.measurement_domain",
        )
        expected_family = dataset_id.split("_", 1)[0]
        expected_pair = (
            ("ci_intensity_v2", "count_intensity")
            if dataset_id.endswith("ci_3p5m")
            else ("legacy_v1", "normalized_amplitude")
        )
        if family != expected_family or (scale, domain) != expected_pair:
            raise DatasetError(f"{path} identity does not match dataset id")
        splits_table = _mapping(item["splits"], f"{path}.splits")
        _closed(splits_table, {"train", "test"}, f"{path}.splits")
        _required(splits_table, {"train", "test"}, f"{path}.splits")
        splits = tuple(
            (
                split,
                _parse_dataset_split_v2(
                    splits_table[split],
                    f"{path}.splits.{split}",
                    domain == "count_intensity",
                ),
            )
            for split in ("train", "test")
        )
        result.append(
            (
                dataset_id,
                ProvenanceDatasetV2(
                    dataset_id,
                    cast(Any, family),
                    cast(Any, scale),
                    cast(Any, domain),
                    splits,
                ),
            )
        )
    return tuple(result)


def _parse_dataset_split_v2(value: Any, path: str, is_count: bool) -> DatasetSplitV2:
    item = _mapping(value, path)
    fields = {
        "path",
        "file_sha256",
        "truth_sha256",
        "xcoords_sha256",
        "ycoords_sha256",
        "raw_probe_sha256",
        "stored_probe_sha256",
        "probe_scale",
        "stored_probe_l2_norm",
    }
    if is_count:
        fields.add("dose")
    _closed(item, fields, path)
    _required(item, fields, path)
    file_path = _nonempty(item["path"], f"{path}.path")
    if Path(file_path).is_absolute() or ".." in Path(file_path).parts:
        raise DatasetError(f"{path}.path must be a relative contained path")
    return DatasetSplitV2(
        path=file_path,
        file_sha256=_hash(item["file_sha256"], f"{path}.file_sha256"),
        truth_sha256=_hash(item["truth_sha256"], f"{path}.truth_sha256"),
        xcoords_sha256=_hash(item["xcoords_sha256"], f"{path}.xcoords_sha256"),
        ycoords_sha256=_hash(item["ycoords_sha256"], f"{path}.ycoords_sha256"),
        raw_probe_sha256=_hash(item["raw_probe_sha256"], f"{path}.raw_probe_sha256"),
        stored_probe_sha256=_hash(
            item["stored_probe_sha256"], f"{path}.stored_probe_sha256"
        ),
        probe_scale=_exact_float(
            item["probe_scale"], f"{path}.probe_scale", positive=True
        ),
        stored_probe_l2_norm=_exact_float(
            item["stored_probe_l2_norm"], f"{path}.stored_probe_l2_norm", positive=True
        ),
        dose=_parse_dose_statistics(item["dose"], f"{path}.dose") if is_count else None,
    )


def _parse_expected_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DatasetError("provenance.expected_dataset_ids must be a list of ids")
    parsed = tuple(
        _identifier(item, "provenance expected dataset id") for item in value
    )
    if parsed != tuple(sorted(set(parsed))):
        raise DatasetError("provenance.expected_dataset_ids must be sorted and unique")
    return parsed


def _parse_seeds(value: Any, dataset_ids: set[str]) -> SeedsV1:
    table = _mapping(value, "provenance.seeds")
    fields = {"object", "train_coordinates", "test_coordinates", "measurements"}
    _closed(table, fields, "provenance.seeds")
    _required(table, fields, "provenance.seeds")
    measurements = _mapping(table["measurements"], "provenance.seeds.measurements")
    if set(measurements) != dataset_ids:
        raise DatasetError(
            "provenance.seeds.measurements ids must exactly match provenance.datasets"
        )
    parsed_measurements: list[tuple[str, SplitSeedsV1]] = []
    for dataset_id in sorted(measurements):
        split = _mapping(
            measurements[dataset_id], f"provenance.seeds.measurements.{dataset_id}"
        )
        _closed(split, {"train", "test"}, f"provenance.seeds.measurements.{dataset_id}")
        _required(
            split, {"train", "test"}, f"provenance.seeds.measurements.{dataset_id}"
        )
        parsed_measurements.append(
            (
                dataset_id,
                SplitSeedsV1(
                    train=_nonnegative_int(
                        split["train"], "provenance measurement seed train"
                    ),
                    test=_nonnegative_int(
                        split["test"], "provenance measurement seed test"
                    ),
                ),
            )
        )
    return SeedsV1(
        object=_nonnegative_int(table["object"], "provenance.seeds.object"),
        train_coordinates=_nonnegative_int(
            table["train_coordinates"], "provenance.seeds.train_coordinates"
        ),
        test_coordinates=_nonnegative_int(
            table["test_coordinates"], "provenance.seeds.test_coordinates"
        ),
        measurements=tuple(parsed_measurements),
    )


def _parse_source_objects(value: Any) -> tuple[tuple[str, SourceObjectV1], ...]:
    table = _mapping(value, "provenance.source_objects")
    parsed: list[tuple[str, SourceObjectV1]] = []
    for source_id in sorted(table):
        _identifier(source_id, "provenance source object id")
        item = _mapping(table[source_id], f"provenance.source_objects.{source_id}")
        _closed(item, {"sha256"}, f"provenance.source_objects.{source_id}")
        _required(item, {"sha256"}, f"provenance.source_objects.{source_id}")
        parsed.append(
            (source_id, SourceObjectV1(_hash(item["sha256"], "source object sha256")))
        )
    return tuple(parsed)


def _parse_coordinate_sets(value: Any) -> tuple[tuple[str, CoordinateSetV1], ...]:
    table = _mapping(value, "provenance.coordinate_sets")
    if not table:
        raise DatasetError("provenance.coordinate_sets must not be empty")
    parsed: list[tuple[str, CoordinateSetV1]] = []
    for coordinate_id in sorted(table):
        _identifier(coordinate_id, "provenance coordinate set id")
        item = _mapping(
            table[coordinate_id], f"provenance.coordinate_sets.{coordinate_id}"
        )
        _closed(item, _COORDINATE_FIELDS, f"provenance.coordinate_sets.{coordinate_id}")
        _required(
            item, _COORDINATE_FIELDS, f"provenance.coordinate_sets.{coordinate_id}"
        )
        values = {
            field: _hash(item[field], f"coordinate {field}")
            for field in _COORDINATE_FIELDS
        }
        parsed.append((coordinate_id, CoordinateSetV1(**values)))
    return tuple(parsed)


def _parse_probe_geometries(
    value: Any, *, require_raw_geometry: bool
) -> tuple[tuple[str, ProbeGeometryV1], ...]:
    table = _mapping(value, "provenance.probe_geometries")
    if not table:
        raise DatasetError("provenance.probe_geometries must not be empty")
    fields = {"sha256", "array_key"} if require_raw_geometry else {"sha256"}
    parsed: list[tuple[str, ProbeGeometryV1]] = []
    for geometry_id in sorted(table):
        _identifier(geometry_id, "provenance probe geometry id")
        item = _mapping(
            table[geometry_id], f"provenance.probe_geometries.{geometry_id}"
        )
        _closed(item, fields, f"provenance.probe_geometries.{geometry_id}")
        _required(item, fields, f"provenance.probe_geometries.{geometry_id}")
        array_key = None
        if require_raw_geometry:
            if item["array_key"] != TASK10_RAW_PROBE_GEOMETRY_KEY:
                raise DatasetError(
                    "Task10 provenance probe geometry array_key must be "
                    f"{TASK10_RAW_PROBE_GEOMETRY_KEY!r}"
                )
            array_key = TASK10_RAW_PROBE_GEOMETRY_KEY
        parsed.append(
            (
                geometry_id,
                ProbeGeometryV1(
                    _hash(item["sha256"], "probe geometry sha256"), array_key
                ),
            )
        )
    return tuple(parsed)


def _parse_dataset_records(
    value: Any, *, is_task10: bool
) -> tuple[tuple[str, ProvenanceDatasetV1], ...]:
    table = _mapping(value, "provenance.datasets")
    return tuple(
        (
            record_id,
            _parse_dataset_record(record_id, table[record_id], is_task10=is_task10),
        )
        for record_id in sorted(table)
    )


def _parse_dataset_record(
    record_id: str, value: Any, *, is_task10: bool
) -> ProvenanceDatasetV1:
    _identifier(record_id, "provenance dataset id")
    record = _mapping(value, f"provenance.datasets.{record_id}")
    domain = record.get("measurement_domain")
    expected = set(_DATASET_FIELDS)
    if domain == "count_intensity":
        expected.add("dose")
        if is_task10:
            expected.add("target_counts_mean")
    _closed(record, expected, f"provenance.datasets.{record_id}")
    _required(record, expected, f"provenance.datasets.{record_id}")
    identity = _parse_record_identity(record)
    lineage = _parse_record_lineage(record, identity[4], is_task10=is_task10)
    files = _parse_files(record["files"], identity[5])
    probe = _parse_probe(record["probe"], identity[3], identity[7])
    dose_train, dose_test = _parse_record_dose(record, identity[3])
    return ProvenanceDatasetV1(
        id=record_id,
        kind=identity[0],
        format="npz_mmap",
        scale_contract_version=identity[2],
        measurement_domain=identity[3],
        truth=identity[4],
        truth_location=identity[5],
        measurement_key=_nonempty(
            record["measurement_key"], "provenance.measurement_key"
        ),
        probe_key=_nonempty(record["probe_key"], "provenance.probe_key"),
        x_key=_nonempty(record["x_key"], "provenance.x_key"),
        y_key=_nonempty(record["y_key"], "provenance.y_key"),
        truth_key=identity[6],
        coords_convention="xy_pixels",
        detector_shape=_detector_shape(record["detector_shape"]),
        grouping_max_C=_positive_int(
            record["grouping_max_C"], "provenance.grouping_max_C"
        ),
        probe_modes=identity[7],
        source_object_id=lineage[0],
        coordinate_set_id=lineage[1],
        probe_geometry_id=lineage[2],
        dose_family_id=lineage[3],
        base_dataset_id=lineage[4],
        dose_multiplier=lineage[5],
        target_counts_mean=lineage[6],
        files=files,
        probe=probe,
        dose_train=dose_train,
        dose_test=dose_test,
    )


def _parse_record_identity(
    record: Mapping[str, Any],
) -> tuple[
    Literal["synthetic", "experimental"],
    Literal["npz_mmap"],
    Literal["ci_intensity_v2", "legacy_v1"],
    MeasurementDomain,
    Literal["object_truth", "reference_reconstruction", "none"],
    Literal["embedded_test", "external_npz", "none"],
    str | None,
    int,
]:
    kind = cast(
        Literal["synthetic", "experimental"],
        _enum(record["kind"], {"synthetic", "experimental"}, "provenance.kind"),
    )
    _enum(record["format"], {"npz_mmap"}, "provenance.format")
    scale = cast(
        Literal["ci_intensity_v2", "legacy_v1"],
        _enum(
            record["scale_contract_version"],
            {"ci_intensity_v2", "legacy_v1"},
            "provenance.scale_contract_version",
        ),
    )
    domain = cast(
        MeasurementDomain,
        _enum(
            record["measurement_domain"],
            {"count_intensity", "normalized_amplitude"},
            "provenance.measurement_domain",
        ),
    )
    if (scale, domain) not in {
        ("ci_intensity_v2", "count_intensity"),
        ("legacy_v1", "normalized_amplitude"),
    }:
        raise DatasetError("provenance profile/domain is not a supported pair")
    truth = cast(
        Literal["object_truth", "reference_reconstruction", "none"],
        _enum(
            record["truth"],
            {"object_truth", "reference_reconstruction", "none"},
            "provenance.truth",
        ),
    )
    location = cast(
        Literal["embedded_test", "external_npz", "none"],
        _enum(
            record["truth_location"],
            {"embedded_test", "external_npz", "none"},
            "provenance.truth_location",
        ),
    )
    if kind == "synthetic" and truth != "object_truth":
        raise DatasetError("provenance synthetic datasets require object_truth")
    if (truth == "none") != (location == "none"):
        raise DatasetError("provenance truth and truth_location disagree")
    truth_key = (
        None
        if truth == "none"
        else _nonempty(record["truth_key"], "provenance.truth_key")
    )
    if truth == "none" and record["truth_key"] is not None:
        raise DatasetError("provenance truth_key must be null when truth=none")
    _enum(record["coords_convention"], {"xy_pixels"}, "provenance.coords_convention")
    probe_modes = _positive_int(record["probe_modes"], "provenance.probe_modes")
    return kind, "npz_mmap", scale, domain, truth, location, truth_key, probe_modes


def _parse_record_lineage(
    record: Mapping[str, Any], truth: str, *, is_task10: bool
) -> tuple[str | None, str, str, str | None, str | None, float | None, float | None]:
    source = record["source_object_id"]
    source_id = (
        _identifier(source, "provenance.source_object_id")
        if truth == "object_truth"
        else None
    )
    if truth != "object_truth" and source is not None:
        raise DatasetError(
            "provenance source_object_id must be null unless truth=object_truth"
        )
    coordinate_id = _identifier(
        record["coordinate_set_id"], "provenance.coordinate_set_id"
    )
    geometry_id = _identifier(
        record["probe_geometry_id"], "provenance.probe_geometry_id"
    )
    if record["measurement_domain"] == "count_intensity":
        family = _identifier(record["dose_family_id"], "provenance.dose_family_id")
        base = _identifier(record["base_dataset_id"], "provenance.base_dataset_id")
        multiplier = _positive_number(
            record["dose_multiplier"], "provenance.dose_multiplier"
        )
        target = (
            _positive_number(
                record["target_counts_mean"], "provenance.target_counts_mean"
            )
            if is_task10
            else None
        )
        return source_id, coordinate_id, geometry_id, family, base, multiplier, target
    if any(
        record[field] is not None
        for field in ("dose_family_id", "base_dataset_id", "dose_multiplier")
    ):
        raise DatasetError(
            "normalized-amplitude provenance forbids dose family/twin identity"
        )
    return source_id, coordinate_id, geometry_id, None, None, None, None


def _parse_files(value: Any, location: str) -> FilesV1:
    table = _mapping(value, "provenance.files")
    expected = {"train", "test"} | (
        {"reference"} if location == "external_npz" else set()
    )
    _closed(table, expected, "provenance.files")
    _required(table, expected, "provenance.files")
    return FilesV1(
        train=_hash(table["train"], "provenance.files.train"),
        test=_hash(table["test"], "provenance.files.test"),
        reference=_hash(table["reference"], "provenance.files.reference")
        if location == "external_npz"
        else None,
    )


def _parse_probe(value: Any, domain: str, probe_modes: int) -> ProbeRecordV1:
    table = _mapping(value, "provenance.probe")
    _closed(table, _PROBE_FIELDS, "provenance.probe")
    _required(table, _PROBE_FIELDS, "provenance.probe")
    calibration = cast(
        Literal["count_amplitude", "legacy_normalized"],
        _enum(
            table["calibration"],
            {"count_amplitude", "legacy_normalized"},
            "provenance.probe.calibration",
        ),
    )
    gauge = cast(
        Literal["physical_count_amplitude", "legacy_normalized"],
        _enum(
            table["gauge"],
            {"physical_count_amplitude", "legacy_normalized"},
            "provenance.probe.gauge",
        ),
    )
    expected_pair = (
        ("count_amplitude", "physical_count_amplitude")
        if domain == "count_intensity"
        else ("legacy_normalized", "legacy_normalized")
    )
    if (calibration, gauge) != expected_pair:
        raise DatasetError("provenance probe calibration/gauge does not match domain")
    train_modes = _parse_mode_energies(
        table["train_mode_energies"], probe_modes, "train"
    )
    test_modes = _parse_mode_energies(table["test_mode_energies"], probe_modes, "test")
    return ProbeRecordV1(
        source=_nonempty(table["source"], "provenance.probe.source"),
        calibration=calibration,
        gauge=gauge,
        mask_policy=cast(
            Literal["model_config", "pre_masked"],
            _enum(
                table["mask_policy"],
                {"model_config", "pre_masked"},
                "provenance.probe.mask_policy",
            ),
        ),
        train_sha256=_hash(table["train_sha256"], "provenance.probe.train_sha256"),
        test_sha256=_hash(table["test_sha256"], "provenance.probe.test_sha256"),
        train_l2_norm=_positive_number(
            table["train_l2_norm"], "provenance.probe.train_l2_norm"
        ),
        test_l2_norm=_positive_number(
            table["test_l2_norm"], "provenance.probe.test_l2_norm"
        ),
        train_total_energy=_positive_number(
            table["train_total_energy"], "provenance.probe.train_total_energy"
        ),
        test_total_energy=_positive_number(
            table["test_total_energy"], "provenance.probe.test_total_energy"
        ),
        train_mode_energies=train_modes,
        test_mode_energies=test_modes,
    )


def _parse_mode_energies(value: Any, probe_modes: int, split: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != probe_modes:
        raise DatasetError(
            f"provenance probe {split}_mode_energies must match probe_modes"
        )
    return tuple(
        _positive_number(item, f"provenance.probe.{split}_mode_energies")
        for item in value
    )


def _parse_record_dose(
    record: Mapping[str, Any], domain: str
) -> tuple[DoseStatistics | None, DoseStatistics | None]:
    if domain != "count_intensity":
        return None, None
    table = _mapping(record["dose"], "provenance.dose")
    _closed(table, {"train", "test"}, "provenance.dose", noun="table")
    _required(table, {"train", "test"}, "provenance.dose")
    return (
        _parse_dose_statistics(table["train"], "provenance.dose.train"),
        _parse_dose_statistics(table["test"], "provenance.dose.test"),
    )


def _validate_record_references(
    records: tuple[tuple[str, ProvenanceDatasetV1], ...],
    sources: tuple[tuple[str, SourceObjectV1], ...],
    coordinates: tuple[tuple[str, CoordinateSetV1], ...],
    geometries: tuple[tuple[str, ProbeGeometryV1], ...],
) -> None:
    source_ids = {key for key, _ in sources}
    coordinate_ids = {key for key, _ in coordinates}
    geometry_ids = {key for key, _ in geometries}
    for _, record in records:
        if (
            record.source_object_id is not None
            and record.source_object_id not in source_ids
        ):
            raise DatasetError("provenance source_object_id is not declared at root")
        if record.coordinate_set_id not in coordinate_ids:
            raise DatasetError("provenance coordinate_set_id is not declared at root")
        if record.probe_geometry_id not in geometry_ids:
            raise DatasetError("provenance probe_geometry_id is not declared at root")


def provenance_to_jsonable(provenance: ProvenanceV1) -> dict[str, Any]:
    return {
        "schema_version": provenance.schema_version,
        "materializer_id": provenance.materializer_id,
        "materializer_version": provenance.materializer_version,
        "generator_commit": provenance.generator_commit,
        "expected_dataset_ids": list(provenance.expected_dataset_ids),
        "seeds": _seeds_to_jsonable(provenance.seeds),
        "source_objects": {
            key: {"sha256": item.sha256} for key, item in provenance.source_objects
        },
        "coordinate_sets": {
            key: _coordinate_to_jsonable(item)
            for key, item in provenance.coordinate_sets
        },
        "probe_geometries": {
            key: _geometry_to_jsonable(item)
            for key, item in provenance.probe_geometries
        },
        "datasets": {
            key: _record_to_jsonable(item) for key, item in provenance.datasets
        },
    }


def _seeds_to_jsonable(seeds: SeedsV1) -> dict[str, Any]:
    return {
        "object": seeds.object,
        "train_coordinates": seeds.train_coordinates,
        "test_coordinates": seeds.test_coordinates,
        "measurements": {
            key: {"train": item.train, "test": item.test}
            for key, item in seeds.measurements
        },
    }


def _coordinate_to_jsonable(item: CoordinateSetV1) -> dict[str, str]:
    return {
        "train_x_sha256": item.train_x_sha256,
        "train_y_sha256": item.train_y_sha256,
        "test_x_sha256": item.test_x_sha256,
        "test_y_sha256": item.test_y_sha256,
    }


def _geometry_to_jsonable(item: ProbeGeometryV1) -> dict[str, str]:
    result = {"sha256": item.sha256}
    if item.array_key is not None:
        result["array_key"] = item.array_key
    return result


def _record_to_jsonable(record: ProvenanceDatasetV1) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": record.kind,
        "format": record.format,
        "scale_contract_version": record.scale_contract_version,
        "measurement_domain": record.measurement_domain,
        "truth": record.truth,
        "truth_location": record.truth_location,
        "measurement_key": record.measurement_key,
        "probe_key": record.probe_key,
        "x_key": record.x_key,
        "y_key": record.y_key,
        "truth_key": record.truth_key,
        "coords_convention": record.coords_convention,
        "detector_shape": list(record.detector_shape),
        "grouping_max_C": record.grouping_max_C,
        "probe_modes": record.probe_modes,
        "source_object_id": record.source_object_id,
        "coordinate_set_id": record.coordinate_set_id,
        "probe_geometry_id": record.probe_geometry_id,
        "dose_family_id": record.dose_family_id,
        "base_dataset_id": record.base_dataset_id,
        "dose_multiplier": record.dose_multiplier,
        "files": {"train": record.files.train, "test": record.files.test},
        "probe": _probe_to_jsonable(record.probe),
    }
    if record.files.reference is not None:
        result["files"]["reference"] = record.files.reference
    if record.target_counts_mean is not None:
        result["target_counts_mean"] = record.target_counts_mean
    if record.dose_train is not None and record.dose_test is not None:
        result["dose"] = {
            "train": record.dose_train.to_dict(),
            "test": record.dose_test.to_dict(),
        }
    return result


def _probe_to_jsonable(probe: ProbeRecordV1) -> dict[str, Any]:
    return {
        "source": probe.source,
        "calibration": probe.calibration,
        "gauge": probe.gauge,
        "mask_policy": probe.mask_policy,
        "train_sha256": probe.train_sha256,
        "test_sha256": probe.test_sha256,
        "train_l2_norm": probe.train_l2_norm,
        "test_l2_norm": probe.test_l2_norm,
        "train_total_energy": probe.train_total_energy,
        "test_total_energy": probe.test_total_energy,
        "train_mode_energies": list(probe.train_mode_energies),
        "test_mode_energies": list(probe.test_mode_energies),
    }


def validate_provenance_claims(
    provenance: ProvenanceV1,
    contents: Mapping[str, _ContentValidation],
) -> None:
    """Bind every typed provenance record to its descriptor and artifact content."""
    records = provenance.dataset_map()
    if set(records) != set(contents):
        raise DatasetError(
            "provenance.datasets must exactly match expected_dataset_ids and supplied ids"
        )
    sources = provenance.source_object_map()
    coordinates = provenance.coordinate_set_map()
    geometries = provenance.probe_geometry_map()
    for dataset_id, content in contents.items():
        record = records[dataset_id]
        _validate_identity_claims(dataset_id, content, record)
        _validate_probe_claims(dataset_id, content, record)
        _validate_coordinate_and_truth_claims(
            dataset_id, content, record, sources, coordinates
        )
        _validate_geometry_claims(dataset_id, content, record, geometries)
        _validate_dose_claims(dataset_id, content, record)


def _validate_identity_claims(
    dataset_id: str, content: _ContentValidation, record: ProvenanceDatasetV1
) -> None:
    descriptor = content.descriptor
    claims = (
        ("kind", record.kind, descriptor.kind),
        ("format", record.format, descriptor.format),
        (
            "scale_contract_version",
            record.scale_contract_version,
            descriptor.scale_contract_version,
        ),
        (
            "measurement_domain",
            record.measurement_domain,
            descriptor.measurement_domain,
        ),
        ("truth", record.truth, descriptor.truth),
        ("truth_location", record.truth_location, descriptor.truth_location),
        ("measurement_key", record.measurement_key, descriptor.measurement_key),
        ("probe_key", record.probe_key, descriptor.probe_key),
        ("x_key", record.x_key, descriptor.x_key),
        ("y_key", record.y_key, descriptor.y_key),
        ("truth_key", record.truth_key, descriptor.truth_key),
        ("coords_convention", record.coords_convention, descriptor.coords_convention),
        ("detector_shape", record.detector_shape, descriptor.detector_shape),
        ("grouping_max_C", record.grouping_max_C, descriptor.grouping_max_C),
        ("probe_modes", record.probe_modes, descriptor.probe_modes),
    )
    for field, declared, expected in claims:
        if declared != expected:
            raise DatasetError(
                f"provenance dataset {dataset_id!r} {field} does not agree with descriptor"
            )
    expected_files = FilesV1(
        descriptor.train_sha256,
        descriptor.test_sha256,
        descriptor.reference_sha256,
    )
    if record.files != expected_files:
        raise DatasetError(
            f"provenance dataset {dataset_id!r} file hashes do not agree with descriptor"
        )


def _validate_probe_claims(
    dataset_id: str, content: _ContentValidation, record: ProvenanceDatasetV1
) -> None:
    descriptor = content.descriptor
    probe = record.probe
    train_hash = descriptor.probe.sha256 or descriptor.probe.train_sha256
    test_hash = descriptor.probe.sha256 or descriptor.probe.test_sha256
    claims = (
        ("source", probe.source, descriptor.probe.source),
        ("calibration", probe.calibration, descriptor.probe.calibration),
        ("gauge", probe.gauge, descriptor.probe.gauge),
        ("mask_policy", probe.mask_policy, descriptor.probe.mask_policy),
        ("train_sha256", probe.train_sha256, train_hash),
        ("test_sha256", probe.test_sha256, test_hash),
    )
    for field, declared, expected in claims:
        if declared != expected:
            raise DatasetError(
                f"provenance dataset {dataset_id!r} probe {field} does not agree with descriptor"
            )
    for split, split_content in (("train", content.train), ("test", content.test)):
        _compare_deterministic_float(
            getattr(probe, f"{split}_l2_norm"),
            split_content.probe_l2_norm,
            f"{dataset_id}.probe.{split}_l2_norm",
        )
        _compare_deterministic_float(
            getattr(probe, f"{split}_total_energy"),
            split_content.probe_total_energy,
            f"{dataset_id}.probe.{split}_total_energy",
        )
        declared_modes = getattr(probe, f"{split}_mode_energies")
        if len(declared_modes) != len(split_content.probe_mode_energies):
            raise DatasetError(f"provenance {dataset_id} probe mode count mismatch")
        for index, observed in enumerate(split_content.probe_mode_energies):
            _compare_deterministic_float(
                declared_modes[index],
                observed,
                f"{dataset_id}.probe.{split}_mode_energies[{index}]",
            )


def _compare_deterministic_float(declared: float, observed: float, path: str) -> None:
    if not deterministic_float_equal(declared, observed):
        raise DatasetError(
            f"provenance {path} mismatch: declared {declared}, observed {observed}"
        )


def _validate_coordinate_and_truth_claims(
    dataset_id: str,
    content: _ContentValidation,
    record: ProvenanceDatasetV1,
    sources: Mapping[str, SourceObjectV1],
    coordinates: Mapping[str, CoordinateSetV1],
) -> None:
    expected_coordinates = CoordinateSetV1(
        content.train.x_sha256,
        content.train.y_sha256,
        content.test.x_sha256,
        content.test.y_sha256,
    )
    if coordinates[record.coordinate_set_id] != expected_coordinates:
        raise DatasetError(
            f"provenance coordinate digests for dataset {dataset_id!r} do not agree with NPZ content"
        )
    if content.descriptor.truth == "object_truth":
        assert record.source_object_id is not None
        if (
            content.truth_sha256 is None
            or sources[record.source_object_id].sha256 != content.truth_sha256
        ):
            raise DatasetError(
                f"provenance source object digest for dataset {dataset_id!r} does not agree with truth content"
            )


def _validate_geometry_claims(
    dataset_id: str,
    content: _ContentValidation,
    record: ProvenanceDatasetV1,
    geometries: Mapping[str, ProbeGeometryV1],
) -> None:
    geometry = geometries[record.probe_geometry_id]
    if geometry.array_key is not None:
        for split, split_content in (("train", content.train), ("test", content.test)):
            if split_content.raw_probe_geometry_sha256 != geometry.sha256:
                raise DatasetError(
                    f"dataset {dataset_id!r} {split} raw probe geometry digest does not match Task10 provenance"
                )
        return
    if geometry.sha256 != probe_geometry_sha256(content.train.probe):
        raise DatasetError(
            f"provenance probe geometry digest for dataset {dataset_id!r} does not agree with train probe geometry"
        )
    validate_positive_real_scalar_multiple(
        content.test.probe,
        content.train.probe,
        label=f"dataset {dataset_id!r} train/test probe geometry",
    )


def _validate_dose_claims(
    dataset_id: str, content: _ContentValidation, record: ProvenanceDatasetV1
) -> None:
    descriptor = content.descriptor
    expected_train = descriptor.dose.train if descriptor.dose is not None else None
    expected_test = descriptor.dose.test if descriptor.dose is not None else None
    if record.dose_train != expected_train or record.dose_test != expected_test:
        raise DatasetError(
            f"provenance dose for dataset {dataset_id!r} does not agree with descriptor"
        )
