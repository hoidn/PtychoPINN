from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from tests.studies.ablation_dataset_fixtures import (
    _api,
    _array_sha256,
    _bundle,
    _file_sha256,
    _mutate_provenance,
    _refresh_provenance,
    _rewrite_split,
)


Descriptor = dict[str, Any]
SplitMutation = Callable[[dict[str, np.ndarray]], None]


def _rewrite_train(
    root: Path, descriptor: Descriptor, mutation: SplitMutation
) -> Descriptor:
    _rewrite_split(root, descriptor, "train", mutation)
    return descriptor


def _rewrite_test(
    root: Path, descriptor: Descriptor, mutation: SplitMutation
) -> Descriptor:
    _rewrite_split(root, descriptor, "test", mutation)
    return descriptor


def _legacy_rewrite(root: Path, mutation: SplitMutation) -> Descriptor:
    return _rewrite_train(root, _bundle(root, domain="normalized_amplitude"), mutation)


def _set_array(
    key: str, transform: Callable[[np.ndarray], np.ndarray]
) -> SplitMutation:
    def mutation(payload: dict[str, np.ndarray]) -> None:
        payload[key] = transform(payload[key])

    return mutation


def _remove_array(key: str) -> SplitMutation:
    def mutation(payload: dict[str, np.ndarray]) -> None:
        del payload[key]

    return mutation


def _set_descriptor(
    descriptor: Descriptor, path: tuple[str, ...], value: Any
) -> Descriptor:
    target: dict[str, Any] = descriptor
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return descriptor


def _increment_descriptor(descriptor: Descriptor, path: tuple[str, ...]) -> Descriptor:
    target: dict[str, Any] = descriptor
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] += 1.0
    return descriptor


def _negative_measurement(payload: dict[str, np.ndarray]) -> None:
    payload["diff3d"] = payload["diff3d"].astype(np.int64)
    payload["diff3d"][0, 0, 0] = -1


def _bad_provenance_json(root: Path, descriptor: Descriptor) -> Descriptor:
    path = root / descriptor["provenance"]
    path.write_text("not json", encoding="utf-8")
    descriptor["provenance_sha256"] = _file_sha256(path)
    return descriptor


def _mutate_provenance_record(
    root: Path, descriptor: Descriptor, field: str, value: object
) -> Descriptor:
    def mutation(payload: dict[str, Any]) -> None:
        payload["datasets"][descriptor["_id"]][field] = value

    _mutate_provenance(root, descriptor, mutation)
    return descriptor


def _refresh(root: Path, descriptor: Descriptor) -> Descriptor:
    _refresh_provenance(root, descriptor)
    return descriptor


def _invalid_content_mutations() -> dict[str, Callable[[Path, Descriptor], Descriptor]]:
    return {
        "file_hash": lambda _r, d: _set_descriptor(d, ("train_sha256",), "0" * 64),
        "missing_key": lambda r, d: _rewrite_train(r, d, _remove_array("diff3d")),
        "legacy_orientation": lambda r, d: _rewrite_train(
            r, d, lambda p: p.__setitem__("diff3d", np.ones((2, 2, 3), dtype=np.uint32))
        ),
        "channel_last": lambda r, d: _rewrite_train(
            r, d, _set_array("diff3d", lambda value: value[..., None])
        ),
        "coordinate_length": lambda r, d: _rewrite_train(
            r, d, _set_array("xcoords", lambda value: value[:-1])
        ),
        "coordinate_nonfinite": lambda r, d: _rewrite_train(
            r, d, lambda p: p["xcoords"].__setitem__(0, np.nan)
        ),
        "measurement_nonfinite": lambda r, _d: _legacy_rewrite(
            r, lambda p: p["diff3d"].__setitem__((0, 0, 0), np.nan)
        ),
        "count_float": lambda r, d: _rewrite_train(
            r, d, _set_array("diff3d", lambda value: value.astype(np.float32))
        ),
        "count_negative": lambda r, d: _rewrite_train(r, d, _negative_measurement),
        "amplitude_integer": lambda r, _d: _legacy_rewrite(
            r,
            _set_array("diff3d", lambda value: np.ones_like(value, dtype=np.uint16)),
        ),
        "probe_shape": lambda r, d: _rewrite_train(
            r,
            d,
            lambda p: p.__setitem__("probeGuess", np.ones((2, 3), dtype=np.complex64)),
        ),
        "probe_modes": lambda _r, d: _set_descriptor(d, ("probe_modes",), 2),
        "probe_hash": lambda r, d: _refresh(
            r, _set_descriptor(d, ("probe", "sha256"), "0" * 64)
        ),
        "probe_split_mismatch": lambda r, d: _rewrite_test(
            r, d, _set_array("probeGuess", lambda value: value * 2)
        ),
        "truth_missing": lambda r, d: _rewrite_test(r, d, _remove_array("objectGuess")),
        "dose_mismatch": lambda r, d: _refresh(
            r,
            _increment_descriptor(d, ("dose", "train", "counts_mean")),
        ),
        "provenance_json": _bad_provenance_json,
        "provenance_claim": lambda r, d: _mutate_provenance_record(
            r, d, "truth", "none"
        ),
        "provenance_unknown": lambda r, d: _mutate_provenance_record(
            r, d, "claim", "trust me"
        ),
    }


def _at(payload: dict[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def _set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    _at(payload, *path[:-1])[path[-1]] = value


def _delete(payload: dict[str, Any], path: tuple[str, ...]) -> None:
    del _at(payload, *path[:-1])[path[-1]]


def _increment(payload: dict[str, Any], path: tuple[str, ...]) -> None:
    parent = _at(payload, *path[:-1])
    parent[path[-1]] += 1.0


def _apply_provenance_mutation(
    payload: dict[str, Any], dataset_id: str, case: str
) -> None:
    record = ("datasets", dataset_id)
    mutations: dict[str, Callable[[], None]] = {
        "root_unknown": lambda: _set(payload, ("claim",), "trust me"),
        "root_missing": lambda: _delete(payload, ("coordinate_sets",)),
        "materializer_version": lambda: _set(payload, ("materializer_version",), True),
        "expected_ids": lambda: _at(payload, "expected_dataset_ids").append(
            "fabricated"
        ),
        "seeds_unknown": lambda: _set(payload, ("seeds", "extra"), 23),
        "generator_commit": lambda: _set(payload, ("generator_commit",), "dirty"),
        "seeds_missing": lambda: _delete(payload, ("seeds", "object")),
        "seed_type": lambda: _set(payload, ("seeds", "train_coordinates"), True),
        "measurement_seed_missing": lambda: _delete(
            payload, ("seeds", "measurements", dataset_id)
        ),
        "source_object_digest": lambda: _set(
            payload, ("source_objects", "shared_object", "sha256"), "0" * 64
        ),
        "source_object_missing": lambda: _delete(
            payload, ("source_objects", "shared_object", "sha256")
        ),
        "coordinate_digest": lambda: _set(
            payload,
            ("coordinate_sets", "shared_coordinates", "train_x_sha256"),
            "0" * 64,
        ),
        "coordinate_unknown": lambda: _set(
            payload,
            ("coordinate_sets", "shared_coordinates", "extra"),
            "0" * 64,
        ),
        "probe_geometry_digest": lambda: _set(
            payload,
            ("probe_geometries", "shared_probe_geometry", "sha256"),
            "0" * 64,
        ),
        "dataset_identity": lambda: _set(payload, record + ("kind",), "experimental"),
        "dataset_shape_claim": lambda: _set(
            payload, record + ("detector_shape",), [3, 3]
        ),
        "dataset_missing": lambda: _delete(payload, record + ("truth_location",)),
        "truth_location": lambda: _set(
            payload, record + ("truth_location",), "external_npz"
        ),
        "source_object_id": lambda: _set(
            payload, record + ("source_object_id",), "missing_object"
        ),
        "coordinate_set_id": lambda: _set(
            payload, record + ("coordinate_set_id",), "missing_coordinates"
        ),
        "file_hash": lambda: _set(payload, record + ("files", "train"), "0" * 64),
        "file_unknown": lambda: _set(payload, record + ("files", "extra"), "0" * 64),
        "probe_source": lambda: _set(
            payload, record + ("probe", "source"), "other_source"
        ),
        "probe_calibration": lambda: _set(
            payload, record + ("probe", "calibration"), "legacy_normalized"
        ),
        "probe_gauge": lambda: _set(
            payload, record + ("probe", "gauge"), "legacy_normalized"
        ),
        "probe_mask_policy": lambda: _set(
            payload, record + ("probe", "mask_policy"), "pre_masked"
        ),
        "probe_hash": lambda: _set(
            payload, record + ("probe", "train_sha256"), "0" * 64
        ),
        "probe_norm": lambda: _increment(payload, record + ("probe", "train_l2_norm")),
        "probe_missing": lambda: _delete(payload, record + ("probe", "source")),
        "dose": lambda: _increment(payload, record + ("dose", "test", "counts_mean")),
        "dose_missing": lambda: _delete(payload, record + ("dose", "test")),
        "dose_family_base": lambda: _set(
            payload, record + ("base_dataset_id",), "fabricated"
        ),
    }
    mutations[case]()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("file_hash", "train_sha256"),
        ("missing_key", "measurement_key"),
        ("legacy_orientation", "sample-first"),
        ("channel_last", "exact runtime shape"),
        ("coordinate_length", "coordinate"),
        ("coordinate_nonfinite", "finite"),
        ("measurement_nonfinite", "finite"),
        ("count_float", "integer dtype"),
        ("count_negative", "nonnegative"),
        ("amplitude_integer", "floating dtype"),
        ("probe_shape", "probe"),
        ("probe_modes", "probe_modes"),
        ("probe_hash", "probe.*sha256"),
        ("probe_split_mismatch", "probe"),
        ("truth_missing", "truth_key|runtime initialization"),
        ("dose_mismatch", "counts_mean"),
        ("provenance_json", "valid JSON"),
        ("provenance_claim", "provenance"),
        ("provenance_unknown", "unknown field"),
    ],
)
def test_rejects_invalid_content_and_provenance(
    tmp_path: Path, case: str, match: str
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    descriptor = _invalid_content_mutations()[case](tmp_path, descriptor)

    with pytest.raises(api.DatasetError, match=match):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


def test_rejects_complex_coordinates_even_when_provenance_matches(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)

    def complex_x(payload: dict[str, np.ndarray]) -> None:
        payload["xcoords"] = payload["xcoords"].astype(np.complex64)

    _rewrite_split(tmp_path, descriptor, "train", complex_x)
    with np.load(tmp_path / descriptor["train"], allow_pickle=False) as archive:
        x_hash = _array_sha256(archive["xcoords"])

    def coordinate_claim(payload: dict[str, Any]) -> None:
        payload["coordinate_sets"]["shared_coordinates"]["train_x_sha256"] = x_hash

    _mutate_provenance(tmp_path, descriptor, coordinate_claim)

    with pytest.raises(api.DatasetError, match="coordinate.*real|complex"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


@pytest.mark.parametrize(
    "case",
    [
        "root_unknown",
        "root_missing",
        "materializer_version",
        "expected_ids",
        "seeds_unknown",
        "generator_commit",
        "seeds_missing",
        "seed_type",
        "measurement_seed_missing",
        "source_object_digest",
        "source_object_missing",
        "coordinate_digest",
        "coordinate_unknown",
        "probe_geometry_digest",
        "dataset_identity",
        "dataset_shape_claim",
        "dataset_missing",
        "truth_location",
        "source_object_id",
        "coordinate_set_id",
        "file_hash",
        "file_unknown",
        "probe_source",
        "probe_calibration",
        "probe_gauge",
        "probe_mask_policy",
        "probe_hash",
        "probe_norm",
        "probe_missing",
        "dose",
        "dose_missing",
        "dose_family_base",
    ],
)
def test_full_provenance_schema_rejects_each_mutation(
    tmp_path: Path, case: str
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    dataset_id = descriptor["_id"]

    def mutation(payload: dict[str, Any]) -> None:
        _apply_provenance_mutation(payload, dataset_id, case)

    _mutate_provenance(tmp_path, descriptor, mutation)

    with pytest.raises(api.DatasetError, match="provenance|dose family"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


def test_provenance_rejects_descriptor_probe_source_mismatch(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    descriptor["probe"]["source"] = "descriptor_only_source"

    with pytest.raises(api.DatasetError, match="provenance.*probe|probe.*provenance"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    ["train_l2_norm", "train_total_energy", "train_mode_energies"],
)
def test_tiny_probe_provenance_rejects_falsified_near_zero_values(
    tmp_path: Path, field: str
) -> None:
    from scripts.studies.ablation import datasets

    probe = np.full((2, 2), 1e-14 + 1e-14j, dtype=np.complex64)
    descriptor = _bundle(tmp_path, probe_array=probe)

    def falsify(payload: dict[str, Any]) -> None:
        record = payload["datasets"][descriptor["_id"]]["probe"]
        if field == "train_mode_energies":
            record[field][0] = 1e-30
        else:
            record[field] = 1e-30

    _mutate_provenance(tmp_path, descriptor, falsify)

    with pytest.raises(datasets.DatasetError, match="provenance.*mismatch"):
        datasets.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=tmp_path
        )


def test_deterministic_float_policy_has_no_absolute_floor() -> None:
    from scripts.studies.ablation.dataset_content import deterministic_float_equal

    value = 1e-30
    assert deterministic_float_equal(value, value)
    assert not deterministic_float_equal(value, np.nextafter(value, np.inf))


def test_provenance_v1_has_one_typed_parser_and_round_trip_serializer(
    tmp_path: Path,
) -> None:
    from scripts.studies.ablation.dataset_provenance import (
        ProvenanceV1,
        parse_provenance_v1,
        provenance_to_jsonable,
    )

    descriptor = _bundle(tmp_path)
    path = tmp_path / descriptor["provenance"]
    payload = json.loads(path.read_text(encoding="utf-8"))

    parsed = parse_provenance_v1(payload)

    assert isinstance(parsed, ProvenanceV1)
    assert provenance_to_jsonable(parsed) == payload
    assert _file_sha256(path) == descriptor["provenance_sha256"]


def test_provenance_v3_source_map_preserves_typed_source_records() -> None:
    from scripts.studies.ablation.dataset_provenance import (
        CI_COMPATIBILITY_PROVENANCE_V3,
        ProvenanceV3,
        SeedsV1,
        SourceObjectV3,
    )

    source = SourceObjectV3(
        generator="grid_lines_rectangular_v1",
        parameters=(("mapping", "rectangular_v1"),),
        dtype="complex64",
        shape=(64, 64),
        sha256="a" * 64,
    )
    provenance = ProvenanceV3(
        schema_version=CI_COMPATIBILITY_PROVENANCE_V3,
        materializer_id="ci_compatibility_twins_v3",
        materializer_version=3,
        generator_commit="b" * 40,
        materialization_profile="fixture",
        expected_dataset_ids=(),
        seeds=SeedsV1(1, 2, 3, ()),
        source_objects=(("lines", source),),
        coordinate_sets=(),
        probe_geometries=(),
        datasets=(),
    )

    sources: dict[str, SourceObjectV3] = provenance.source_object_map()

    assert provenance.schema_version == CI_COMPATIBILITY_PROVENANCE_V3
    assert sources == {"lines": source}
