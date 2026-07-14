from __future__ import annotations

import json
import importlib
import os
from dataclasses import replace
from pathlib import Path
from shutil import copyfile
from typing import Any

import numpy as np
import pytest

from tests.studies.ablation_dataset_fixtures import (
    _api,
    _array_sha256,
    _bundle,
    _canonical_probe,
    _file_sha256,
    _mutate_provenance,
    _probe_energies,
    _probe_l2_norm,
    _refresh_provenance,
    _replace_single_descriptor_probes,
    _rewrite_split,
    _task10_bundle,
)


@pytest.mark.parametrize(
    ("kind", "truth", "expected_truth", "expected_reference"),
    [
        ("synthetic", "object_truth", True, False),
        ("experimental", "reference_reconstruction", False, True),
        ("experimental", "none", False, False),
    ],
)
def test_validates_count_descriptors_and_derives_immutable_capabilities(
    tmp_path: Path,
    kind: str,
    truth: str,
    expected_truth: bool,
    expected_reference: bool,
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path, kind=kind, truth=truth, dataset_id=f"{kind}_{truth}")

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert validated.descriptor.path_origin == "repository_root"
    assert validated.descriptor.train == (tmp_path / "train.npz").resolve()
    assert validated.descriptor.path_declarations.train == "train.npz"
    assert validated.capabilities.has_object_truth is expected_truth
    assert validated.capabilities.has_reference is expected_reference
    assert validated.capabilities.has_physical_probe is True
    assert validated.capabilities.supports_count_metrics is True
    assert validated.capabilities.supports_dose_sweep is False
    assert validated.capabilities.supports_grouping_C.max_C == 4
    with pytest.raises(AttributeError):
        validated.descriptor.kind = "experimental"


@pytest.mark.parametrize("jitter", [False, True])
def test_grouping_capability_uses_runtime_valid_grid_policy(
    tmp_path: Path, jitter: bool
) -> None:
    api = _api()
    grid_x, grid_y = np.meshgrid(
        np.arange(5, dtype=np.float64), np.arange(5, dtype=np.float64)
    )
    xcoords = grid_x.ravel()
    ycoords = grid_y.ravel()
    if jitter:
        rng = np.random.default_rng(91)
        xcoords = xcoords + rng.uniform(-0.05, 0.05, size=xcoords.shape)
        ycoords = ycoords + rng.uniform(-0.05, 0.05, size=ycoords.shape)
    descriptor = _bundle(
        tmp_path,
        coordinates=(xcoords, ycoords),
    )

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert validated.capabilities.supports_grouping_C.max_C == 4


@pytest.mark.parametrize("sample_count", [3])
def test_grouping_preflight_rejects_insufficient_or_duplicate_runtime_C4(
    tmp_path: Path, sample_count: int
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path, sample_count=sample_count)

    with pytest.raises(api.DatasetError, match="grouping|neighbor|grouping_max_C"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


def test_grouping_preflight_accepts_exactly_four_distinct_candidates(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path, sample_count=4)

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert validated.capabilities.supports_grouping_C.max_C == 4


def test_grouping_canonical_nearest_accepts_sparse_exact_C4(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(
        tmp_path,
        sample_count=7,
        coordinates=(
            np.arange(7, dtype=np.float64) * 100.0,
            np.arange(7, dtype=np.float64) * -50.0,
        ),
    )

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert validated.capabilities.supports_grouping_C.max_C == 4


def test_grouping_capability_preserves_single_sample_C1(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path, sample_count=1)
    descriptor["grouping_max_C"] = 1
    _refresh_provenance(tmp_path, descriptor)

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert validated.capabilities.supports_grouping_C.max_C == 1


def test_validates_embedded_experimental_reference(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(
        tmp_path,
        dataset_id="embedded_reference",
        kind="experimental",
        truth="reference_reconstruction",
        truth_location="embedded_test",
    )

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert validated.descriptor.truth_location == "embedded_test"
    assert validated.descriptor.reference is None
    assert validated.capabilities.has_reference is True
    assert validated.capabilities.has_object_truth is False


def test_validates_legacy_normalized_amplitude(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(
        tmp_path,
        dataset_id="legacy_amp",
        domain="normalized_amplitude",
    )

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert validated.descriptor.scale_contract_version == "legacy_v1"
    assert validated.capabilities.has_physical_probe is False
    assert validated.capabilities.supports_count_metrics is False
    assert validated.capabilities.supports_dose_sweep is False


def test_checked_preflight_rejects_replaced_external_reference_path(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptor = _bundle(
        tmp_path,
        kind="experimental",
        truth="reference_reconstruction",
        truth_location="external_npz",
    )
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    assert validated.descriptor.reference is not None
    outside = tmp_path.parent / f"{tmp_path.name}_outside_reference.npz"
    outside.write_bytes(validated.descriptor.reference.read_bytes())

    forged = replace(validated.descriptor, reference=outside.resolve())
    with pytest.raises(api.DatasetError, match="reference.*repository|origin|identity"):
        api.preflight_dataset(forged)


def test_requires_runtime_object_guess_in_every_split(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path, kind="experimental", truth="none")

    def remove_object(payload: dict[str, np.ndarray]) -> None:
        del payload["objectGuess"]

    _rewrite_split(tmp_path, descriptor, "train", remove_object)

    with pytest.raises(api.DatasetError, match="objectGuess|initialization"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


def test_accepts_real_runtime_initialization_without_inferred_truth(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path, kind="experimental", truth="none")

    def real_initialization(payload: dict[str, np.ndarray]) -> None:
        payload["objectGuess"] = np.ones((4, 4), dtype=np.float32)

    _rewrite_split(tmp_path, descriptor, "train", real_initialization)
    _rewrite_split(tmp_path, descriptor, "test", real_initialization)

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    assert validated.capabilities.has_object_truth is False
    assert validated.capabilities.has_reference is False


def test_rejects_real_synthetic_object_truth(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)

    def real_truth(payload: dict[str, np.ndarray]) -> None:
        payload["objectGuess"] = np.ones((4, 4), dtype=np.float32)

    _rewrite_split(tmp_path, descriptor, "test", real_truth)
    with np.load(tmp_path / descriptor["test"], allow_pickle=False) as archive:
        truth_hash = _array_sha256(archive["objectGuess"])

    def truth_claim(payload: dict[str, Any]) -> None:
        payload["source_objects"]["shared_object"]["sha256"] = truth_hash

    _mutate_provenance(tmp_path, descriptor, truth_claim)

    with pytest.raises(api.DatasetError, match="object_truth.*complex|complex.*truth"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


@pytest.mark.parametrize("case", ["zero_total", "zero_mode"])
def test_rejects_zero_total_or_per_mode_probe_energy(tmp_path: Path, case: str) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    base = np.ones((2, 2), dtype=np.complex64)
    probe = (
        np.zeros((2, 2), dtype=np.complex64)
        if case == "zero_total"
        else np.stack([base, np.zeros_like(base)], axis=0)
    )
    _replace_single_descriptor_probes(tmp_path, descriptor, probe)

    with pytest.raises(api.DatasetError, match="probe.*energy|mode.*positive"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


def test_accepts_all_canonical_probe_layouts_and_multimode(tmp_path: Path) -> None:
    api = _api()
    for layout in ("2d", "trailing", "modes"):
        root = tmp_path / layout
        descriptor = _bundle(root, dataset_id=layout, probe_layout=layout)
        validated = api.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=root
        )
        assert validated.descriptor.probe_modes == (2 if layout == "modes" else 1)


@pytest.mark.parametrize("case", ["missing_key", "bad_shape", "nonfinite"])
def test_validates_external_reference_hash_key_shape_and_finite_values(
    tmp_path: Path, case: str
) -> None:
    api = _api()
    descriptor = _bundle(
        tmp_path,
        dataset_id="experimental_reference",
        kind="experimental",
        truth="reference_reconstruction",
    )
    reference = tmp_path / descriptor["reference"]
    if case == "missing_key":
        np.savez(reference, reconstruction=np.ones((2, 2), dtype=np.complex64))
    elif case == "bad_shape":
        np.savez(reference, object=np.ones((2,), dtype=np.complex64))
    else:
        np.savez(reference, object=np.array([[np.nan]], dtype=np.float32))
    descriptor["reference_sha256"] = _file_sha256(reference)

    def update_reference_hash(payload: dict[str, Any]) -> None:
        payload["datasets"][descriptor["_id"]]["files"]["reference"] = descriptor[
            "reference_sha256"
        ]

    _mutate_provenance(tmp_path, descriptor, update_reference_hash)

    with pytest.raises(api.DatasetError, match="reference|truth_key"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


def test_grouping_preflight_does_not_mutate_global_rng_or_write_stdout(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    from scripts.studies.ablation import datasets

    descriptor = _bundle(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("grouping viability must not touch global NumPy RNG")

    monkeypatch.setattr(np.random, "get_state", forbidden)
    monkeypatch.setattr(np.random, "set_state", forbidden)
    monkeypatch.setattr(np.random, "seed", forbidden)

    datasets.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_bundle_preflight_hashes_and_analyzes_each_shared_artifact_once(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from scripts.studies.ablation import dataset_content, datasets

    descriptors = _task10_bundle(tmp_path)
    snapshot_counts: dict[Path, int] = {}
    load_counts: dict[Path, int] = {}
    real_after_hash = dataset_content.SnapshotReader._after_hash
    real_load = np.load

    def counted_snapshot(reader: Any, handle: Any) -> None:
        snapshot_counts[reader.path] = snapshot_counts.get(reader.path, 0) + 1
        real_after_hash(reader, handle)

    def counted_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        resolved = Path(path.name).resolve()
        load_counts[resolved] = load_counts.get(resolved, 0) + 1
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(dataset_content.SnapshotReader, "_after_hash", counted_snapshot)
    monkeypatch.setattr(np, "load", counted_load)

    datasets.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    assert snapshot_counts
    assert load_counts
    assert len(snapshot_counts) == 5
    assert max(snapshot_counts.values()) == 1
    assert max(load_counts.values()) == 1


@pytest.mark.parametrize("artifact", ["train", "provenance", "reference"])
def test_snapshot_rejects_same_content_path_replacement_after_hash(
    tmp_path: Path, monkeypatch: Any, artifact: str
) -> None:
    from scripts.studies.ablation import dataset_content
    from scripts.studies.ablation import datasets

    descriptor = _bundle(
        tmp_path,
        kind="experimental" if artifact == "reference" else "synthetic",
        truth="reference_reconstruction" if artifact == "reference" else "object_truth",
    )
    target = tmp_path / descriptor[artifact]
    replacement = tmp_path / f"replacement-{target.name}"
    copyfile(target, replacement)
    raced = False

    def replace_after_hash(reader: Any, _handle: Any) -> None:
        nonlocal raced
        if not raced and reader.path == target.resolve():
            raced = True
            replacement.replace(target)

    monkeypatch.setattr(
        dataset_content.SnapshotReader, "_after_hash", replace_after_hash
    )

    with pytest.raises(datasets.DatasetError, match="snapshot|changed|replaced"):
        datasets.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=tmp_path
        )
    assert raced


@pytest.mark.parametrize("mutation", ["append", "truncate"])
def test_snapshot_rejects_in_place_npz_mutation_after_hash(
    tmp_path: Path, monkeypatch: Any, mutation: str
) -> None:
    from scripts.studies.ablation import dataset_content
    from scripts.studies.ablation import datasets

    descriptor = _bundle(tmp_path)
    target = (tmp_path / descriptor["train"]).resolve()
    raced = False

    def mutate_after_hash(reader: Any, _handle: Any) -> None:
        nonlocal raced
        if not raced and reader.path == target:
            raced = True
            if mutation == "append":
                with target.open("ab") as output:
                    output.write(b"in-place-race")
            else:
                with target.open("r+b") as output:
                    output.truncate(target.stat().st_size - 16)

    monkeypatch.setattr(
        dataset_content.SnapshotReader, "_after_hash", mutate_after_hash
    )

    with pytest.raises(datasets.DatasetError, match="snapshot|changed|modified"):
        datasets.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=tmp_path
        )
    assert raced


def test_snapshot_rejects_same_size_overwrite_with_restored_mtime(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from scripts.studies.ablation import dataset_content, datasets

    descriptor = _bundle(tmp_path)
    target = (tmp_path / descriptor["train"]).resolve()
    raced = False

    def overwrite_after_hash(reader: Any, _handle: Any) -> None:
        nonlocal raced
        if not raced and reader.path == target:
            raced = True
            before = target.stat()
            with target.open("r+b") as output:
                first = output.read(1)
                output.seek(0)
                output.write(bytes([first[0] ^ 0x01]))
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert target.stat().st_size == before.st_size
            assert target.stat().st_mtime_ns == before.st_mtime_ns

    monkeypatch.setattr(
        dataset_content.SnapshotReader, "_after_hash", overwrite_after_hash
    )

    with pytest.raises(datasets.DatasetError, match="changed or was replaced"):
        datasets.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=tmp_path
        )
    assert raced


@pytest.mark.parametrize("order", ["experimental_first", "synthetic_first"])
def test_split_cache_reapplies_synthetic_complex_truth_constraint(
    tmp_path: Path, order: str
) -> None:
    from scripts.studies.ablation import dataset_content, datasets

    descriptor = _bundle(
        tmp_path,
        kind="experimental",
        truth="reference_reconstruction",
        truth_location="embedded_test",
    )

    def real_truth(payload: dict[str, np.ndarray]) -> None:
        payload["objectGuess"] = np.ones((4, 4), dtype=np.float32)

    _rewrite_split(tmp_path, descriptor, "train", real_truth)
    _rewrite_split(tmp_path, descriptor, "test", real_truth)
    validated = datasets.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    experimental = validated.descriptor
    synthetic = replace(
        experimental,
        kind="synthetic",
        truth="object_truth",
    )
    cache = dataset_content.ArtifactCache()

    def analyze(item: Any) -> Any:
        return dataset_content._validate_split(
            item,
            "test",
            item.test,
            item.test_sha256,
            cache,
            truth_key=item.truth_key,
            requires_raw_probe_geometry=False,
        )

    if order == "experimental_first":
        cached = analyze(experimental)
        assert cached.truth_dtype == "float32"
        assert cached.truth_is_complex is False
        with pytest.raises(
            datasets.DatasetError, match="object_truth.*complex|complex.*truth"
        ):
            analyze(synthetic)
    else:
        with pytest.raises(
            datasets.DatasetError, match="object_truth.*complex|complex.*truth"
        ):
            analyze(synthetic)
        assert analyze(experimental).truth_is_complex is False


def test_preflight_ignores_unrequested_object_array_npz_member(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from numpy.lib.npyio import NpzFile

    from scripts.studies.ablation import datasets

    descriptor = _bundle(tmp_path)
    train_path = tmp_path / descriptor["train"]
    with np.load(train_path, allow_pickle=False) as archive:
        payload = {
            key: archive[key]
            for key in (
                "diff3d",
                "probeGuess",
                "xcoords",
                "ycoords",
                "objectGuess",
            )
        }
    np.savez(
        train_path,
        **payload,
        unused_metadata=np.array([{"must_not_load": True}], dtype=object),
    )
    descriptor["train_sha256"] = _file_sha256(train_path)

    def update_hash(payload_json: dict[str, Any]) -> None:
        payload_json["datasets"][descriptor["_id"]]["files"]["train"] = descriptor[
            "train_sha256"
        ]

    _mutate_provenance(tmp_path, descriptor, update_hash)
    accessed: list[str] = []
    real_getitem = NpzFile.__getitem__

    def counted_getitem(self: NpzFile, key: str) -> np.ndarray:
        accessed.append(key)
        return np.asarray(real_getitem(self, key))

    monkeypatch.setattr(NpzFile, "__getitem__", counted_getitem)

    datasets.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)

    assert "unused_metadata" not in accessed
    assert {"diff3d", "probeGuess", "xcoords", "ycoords", "objectGuess"} <= set(
        accessed
    )


def test_generic_descriptor_ignores_unrelated_object_probe_geometry(
    tmp_path: Path,
) -> None:
    from scripts.studies.ablation import datasets

    descriptor = _bundle(tmp_path)
    train_path = tmp_path / descriptor["train"]
    with np.load(train_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    np.savez(
        train_path,
        **payload,
        probeGeometry=np.array([{"not": "task10"}], dtype=object),
    )
    descriptor["train_sha256"] = _file_sha256(train_path)

    def update_hash(provenance: dict[str, Any]) -> None:
        provenance["datasets"][descriptor["_id"]]["files"]["train"] = descriptor[
            "train_sha256"
        ]

    _mutate_provenance(tmp_path, descriptor, update_hash)

    datasets.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


def test_task10_requires_typed_raw_probe_geometry_member(tmp_path: Path) -> None:
    from scripts.studies.ablation import datasets

    descriptors = _task10_bundle(tmp_path)
    train_path = tmp_path / descriptors["ci_3p5m"]["train"]
    with np.load(train_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files if key != "probeGeometry"}
    np.savez(
        train_path,
        **payload,
        probeGeometry=np.array([{"malformed": True}], dtype=object),
    )
    train_hash = _file_sha256(train_path)
    provenance_path = tmp_path / descriptors["ci_3p5m"]["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    descriptors["ci_3p5m"]["train_sha256"] = train_hash
    provenance["datasets"]["ci_3p5m"]["files"]["train"] = train_hash
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    provenance_hash = _file_sha256(provenance_path)
    for item in descriptors.values():
        item["provenance_sha256"] = provenance_hash

    with pytest.raises(datasets.DatasetError, match="probeGeometry|pickle|raw probe"):
        datasets.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)


@pytest.mark.parametrize("mutation", ["phase", "mode_power"])
def test_generic_split_probe_geometry_uses_dimensionless_scalar_residual(
    tmp_path: Path, mutation: str
) -> None:
    from scripts.studies.ablation import datasets

    base = np.stack(
        [
            np.full((2, 2), 1e-14 + 2e-14j, dtype=np.complex64),
            np.full((2, 2), 3e-14 - 1e-14j, dtype=np.complex64),
        ]
    )
    descriptor = _bundle(tmp_path, probe_array=base)

    def mutate_test(payload: dict[str, np.ndarray]) -> None:
        probe = payload["probeGuess"].copy()
        if mutation == "phase":
            probe *= np.complex64(np.exp(0.2j))
        else:
            probe[1] *= np.complex64(1.2)
        payload["probeGuess"] = probe

    _rewrite_split(tmp_path, descriptor, "test", mutate_test)
    with np.load(tmp_path / descriptor["test"], allow_pickle=False) as archive:
        test_probe = archive["probeGuess"]
    canonical_train = _canonical_probe(base)
    canonical_test = _canonical_probe(test_probe)
    descriptor["probe"] = {
        key: value for key, value in descriptor["probe"].items() if key != "sha256"
    }
    descriptor["probe"].update(
        train_sha256=_array_sha256(canonical_train),
        test_sha256=_array_sha256(canonical_test),
    )
    test_total, test_modes = _probe_energies(test_probe)

    def update_probe(provenance: dict[str, Any]) -> None:
        record = provenance["datasets"][descriptor["_id"]]
        record["probe"].update(
            test_sha256=descriptor["probe"]["test_sha256"],
            test_l2_norm=_probe_l2_norm(test_probe),
            test_total_energy=test_total,
            test_mode_energies=test_modes,
        )

    _mutate_provenance(tmp_path, descriptor, update_probe)

    with pytest.raises(datasets.DatasetError, match="positive-real|geometry|scalar"):
        datasets.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=tmp_path
        )


def test_grouping_queries_each_unique_coordinate_policy_once(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from scripts.studies.ablation import dataset_content, datasets

    descriptors = _task10_bundle(tmp_path)
    calls = 0
    real_query = dataset_content.nearest_neighbor_indices

    def counted_query(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray(real_query(*args, **kwargs))

    monkeypatch.setattr(dataset_content, "nearest_neighbor_indices", counted_query)

    datasets.load_checked_dataset_bundle(descriptors, repo_root=tmp_path)

    assert calls == 1


@pytest.mark.parametrize(
    "coordinates",
    [
        np.array([(x, y) for y in range(3) for x in range(3)], dtype=np.float64),
        np.column_stack((np.arange(9.0) * 100.0, np.arange(9.0) * -7.0)),
    ],
)
def test_local_nearest_adapter_matches_runtime(
    coordinates: np.ndarray,
) -> None:
    from scripts.studies.ablation.dataset_content import nearest_neighbor_indices

    config_module = importlib.import_module("ptycho_torch.config_params")
    patch_generator = importlib.import_module("ptycho_torch.patch_generator")
    xcoords = np.asarray(coordinates[:, 0], dtype=np.float64)
    ycoords = np.asarray(coordinates[:, 1], dtype=np.float64)
    config = config_module.DataConfig(K=6, neighbor_function="Nearest")

    expected = patch_generator.get_neighbor_indices(xcoords, ycoords, config, K=6)

    assert np.array_equal(nearest_neighbor_indices(xcoords, ycoords, K=6), expected)
