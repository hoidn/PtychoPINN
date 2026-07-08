import numpy as np
import pytest


def _write_pair(root, data_name, data_shape, model_name, model_shape):
    np.save(root / data_name, np.zeros(data_shape, dtype=np.float32))
    np.save(root / model_name, np.ones(model_shape, dtype=np.float32))


def test_validate_expected_flatvel_shapes(tmp_path):
    from scripts.studies.openfwi_flatvel_a.data import inspect_shard_pair

    _write_pair(tmp_path, "data1.npy", (4, 5, 1000, 70), "model1.npy", (4, 1, 70, 70))

    payload = inspect_shard_pair(tmp_path / "data1.npy", tmp_path / "model1.npy", expected_samples=None)

    assert payload["data_shape"][1:] == [5, 1000, 70]
    assert payload["model_shape"][1:] == [1, 70, 70]
    assert payload["num_samples"] == 4
    assert payload["sample_count_contract"] == "synthetic_fixture"


def test_real_flatvel_shape_contract_requires_500_samples(tmp_path):
    from scripts.studies.openfwi_flatvel_a.data import inspect_shard_pair

    _write_pair(tmp_path, "data1.npy", (4, 5, 1000, 70), "model1.npy", (4, 1, 70, 70))

    payload = inspect_shard_pair(tmp_path / "data1.npy", tmp_path / "model1.npy")

    assert payload["status"] == "schema_difference"
    assert "data_sample_count" in payload["differences"]
    assert "model_sample_count" in payload["differences"]
    assert payload["expected_data_shape"] == [500, 5, 1000, 70]
    assert payload["expected_model_shape"] == [500, 1, 70, 70]


def test_unexpected_shape_records_schema_difference(tmp_path):
    from scripts.studies.openfwi_flatvel_a.data import inspect_shard_pair

    _write_pair(tmp_path, "data1.npy", (4, 5, 999, 70), "model1.npy", (4, 1, 70, 70))

    payload = inspect_shard_pair(tmp_path / "data1.npy", tmp_path / "model1.npy")

    assert payload["status"] == "schema_difference"
    assert "data_shape_suffix" in payload["differences"]


def test_split_manifest_uses_train_and_test_shards_with_seed():
    from scripts.studies.openfwi_flatvel_a.data import build_split_manifest

    manifest = build_split_manifest(
        train_count=500,
        test_count=500,
        train_samples=32,
        val_samples=16,
        test_samples=16,
        seed=20260420,
    )

    assert manifest["train"]["data_shard"] == "data1.npy"
    assert manifest["val"]["data_shard"] == "data49.npy"
    assert manifest["test"]["data_shard"] == "data49.npy"
    assert manifest["seed"] == 20260420
    assert set(manifest["val"]["indices"]).isdisjoint(manifest["test"]["indices"])


def test_dataset_returns_local_adapter_tensors(tmp_path):
    from scripts.studies.openfwi_flatvel_a.data import OpenFWIShardDataset, build_split_manifest

    data = np.random.default_rng(3).normal(size=(4, 5, 1000, 70)).astype(np.float32)
    model = np.random.default_rng(4).normal(size=(4, 1, 70, 70)).astype(np.float32)
    np.save(tmp_path / "data1.npy", data)
    np.save(tmp_path / "model1.npy", model)
    split = build_split_manifest(train_count=4, test_count=4, train_samples=2, val_samples=1, test_samples=1)

    dataset = OpenFWIShardDataset(
        data_path=tmp_path / "data1.npy",
        model_path=tmp_path / "model1.npy",
        split_name="train",
        indices=split["train"]["indices"],
    )
    item = dataset[0]

    assert item["input_raw"].shape == (5, 1000, 70)
    assert item["input"].shape == (5, 70, 70)
    assert item["target"].shape == (1, 70, 70)
    assert item["sample_id"].startswith("data1.npy:")


def test_normalization_stats_use_train_split_only(tmp_path):
    from scripts.studies.openfwi_flatvel_a.data import (
        OpenFWIShardDataset,
        build_split_manifest,
        compute_normalization_stats,
    )

    np.save(tmp_path / "data1.npy", np.ones((4, 5, 1000, 70), dtype=np.float32) * 2)
    np.save(tmp_path / "model1.npy", np.ones((4, 1, 70, 70), dtype=np.float32) * 3)
    split = build_split_manifest(train_count=4, test_count=4, train_samples=2, val_samples=1, test_samples=1)
    dataset = OpenFWIShardDataset(
        data_path=tmp_path / "data1.npy",
        model_path=tmp_path / "model1.npy",
        split_name="train",
        indices=split["train"]["indices"],
    )

    stats = compute_normalization_stats(dataset)

    assert stats["source"] == "train_split_only"
    assert stats["input"]["mean"] == pytest.approx(2.0)
    assert stats["target"]["mean"] == pytest.approx(3.0)
