import json
from pathlib import Path

import h5py
import numpy as np
import torch


def _write_swe_h5(path: Path, *, shape=(5, 4, 3, 4, 2)) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values)
    return values


def test_build_trajectory_split_is_deterministic_and_disjoint():
    from scripts.studies.pdebench_swe.splits import build_trajectory_split

    first = build_trajectory_split(10, seed=20260420)
    second = build_trajectory_split(10, seed=20260420)

    assert first == second
    assert len(first["train"]) == 8
    assert len(first["val"]) == 1
    assert len(first["test"]) == 1
    all_ids = first["train"] + first["val"] + first["test"]
    assert sorted(all_ids) == list(range(10))
    assert len(all_ids) == len(set(all_ids))


def test_write_split_manifest_records_pair_counts_and_source_identity(tmp_path):
    from scripts.studies.pdebench_swe.splits import (
        build_trajectory_split,
        write_split_manifest,
    )

    split = build_trajectory_split(5, seed=20260420)
    path = write_split_manifest(
        output_root=tmp_path,
        source_file_identity={"path": "/data/2D_rdb_NA_NA.h5", "sha256": "abc"},
        state_dataset="data",
        axis_order="NTHWC",
        shape=[5, 4, 3, 4, 2],
        split=split,
        max_pairs_per_trajectory=2,
        run_id="split-test",
    )

    payload = json.loads(path.read_text())
    assert payload["run_id"] == "split-test"
    assert payload["seed"] == 20260420
    assert payload["state_dataset"] == "data"
    assert payload["axis_order"] == "NTHWC"
    assert payload["pair_counts"]["train"] == len(split["train"]) * 2
    assert payload["source_file_identity"]["sha256"] == "abc"


def test_write_full_and_run_split_manifests_preserves_full_ids_and_capped_subset(tmp_path):
    from scripts.studies.pdebench_swe.splits import (
        build_run_subset_split,
        build_trajectory_split,
        write_longer_split_manifests,
    )

    full_split = build_trajectory_split(1000, seed=20260420)
    run_split = build_run_subset_split(
        full_split,
        max_train_trajectories=8,
        max_val_trajectories=3,
        max_test_trajectories=2,
    )

    assert len(full_split["train"]) == 800
    assert len(full_split["val"]) == 100
    assert len(full_split["test"]) == 100
    assert run_split["train"] == full_split["train"][:8]
    assert run_split["val"] == full_split["val"][:3]
    assert run_split["test"] == full_split["test"][:2]

    paths = write_longer_split_manifests(
        output_root=tmp_path,
        source_file_identity={"path": "/data/2D_rdb_NA_NA.h5", "sha256": "abc"},
        state_dataset="*/data",
        axis_order="NTHWC",
        shape=[1000, 101, 128, 128, 1],
        full_split=full_split,
        run_split=run_split,
        max_pairs_per_trajectory=10,
        run_id="longer-split",
    )

    full_payload = json.loads(paths["full"].read_text())
    run_payload = json.loads(paths["run"].read_text())
    assert full_payload["manifest_kind"] == "full_split"
    assert run_payload["manifest_kind"] == "run_subset"
    assert full_payload["pair_counts"]["train"] == 800 * 100
    assert run_payload["pair_counts"]["train"] == 8 * 10
    assert run_payload["full_split_manifest"] == "split_manifest_full.json"
    assert run_payload["subset_of_full_split"] is True


def test_swe_one_step_dataset_returns_channel_first_tensors_and_metadata(tmp_path):
    from scripts.studies.pdebench_swe.data import SweOneStepDataset

    data_file = tmp_path / "2D_rdb_NA_NA.h5"
    values = _write_swe_h5(data_file)

    dataset = SweOneStepDataset(
        data_file=data_file,
        state_dataset="data",
        trajectory_ids=[0, 1],
        axis_order="NTHWC",
        max_pairs_per_trajectory=1,
    )

    assert dataset._handle is None
    item = dataset[0]

    assert item["input"].shape == (2, 3, 4)
    assert item["target"].shape == (2, 3, 4)
    assert item["trajectory_id"] in {0, 1}
    assert item["time_index"] == 0
    expected = torch.from_numpy(values[item["trajectory_id"], 0]).permute(2, 0, 1)
    assert torch.equal(item["input"], expected)


def test_swe_one_step_dataset_applies_normalization_and_padding(tmp_path):
    from scripts.studies.pdebench_swe.data import SweOneStepDataset

    data_file = tmp_path / "2D_rdb_NA_NA.h5"
    _write_swe_h5(data_file, shape=(2, 3, 3, 5, 1))

    dataset = SweOneStepDataset(
        data_file=data_file,
        state_dataset="data",
        trajectory_ids=[0],
        axis_order="NTHWC",
        normalization={"mean": [1.0], "std": [2.0]},
        max_pairs_per_trajectory=1,
        pad_multiple=4,
    )

    item = dataset[0]

    assert item["input"].shape == (1, 4, 8)
    assert item["pad"]["original_shape"] == [3, 5]
    assert item["pad"]["padded_shape"] == [4, 8]
    assert torch.isclose(item["input"][0, 0, 0], torch.tensor(-0.5))


def test_swe_one_step_dataset_reads_group_per_trajectory_layout(tmp_path):
    from scripts.studies.pdebench_swe.data import SweOneStepDataset

    data_file = tmp_path / "grouped.h5"
    with h5py.File(data_file, "w") as handle:
        for trajectory_id in range(2):
            group = handle.create_group(f"{trajectory_id:04d}")
            values = np.full((3, 2, 2, 1), fill_value=trajectory_id, dtype=np.float32)
            values[1] = trajectory_id + 10
            group.create_dataset("data", data=values)

    dataset = SweOneStepDataset(
        data_file=data_file,
        state_dataset="*/data",
        trajectory_ids=[1],
        axis_order="NTHWC",
        max_pairs_per_trajectory=1,
    )

    item = dataset[0]

    assert item["trajectory_id"] == 1
    assert item["input"].shape == (1, 2, 2)
    assert torch.equal(item["input"], torch.ones(1, 2, 2))
    assert torch.equal(item["target"], torch.full((1, 2, 2), 11.0))
