import json
from pathlib import Path

import h5py
import numpy as np
import torch


FIELD_ORDER = ("density", "Vx", "Vy", "pressure")


def _write_cfd_file(path: Path, *, trajectories: int = 4, time_steps: int = 5, height: int = 3, width: int = 4) -> dict[str, np.ndarray]:
    values = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["eta"] = 0.01
        handle.attrs["zeta"] = 0.02
        for field_index, field in enumerate(FIELD_ORDER):
            data = (
                field_index * 10_000
                + np.arange(trajectories * time_steps * height * width, dtype=np.float32).reshape(
                    trajectories,
                    time_steps,
                    height,
                    width,
                )
            )
            values[field] = data
            handle.create_dataset(field, data=data)
        handle.create_dataset("x-coordinate", data=np.arange(width, dtype=np.float32) * 0.25)
        handle.create_dataset("y-coordinate", data=np.arange(height, dtype=np.float32) * 0.5)
        handle.create_dataset("t-coordinate", data=np.arange(time_steps + 1, dtype=np.float32) * 0.125)
    return values


def test_cfd_cns_history_window_dataset_stacks_fields_and_history(tmp_path):
    from scripts.studies.pdebench_image128.data import MultiFieldHistoryWindowDataset

    data_file = tmp_path / "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    values = _write_cfd_file(data_file)
    dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=FIELD_ORDER,
        trajectory_ids=[0],
        axis_order="NTHW",
        history_len=2,
        max_windows_per_trajectory=1,
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert tuple(item["input"].shape) == (8, 3, 4)
    assert tuple(item["target"].shape) == (4, 3, 4)
    assert item["target_time_index"] == 2
    assert item["input_time_indices"] == [0, 1]
    expected_input = torch.cat(
        [
            torch.stack([torch.from_numpy(values[field][0, 0]) for field in FIELD_ORDER]),
            torch.stack([torch.from_numpy(values[field][0, 1]) for field in FIELD_ORDER]),
        ],
        dim=0,
    )
    expected_target = torch.stack([torch.from_numpy(values[field][0, 2]) for field in FIELD_ORDER])
    assert torch.equal(item["input"], expected_input)
    assert torch.equal(item["target"], expected_target)
    assert item["field_order"] == list(FIELD_ORDER)


def test_cfd_cns_metadata_and_train_only_multifield_stats(tmp_path):
    from scripts.studies.pdebench_image128.data import inspect_cfd_cns_hdf5
    from scripts.studies.pdebench_image128.normalization import compute_multifield_dynamic_stats

    data_file = tmp_path / "cfd.hdf5"
    values = _write_cfd_file(data_file, trajectories=3, time_steps=4, height=2, width=2)

    metadata = inspect_cfd_cns_hdf5(data_file, history_len=2)
    stats = compute_multifield_dynamic_stats(
        data_file=data_file,
        field_order=FIELD_ORDER,
        axis_order="NTHW",
        train_trajectory_ids=[0, 2],
    )

    expected_mean = [float(values[field][[0, 2]].mean()) for field in FIELD_ORDER]
    expected_std = [float(values[field][[0, 2]].std()) for field in FIELD_ORDER]

    assert metadata["task_id"] == "2d_cfd_cns"
    assert metadata["field_order"] == list(FIELD_ORDER)
    assert metadata["input_channels"] == 8
    assert metadata["target_channels"] == 4
    assert metadata["available_windows"] == 6
    assert metadata["dx"] == 0.25
    assert metadata["dy"] == 0.5
    assert metadata["dt"] == 0.125
    assert metadata["boundary_condition"] == "periodic"
    assert metadata["eta"] == 0.01
    assert metadata["zeta"] == 0.02
    assert stats["field_order"] == list(FIELD_ORDER)
    assert np.allclose(stats["mean"], expected_mean)
    assert np.allclose(stats["std"], expected_std)


def test_cfd_cns_trajectory_split_manifest_counts_history_windows(tmp_path):
    from scripts.studies.pdebench_image128.splits import build_trajectory_split, write_trajectory_split_manifest

    data_file = tmp_path / "cfd.hdf5"
    _write_cfd_file(data_file, trajectories=5, time_steps=6)
    split = build_trajectory_split(5, seed=20260420)
    path = write_trajectory_split_manifest(
        output_root=tmp_path / "out",
        data_file=data_file,
        split=split,
        state_dataset="density,Vx,Vy,pressure",
        axis_order="NTHW",
        shape=[5, 6, 3, 4],
        history_len=2,
        max_windows_per_trajectory=3,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["split_unit"] == "trajectory_id"
    assert payload["history_len"] == 2
    assert payload["window_counts"]["train"] == len(split["train"]) * 3
    assert payload["horizon"] == "one_step"
