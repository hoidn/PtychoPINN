import json
from pathlib import Path

import h5py
import numpy as np
import torch


def _write_darcy_file(path: Path, *, n: int = 10, beta: float = 1.0) -> Path:
    with h5py.File(path, "w") as handle:
        handle.attrs["beta"] = beta
        nu = np.arange(n * 4 * 5, dtype=np.float32).reshape(n, 4, 5)
        tensor = (1000 + np.arange(n * 4 * 5, dtype=np.float32)).reshape(n, 1, 4, 5)
        handle.create_dataset("nu", data=nu)
        handle.create_dataset("tensor", data=tensor)
    return path


def test_darcy_static_operator_dataset_is_lazy_channel_first(tmp_path):
    from scripts.studies.pdebench_image128.data import DarcyStaticOperatorDataset

    data_file = _write_darcy_file(tmp_path / "darcy.hdf5", n=3)
    dataset = DarcyStaticOperatorDataset(
        data_file=data_file,
        input_dataset="nu",
        target_dataset="tensor",
        sample_indices=[2, 0],
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert set(sample) == {"input", "target", "sample_index"}
    assert sample["sample_index"] == 2
    assert tuple(sample["input"].shape) == (1, 4, 5)
    assert tuple(sample["target"].shape) == (1, 4, 5)
    assert sample["input"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32
    assert "time_index" not in sample
    assert "trajectory_id" not in sample


def test_staged_darcy_file_metadata_when_available():
    from scripts.studies.pdebench_image128.data import inspect_darcy_hdf5

    data_file = Path("/home/ollie/Documents/pdebench-data/darcy/2D_DarcyFlow_beta1.0_Train.hdf5")
    if not data_file.exists():
        return

    payload = inspect_darcy_hdf5(data_file)

    assert payload["input_dataset"] == "nu"
    assert payload["input_shape"] == [10000, 128, 128]
    assert payload["target_dataset"] == "tensor"
    assert payload["target_shape"] == [10000, 1, 128, 128]
    assert payload["beta"] == 1.0


def test_default_darcy_sample_split_is_deterministic_and_disjoint():
    from scripts.studies.pdebench_image128.splits import build_sample_split

    first = build_sample_split(10000, seed=20260420)
    second = build_sample_split(10000, seed=20260420)

    assert first == second
    assert len(first["train"]) == 8000
    assert len(first["val"]) == 1000
    assert len(first["test"]) == 1000
    assert set(first["train"]).isdisjoint(first["val"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["val"]).isdisjoint(first["test"])


def test_darcy_split_manifest_records_source_identity_and_counts(tmp_path):
    from scripts.studies.pdebench_image128.splits import build_sample_split, write_sample_split_manifest

    data_file = _write_darcy_file(tmp_path / "darcy.hdf5", n=10, beta=1.0)
    split = build_sample_split(10, seed=20260420, counts=(8, 1, 1))
    path = write_sample_split_manifest(
        output_root=tmp_path / "out",
        data_file=data_file,
        split=split,
        beta=1.0,
        input_dataset="nu",
        target_dataset="tensor",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 20260420
    assert payload["source_file"]["path"] == str(data_file)
    assert payload["source_file"]["size_bytes"] == data_file.stat().st_size
    assert payload["beta"] == 1.0
    assert payload["split_counts"] == {"train": 8, "val": 1, "test": 1}
    assert payload["input_dataset"] == "nu"
    assert payload["target_dataset"] == "tensor"
