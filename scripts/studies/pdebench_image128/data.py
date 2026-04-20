"""Data contracts for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _channel_first(array: np.ndarray, axis_order: str) -> np.ndarray:
    axis_order = axis_order.upper()
    if "N" in axis_order:
        raise ValueError("_channel_first expects one sample without an N axis")
    if axis_order == "HW":
        array = array[np.newaxis, ...]
        axis_order = "CHW"
    if "C" not in axis_order:
        array = array[np.newaxis, ...]
        axis_order = "CHW"
    channel_axis = axis_order.index("C")
    height_axis = axis_order.index("H")
    width_axis = axis_order.index("W")
    return np.ascontiguousarray(np.moveaxis(array, [channel_axis, height_axis, width_axis], [0, 1, 2]))


def read_sample_channel_first(dataset: h5py.Dataset, sample_index: int, axis_order: str) -> np.ndarray:
    """Read one HDF5 sample and return float32 channel-first CHW data."""
    axis_order = axis_order.upper()
    if "N" not in axis_order:
        raise ValueError(f"axis_order {axis_order!r} must include sample axis N")
    selection: list[Any] = [slice(None)] * len(axis_order)
    selection[axis_order.index("N")] = int(sample_index)
    array = np.asarray(dataset[tuple(selection)], dtype=np.float32)
    local_order = axis_order.replace("N", "", 1)
    return _channel_first(array, local_order)


def inspect_darcy_hdf5(
    data_file: Path,
    *,
    input_dataset: str = "nu",
    target_dataset: str = "tensor",
) -> dict[str, Any]:
    """Inspect the official Darcy beta 1.0 static-operator HDF5 schema."""
    data_file = Path(data_file)
    with h5py.File(data_file, "r") as handle:
        input_ds = handle[input_dataset]
        target_ds = handle[target_dataset]
        beta = handle.attrs.get("beta")
        return {
            "schema_version": "pdebench_darcy_hdf5_metadata_v1",
            "data_file": str(data_file),
            "file_size_bytes": int(data_file.stat().st_size),
            "beta": None if beta is None else float(beta),
            "input_dataset": input_dataset,
            "input_shape": [int(item) for item in input_ds.shape],
            "input_dtype": str(input_ds.dtype),
            "input_axis_order": "NHW",
            "target_dataset": target_dataset,
            "target_shape": [int(item) for item in target_ds.shape],
            "target_dtype": str(target_ds.dtype),
            "target_axis_order": "NCHW",
            "sample_count": int(input_ds.shape[0]),
            "static_operator_contract": "nu[i] -> tensor[i]",
        }


class DarcyStaticOperatorDataset(Dataset):
    """Lazy HDF5-backed Darcy operator dataset with `nu[i] -> tensor[i]` samples."""

    def __init__(
        self,
        *,
        data_file: Path,
        input_dataset: str = "nu",
        target_dataset: str = "tensor",
        sample_indices: Sequence[int] | None = None,
        input_axis_order: str = "NHW",
        target_axis_order: str = "NCHW",
        input_stats: dict[str, Any] | None = None,
        target_stats: dict[str, Any] | None = None,
    ):
        self.data_file = Path(data_file)
        self.input_dataset = input_dataset
        self.target_dataset = target_dataset
        self.input_axis_order = input_axis_order
        self.target_axis_order = target_axis_order
        self.input_stats = input_stats
        self.target_stats = target_stats
        self._handle: h5py.File | None = None
        with h5py.File(self.data_file, "r") as handle:
            count = int(handle[input_dataset].shape[0])
            target_count = int(handle[target_dataset].shape[0])
        if count != target_count:
            raise ValueError(f"input/target sample counts differ: {count} vs {target_count}")
        self.sample_indices = list(range(count)) if sample_indices is None else [int(item) for item in sample_indices]

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.data_file, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()

    @staticmethod
    def _apply_stats(tensor: torch.Tensor, stats: dict[str, Any] | None) -> torch.Tensor:
        if not stats:
            return tensor
        mean = torch.tensor(stats["mean"], dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(stats["std"], dtype=tensor.dtype).view(-1, 1, 1)
        return (tensor - mean) / torch.clamp(std, min=1e-12)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index = self.sample_indices[index]
        handle = self._h5()
        x = read_sample_channel_first(handle[self.input_dataset], sample_index, self.input_axis_order)
        y = read_sample_channel_first(handle[self.target_dataset], sample_index, self.target_axis_order)
        input_tensor = self._apply_stats(torch.from_numpy(x).float(), self.input_stats)
        target_tensor = self._apply_stats(torch.from_numpy(y).float(), self.target_stats)
        return {
            "input": input_tensor,
            "target": target_tensor,
            "sample_index": int(sample_index),
        }
