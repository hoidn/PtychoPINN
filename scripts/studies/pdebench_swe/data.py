"""Lazy one-step HDF5 dataset for PDEBench SWE smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from scripts.studies.pdebench_swe.splits import axis_index, infer_dimensions


class SweOneStepDataset(Dataset):
    """Lazy HDF5-backed one-step dataset with channel-first tensor output."""

    def __init__(
        self,
        *,
        data_file: Path,
        state_dataset: str,
        trajectory_ids: list[int],
        axis_order: str,
        normalization: dict[str, Any] | None = None,
        max_pairs_per_trajectory: int | None = None,
        pad_multiple: int = 1,
    ):
        self.data_file = Path(data_file)
        self.state_dataset = state_dataset.strip("/")
        self.trajectory_ids = [int(item) for item in trajectory_ids]
        self.axis_order = axis_order
        self.normalization = normalization
        self.pad_multiple = max(1, int(pad_multiple))
        self._handle: h5py.File | None = None
        self._trajectory_paths: dict[int, str] | None = None

        with h5py.File(self.data_file, "r") as handle:
            if self.state_dataset == "*/data":
                paths = {
                    int(name): f"{name}/data"
                    for name in handle.keys()
                    if str(name).isdigit() and isinstance(handle.get(f"{name}/data"), h5py.Dataset)
                }
                if not paths:
                    raise KeyError("state_dataset '*/data' did not match any trajectory data groups")
                first_path = paths[sorted(paths)[0]]
                child_shape = [int(item) for item in handle[first_path].shape]
                shape = [len(paths), *child_shape]
                self._trajectory_paths = paths
            else:
                shape = [int(item) for item in handle[self.state_dataset].shape]
        self.shape = shape
        self.dimensions = infer_dimensions(shape, axis_order)
        pairs_per_trajectory = self.dimensions["time_steps"] - 1
        if max_pairs_per_trajectory is not None:
            pairs_per_trajectory = min(pairs_per_trajectory, int(max_pairs_per_trajectory))
        self.pairs = [
            (trajectory_id, time_index)
            for trajectory_id in self.trajectory_ids
            for time_index in range(pairs_per_trajectory)
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def _dataset(self, trajectory_id: int | None = None) -> h5py.Dataset:
        if self._handle is None:
            self._handle = h5py.File(self.data_file, "r")
        if self._trajectory_paths is not None:
            if trajectory_id is None:
                raise ValueError("trajectory_id is required for grouped trajectory datasets")
            return self._handle[self._trajectory_paths[int(trajectory_id)]]
        return self._handle[self.state_dataset]

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()

    def _read_state(self, trajectory_id: int, time_index: int) -> torch.Tensor:
        dataset = self._dataset(trajectory_id)
        local_axis_order = self.axis_order
        selection: list[Any] = [slice(None)] * len(local_axis_order)
        if self._trajectory_paths is not None:
            local_axis_order = self.axis_order.replace("N", "", 1)
            selection = [slice(None)] * len(local_axis_order)
        else:
            selection[axis_index(local_axis_order, "N")] = int(trajectory_id)
        selection[axis_index(local_axis_order, "T")] = int(time_index)
        array = np.asarray(dataset[tuple(selection)], dtype=np.float32)

        remaining_axes = [axis for axis in local_axis_order if axis not in {"N", "T"}]
        if "C" not in remaining_axes:
            array = array[..., np.newaxis]
            remaining_axes.append("C")
        channel_axis = remaining_axes.index("C")
        height_axis = remaining_axes.index("H")
        width_axis = remaining_axes.index("W")
        array = np.moveaxis(array, [channel_axis, height_axis, width_axis], [0, 1, 2])
        return torch.from_numpy(np.ascontiguousarray(array)).float()

    def _apply_normalization(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalization:
            return tensor
        mean = torch.tensor(self.normalization["mean"], dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(self.normalization["std"], dtype=tensor.dtype).view(-1, 1, 1)
        return (tensor - mean) / torch.clamp(std, min=1e-12)

    def _pad(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        _, height, width = tensor.shape
        padded_height = ((height + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        padded_width = ((width + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        pad_bottom = padded_height - height
        pad_right = padded_width - width
        pad = {
            "original_shape": [int(height), int(width)],
            "padded_shape": [int(padded_height), int(padded_width)],
            "pad": [0, int(pad_right), 0, int(pad_bottom)],
        }
        if pad_bottom or pad_right:
            tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom))
        return tensor, pad

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_id, time_index = self.pairs[index]
        x = self._apply_normalization(self._read_state(trajectory_id, time_index))
        y = self._apply_normalization(self._read_state(trajectory_id, time_index + 1))
        x, pad = self._pad(x)
        y, _ = self._pad(y)
        return {
            "input": x,
            "target": y,
            "trajectory_id": int(trajectory_id),
            "time_index": int(time_index),
            "pad": pad,
        }
