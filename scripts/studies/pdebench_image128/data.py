"""Data contracts for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from scripts.studies.pdebench_image128.splits import axis_index, infer_dynamic_dimensions

CFD_CNS_FIELD_ORDER = ("density", "Vx", "Vy", "pressure")


def _uniform_coordinate_step(values: np.ndarray, *, name: str) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    deltas = np.diff(values)
    reference = float(deltas[0])
    if not np.allclose(deltas, reference):
        raise ValueError(f"{name} must be uniformly spaced")
    return reference


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


def grouped_trajectory_paths(handle: h5py.File, state_dataset: str) -> dict[int, str] | None:
    """Resolve a grouped trajectory dataset pattern such as `*/data`."""
    normalized = state_dataset.strip("/")
    if normalized != "*/data":
        return None
    paths = {
        int(name): f"{name}/data"
        for name in handle.keys()
        if str(name).isdigit() and isinstance(handle.get(f"{name}/data"), h5py.Dataset)
    }
    if not paths:
        raise KeyError("state_dataset '*/data' did not match any trajectory data groups")
    return paths


def read_dynamic_state_channel_first(
    handle: h5py.File,
    *,
    state_dataset: str,
    trajectory_id: int,
    time_index: int,
    axis_order: str,
    trajectory_paths: dict[int, str] | None = None,
) -> np.ndarray:
    """Read one trajectory/time state and return float32 channel-first CHW data."""
    normalized = state_dataset.strip("/")
    local_axis_order = axis_order.upper()
    if trajectory_paths is None:
        trajectory_paths = grouped_trajectory_paths(handle, normalized)
    if trajectory_paths is not None:
        dataset = handle[trajectory_paths[int(trajectory_id)]]
        local_axis_order = local_axis_order.replace("N", "", 1)
        selection: list[Any] = [slice(None)] * len(local_axis_order)
    else:
        dataset = handle[normalized]
        selection = [slice(None)] * len(local_axis_order)
        selection[axis_index(local_axis_order, "N")] = int(trajectory_id)
    selection[axis_index(local_axis_order, "T")] = int(time_index)
    array = np.asarray(dataset[tuple(selection)], dtype=np.float32)
    remaining_axes = [axis for axis in local_axis_order if axis not in {"N", "T"}]
    return _channel_first(array, "".join(remaining_axes))


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


def inspect_cfd_cns_hdf5(
    data_file: Path,
    *,
    field_order: Sequence[str] = CFD_CNS_FIELD_ORDER,
    axis_order: str = "NTHW",
    history_len: int = 2,
) -> dict[str, Any]:
    """Inspect the PDEBench 2D compressible Navier-Stokes HDF5 schema."""
    data_file = Path(data_file)
    field_order = tuple(str(item) for item in field_order)
    history_len = int(history_len)
    if history_len < 1:
        raise ValueError("history_len must be at least 1")
    with h5py.File(data_file, "r") as handle:
        missing = [field for field in field_order if field not in handle]
        if missing:
            raise KeyError(f"missing 2D CFD CNS fields: {missing}")
        coordinate_names = ("x-coordinate", "y-coordinate", "t-coordinate")
        missing_coordinates = [name for name in coordinate_names if name not in handle]
        if missing_coordinates:
            raise KeyError(f"missing 2D CFD CNS coordinates: {missing_coordinates}")
        shapes = {field: [int(item) for item in handle[field].shape] for field in field_order}
        dtypes = {field: str(handle[field].dtype) for field in field_order}
        dx = _uniform_coordinate_step(handle["x-coordinate"][...], name="x-coordinate")
        dy = _uniform_coordinate_step(handle["y-coordinate"][...], name="y-coordinate")
        dt = _uniform_coordinate_step(handle["t-coordinate"][...], name="t-coordinate")
        eta = handle.attrs.get("eta")
        zeta = handle.attrs.get("zeta")
    unique_shapes = {tuple(shape) for shape in shapes.values()}
    if len(unique_shapes) != 1:
        raise ValueError(f"2D CFD CNS fields must share one shape; observed {shapes}")
    shape = list(next(iter(unique_shapes)))
    dims = infer_dynamic_dimensions(shape, axis_order)
    if dims["time_steps"] <= history_len:
        raise ValueError("history_len leaves no eligible one-step windows")
    return {
        "schema_version": "pdebench_2d_cfd_cns_hdf5_metadata_v1",
        "task_id": "2d_cfd_cns",
        "pde_name": "2d_cfd",
        "data_file": str(data_file),
        "file_size_bytes": int(data_file.stat().st_size),
        "field_order": list(field_order),
        "field_shapes": shapes,
        "field_dtypes": dtypes,
        "dx": float(dx),
        "dy": float(dy),
        "dt": float(dt),
        "eta": None if eta is None else float(eta),
        "zeta": None if zeta is None else float(zeta),
        "boundary_condition": "periodic",
        "state_shape": shape,
        "field_axis_order": axis_order,
        "dimensions": dims,
        "trajectory_count": int(dims["num_trajectories"]),
        "time_steps": int(dims["time_steps"]),
        "state_channels": int(len(field_order)),
        "history_len": history_len,
        "input_channels": int(len(field_order) * history_len),
        "target_channels": int(len(field_order)),
        "windows_per_trajectory": int(dims["time_steps"] - history_len),
        "available_windows": int(dims["num_trajectories"] * (dims["time_steps"] - history_len)),
        "dynamic_history_contract": f"concat u[t-{history_len}:t] -> u[t]",
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


class DynamicHistoryWindowDataset(Dataset):
    """Lazy HDF5-backed dynamic-state dataset with fixed history windows."""

    def __init__(
        self,
        *,
        data_file: Path,
        state_dataset: str,
        trajectory_ids: Sequence[int],
        axis_order: str,
        history_len: int,
        normalization: dict[str, Any] | None = None,
        max_windows_per_trajectory: int | None = None,
        pad_multiple: int = 1,
    ):
        self.data_file = Path(data_file)
        self.state_dataset = state_dataset.strip("/")
        self.trajectory_ids = [int(item) for item in trajectory_ids]
        self.axis_order = axis_order.upper()
        self.history_len = int(history_len)
        if self.history_len < 1:
            raise ValueError("history_len must be at least 1")
        self.normalization = normalization
        self.pad_multiple = max(1, int(pad_multiple))
        self._handle: h5py.File | None = None
        self._trajectory_paths: dict[int, str] | None = None

        with h5py.File(self.data_file, "r") as handle:
            paths = grouped_trajectory_paths(handle, self.state_dataset)
            if paths is not None:
                first_path = paths[sorted(paths)[0]]
                child_shape = [int(item) for item in handle[first_path].shape]
                shape = [len(paths), *child_shape]
                self._trajectory_paths = paths
            else:
                shape = [int(item) for item in handle[self.state_dataset].shape]
        self.shape = shape
        self.dimensions = infer_dynamic_dimensions(shape, self.axis_order)
        windows_per_trajectory = self.dimensions["time_steps"] - self.history_len
        if windows_per_trajectory <= 0:
            raise ValueError("history_len leaves no eligible one-step windows")
        if max_windows_per_trajectory is not None:
            windows_per_trajectory = min(windows_per_trajectory, int(max_windows_per_trajectory))
        self.windows = [
            (trajectory_id, target_time_index)
            for trajectory_id in self.trajectory_ids
            for target_time_index in range(self.history_len, self.history_len + windows_per_trajectory)
        ]

    def __len__(self) -> int:
        return len(self.windows)

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

    def _read_state(self, trajectory_id: int, time_index: int) -> torch.Tensor:
        array = read_dynamic_state_channel_first(
            self._h5(),
            state_dataset=self.state_dataset,
            trajectory_id=trajectory_id,
            time_index=time_index,
            axis_order=self.axis_order,
            trajectory_paths=self._trajectory_paths,
        )
        return torch.from_numpy(array).float()

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
        trajectory_id, target_time_index = self.windows[index]
        input_time_indices = list(range(target_time_index - self.history_len, target_time_index))
        states = [
            self._apply_normalization(self._read_state(trajectory_id, time_index))
            for time_index in input_time_indices
        ]
        target = self._apply_normalization(self._read_state(trajectory_id, target_time_index))
        model_input = torch.cat(states, dim=0)
        model_input, pad = self._pad(model_input)
        target, _ = self._pad(target)
        return {
            "input": model_input,
            "target": target,
            "trajectory_id": int(trajectory_id),
            "target_time_index": int(target_time_index),
            "input_time_indices": [int(item) for item in input_time_indices],
            "history_len": int(self.history_len),
            "pad": pad,
        }


class MultiFieldHistoryWindowDataset(Dataset):
    """Lazy HDF5 dynamic dataset that stacks separate physical-field datasets."""

    def __init__(
        self,
        *,
        data_file: Path,
        field_order: Sequence[str],
        trajectory_ids: Sequence[int],
        axis_order: str,
        history_len: int,
        normalization: dict[str, Any] | None = None,
        max_windows_per_trajectory: int | None = None,
        pad_multiple: int = 1,
    ):
        self.data_file = Path(data_file)
        self.field_order = [str(item) for item in field_order]
        self.trajectory_ids = [int(item) for item in trajectory_ids]
        self.axis_order = axis_order.upper()
        self.history_len = int(history_len)
        if self.history_len < 1:
            raise ValueError("history_len must be at least 1")
        self.normalization = normalization
        self.pad_multiple = max(1, int(pad_multiple))
        self._handle: h5py.File | None = None

        with h5py.File(self.data_file, "r") as handle:
            missing = [field for field in self.field_order if field not in handle]
            if missing:
                raise KeyError(f"missing multi-field datasets: {missing}")
            shapes = {tuple(handle[field].shape) for field in self.field_order}
            if len(shapes) != 1:
                raise ValueError("all field datasets must share one shape")
            shape = [int(item) for item in next(iter(shapes))]
        self.field_shape = shape
        self.dimensions = infer_dynamic_dimensions(shape, self.axis_order)
        windows_per_trajectory = self.dimensions["time_steps"] - self.history_len
        if windows_per_trajectory <= 0:
            raise ValueError("history_len leaves no eligible one-step windows")
        if max_windows_per_trajectory is not None:
            windows_per_trajectory = min(windows_per_trajectory, int(max_windows_per_trajectory))
        self.windows = [
            (trajectory_id, target_time_index)
            for trajectory_id in self.trajectory_ids
            for target_time_index in range(self.history_len, self.history_len + windows_per_trajectory)
        ]

    def __len__(self) -> int:
        return len(self.windows)

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

    def _read_state(self, trajectory_id: int, time_index: int) -> torch.Tensor:
        handle = self._h5()
        state_channels = []
        for field in self.field_order:
            dataset = handle[field]
            selection: list[Any] = [slice(None)] * len(self.axis_order)
            selection[axis_index(self.axis_order, "N")] = int(trajectory_id)
            selection[axis_index(self.axis_order, "T")] = int(time_index)
            array = np.asarray(dataset[tuple(selection)], dtype=np.float32)
            remaining_axes = [axis for axis in self.axis_order if axis not in {"N", "T"}]
            state_channels.append(torch.from_numpy(_channel_first(array, "".join(remaining_axes))).float())
        return torch.cat(state_channels, dim=0)

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
        trajectory_id, target_time_index = self.windows[index]
        input_time_indices = list(range(target_time_index - self.history_len, target_time_index))
        states = [
            self._apply_normalization(self._read_state(trajectory_id, time_index))
            for time_index in input_time_indices
        ]
        target = self._apply_normalization(self._read_state(trajectory_id, target_time_index))
        model_input = torch.cat(states, dim=0)
        model_input, pad = self._pad(model_input)
        target, _ = self._pad(target)
        return {
            "input": model_input,
            "target": target,
            "trajectory_id": int(trajectory_id),
            "target_time_index": int(target_time_index),
            "input_time_indices": [int(item) for item in input_time_indices],
            "history_len": int(self.history_len),
            "field_order": list(self.field_order),
            "pad": pad,
        }
