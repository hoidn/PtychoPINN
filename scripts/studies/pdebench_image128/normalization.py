"""Normalization helpers for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import torch

from scripts.studies.pdebench_image128.data import (
    grouped_trajectory_paths,
    read_dynamic_state_channel_first,
    read_sample_channel_first,
)
from scripts.studies.pdebench_image128.splits import infer_dynamic_dimensions


def _stats_from_samples(
    *,
    data_file: Path,
    dataset_name: str,
    axis_order: str,
    sample_indices: Sequence[int],
    source: str,
) -> dict[str, Any]:
    sums: torch.Tensor | None = None
    sums_sq: torch.Tensor | None = None
    count = 0
    with h5py.File(data_file, "r") as handle:
        dataset = handle[dataset_name]
        for sample_index in sample_indices:
            tensor = torch.from_numpy(read_sample_channel_first(dataset, int(sample_index), axis_order)).double()
            channels = tensor.shape[0]
            flat = tensor.view(channels, -1)
            if sums is None:
                sums = torch.zeros(channels, dtype=torch.float64)
                sums_sq = torch.zeros(channels, dtype=torch.float64)
            sums += flat.sum(dim=1)
            sums_sq += flat.square().sum(dim=1)
            count += flat.shape[1]
    if sums is None or sums_sq is None or count == 0:
        raise ValueError("cannot compute normalization stats from empty sample set")
    mean = sums / count
    variance = torch.clamp((sums_sq / count) - mean.square(), min=0.0)
    std = torch.sqrt(variance)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    return {
        "schema_version": "pdebench_image128_normalization_stats_v1",
        "mean": [float(item) for item in mean.tolist()],
        "std": [float(item) for item in std.tolist()],
        "num_samples": int(len(sample_indices)),
        "num_values_per_channel": int(count),
        "source": source,
        "dataset": dataset_name,
        "axis_order": axis_order,
    }


def compute_static_operator_stats(
    *,
    data_file: Path,
    input_dataset: str,
    target_dataset: str,
    train_indices: Sequence[int],
    input_axis_order: str = "NHW",
    target_axis_order: str = "NCHW",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute separate train-only input and target stats for static operator maps."""
    indices = [int(item) for item in train_indices]
    input_stats = _stats_from_samples(
        data_file=data_file,
        dataset_name=input_dataset,
        axis_order=input_axis_order,
        sample_indices=indices,
        source="train_split_inputs_only",
    )
    target_stats = _stats_from_samples(
        data_file=data_file,
        dataset_name=target_dataset,
        axis_order=target_axis_order,
        sample_indices=indices,
        source="train_split_targets_only",
    )
    return input_stats, target_stats


def compute_dynamic_state_stats(
    *,
    data_file: Path,
    state_dataset: str,
    axis_order: str,
    train_trajectory_ids: Sequence[int],
) -> dict[str, Any]:
    """Compute train-only per-physical-channel stats for dynamic state datasets."""
    trajectory_ids = [int(item) for item in train_trajectory_ids]
    sums: torch.Tensor | None = None
    sums_sq: torch.Tensor | None = None
    count = 0
    with h5py.File(data_file, "r") as handle:
        trajectory_paths = grouped_trajectory_paths(handle, state_dataset)
        if trajectory_paths is not None:
            first_path = trajectory_paths[sorted(trajectory_paths)[0]]
            shape = [len(trajectory_paths), *[int(item) for item in handle[first_path].shape]]
        else:
            shape = [int(item) for item in handle[state_dataset.strip("/")].shape]
        dims = infer_dynamic_dimensions(shape, axis_order)
        for trajectory_id in trajectory_ids:
            for time_index in range(dims["time_steps"]):
                tensor = torch.from_numpy(
                    read_dynamic_state_channel_first(
                        handle,
                        state_dataset=state_dataset,
                        trajectory_id=trajectory_id,
                        time_index=time_index,
                        axis_order=axis_order,
                        trajectory_paths=trajectory_paths,
                    )
                ).double()
                channels = tensor.shape[0]
                flat = tensor.view(channels, -1)
                if sums is None:
                    sums = torch.zeros(channels, dtype=torch.float64)
                    sums_sq = torch.zeros(channels, dtype=torch.float64)
                sums += flat.sum(dim=1)
                sums_sq += flat.square().sum(dim=1)
                count += flat.shape[1]
    if sums is None or sums_sq is None or count == 0:
        raise ValueError("cannot compute normalization stats from empty trajectory set")
    mean = sums / count
    variance = torch.clamp((sums_sq / count) - mean.square(), min=0.0)
    std = torch.sqrt(variance)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    return {
        "schema_version": "pdebench_image128_dynamic_state_normalization_stats_v1",
        "mean": [float(item) for item in mean.tolist()],
        "std": [float(item) for item in std.tolist()],
        "num_trajectories": int(len(trajectory_ids)),
        "num_time_steps_per_trajectory": int(dims["time_steps"]),
        "num_values_per_channel": int(count),
        "source": "train_split_trajectories_all_time_steps",
        "dataset": state_dataset,
        "axis_order": axis_order,
    }


def compute_multifield_dynamic_stats(
    *,
    data_file: Path,
    field_order: Sequence[str],
    axis_order: str,
    train_trajectory_ids: Sequence[int],
) -> dict[str, Any]:
    """Compute train-only per-field stats for separate-field dynamic datasets."""
    trajectory_ids = [int(item) for item in train_trajectory_ids]
    field_order = [str(item) for item in field_order]
    sums = torch.zeros(len(field_order), dtype=torch.float64)
    sums_sq = torch.zeros(len(field_order), dtype=torch.float64)
    count = 0
    with h5py.File(data_file, "r") as handle:
        missing = [field for field in field_order if field not in handle]
        if missing:
            raise KeyError(f"missing multi-field datasets: {missing}")
        shapes = {tuple(handle[field].shape) for field in field_order}
        if len(shapes) != 1:
            raise ValueError("all field datasets must share one shape")
        shape = [int(item) for item in next(iter(shapes))]
        dims = infer_dynamic_dimensions(shape, axis_order)
        n_axis = axis_order.index("N")
        t_axis = axis_order.index("T")
        for field_index, field in enumerate(field_order):
            dataset = handle[field]
            for trajectory_id in trajectory_ids:
                for time_index in range(dims["time_steps"]):
                    selection: list[Any] = [slice(None)] * len(axis_order)
                    selection[n_axis] = int(trajectory_id)
                    selection[t_axis] = int(time_index)
                    tensor = torch.as_tensor(dataset[tuple(selection)], dtype=torch.float64)
                    sums[field_index] += tensor.sum()
                    sums_sq[field_index] += tensor.square().sum()
                    if field_index == 0:
                        count += tensor.numel()
    if count == 0:
        raise ValueError("cannot compute normalization stats from empty trajectory set")
    mean = sums / count
    variance = torch.clamp((sums_sq / count) - mean.square(), min=0.0)
    std = torch.sqrt(variance)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    return {
        "schema_version": "pdebench_image128_multifield_dynamic_normalization_stats_v1",
        "mean": [float(item) for item in mean.tolist()],
        "std": [float(item) for item in std.tolist()],
        "field_order": field_order,
        "num_trajectories": int(len(trajectory_ids)),
        "num_time_steps_per_trajectory": int(dims["time_steps"]),
        "num_values_per_field": int(count),
        "source": "train_split_trajectories_all_time_steps",
        "axis_order": axis_order,
    }


def _stats_tensors(stats: dict[str, Any], *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats["mean"], dtype=dtype, device=device).view(1, -1, 1, 1)
    std = torch.tensor(stats["std"], dtype=dtype, device=device).view(1, -1, 1, 1)
    return mean, torch.clamp(std, min=1e-12)


def normalize_batch(batch: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    mean, std = _stats_tensors(stats, dtype=batch.dtype, device=batch.device)
    return (batch - mean) / std


def denormalize_batch(batch: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    mean, std = _stats_tensors(stats, dtype=batch.dtype, device=batch.device)
    return batch * std + mean
