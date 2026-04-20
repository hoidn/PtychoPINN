"""Normalization helpers for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import torch

from scripts.studies.pdebench_image128.data import read_sample_channel_first


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
