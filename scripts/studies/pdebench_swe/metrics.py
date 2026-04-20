"""Local normalization and metric contract for PDEBench SWE smoke runs."""

from __future__ import annotations

from typing import Iterable

import torch


def _stats_tensors(stats: dict, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats["mean"], dtype=dtype, device=device).view(1, -1, 1, 1)
    std = torch.tensor(stats["std"], dtype=dtype, device=device).view(1, -1, 1, 1)
    return mean, torch.clamp(std, min=1e-12)


def compute_channel_stats(dataset, *, max_batches: int | None = None) -> dict:
    """Compute per-channel train-split stats from dataset inputs only."""
    sums: torch.Tensor | None = None
    sums_sq: torch.Tensor | None = None
    count = 0
    num_samples = 0
    limit = len(dataset) if max_batches is None else min(len(dataset), int(max_batches))
    for index in range(limit):
        tensor = dataset[index]["input"].float()
        channels = tensor.shape[0]
        flat = tensor.view(channels, -1)
        if sums is None:
            sums = torch.zeros(channels, dtype=torch.float64)
            sums_sq = torch.zeros(channels, dtype=torch.float64)
        sums += flat.double().sum(dim=1)
        sums_sq += flat.double().square().sum(dim=1)
        count += flat.shape[1]
        num_samples += 1
    if sums is None or sums_sq is None or count == 0:
        raise ValueError("cannot compute normalization stats from an empty dataset")
    mean = sums / count
    variance = torch.clamp((sums_sq / count) - mean.square(), min=0.0)
    std = torch.sqrt(variance)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    return {
        "schema_version": "pdebench_swe_normalization_stats_v1",
        "mean": [float(item) for item in mean.tolist()],
        "std": [float(item) for item in std.tolist()],
        "num_samples": int(num_samples),
        "num_values_per_channel": int(count),
        "source": "train_split_inputs_only",
    }


def normalize_batch(batch: torch.Tensor, stats: dict) -> torch.Tensor:
    mean, std = _stats_tensors(stats, dtype=batch.dtype, device=batch.device)
    return (batch - mean) / std


def denormalize_batch(batch: torch.Tensor, stats: dict) -> torch.Tensor:
    mean, std = _stats_tensors(stats, dtype=batch.dtype, device=batch.device)
    return batch * std + mean


def err_rmse(prediction: torch.Tensor, target: torch.Tensor, *, dims=None) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - target).square(), dim=dims))


def err_nrmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    dims=None,
    eps: float = 1e-12,
) -> torch.Tensor:
    numerator = torch.sum((prediction - target).square(), dim=dims)
    denominator = torch.clamp(torch.sum(target.square(), dim=dims), min=eps)
    return torch.sqrt(numerator / denominator)


def _round_float(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def metric_payload(
    prediction_batches: Iterable[torch.Tensor],
    target_batches: Iterable[torch.Tensor],
    *,
    normalized: bool,
    stats: dict | None,
) -> dict:
    predictions = [batch.detach().cpu().float() for batch in prediction_batches]
    targets = [batch.detach().cpu().float() for batch in target_batches]
    if not predictions or not targets:
        raise ValueError("metric_payload requires at least one prediction and target batch")
    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    metric_units = "normalized_units"
    if normalized and stats is not None:
        prediction = denormalize_batch(prediction, stats)
        target = denormalize_batch(target, stats)
        metric_units = "denormalized_physical_units"
    elif not normalized:
        metric_units = "physical_or_raw_units"

    per_channel_dims = (0, 2, 3)
    per_channel_rmse = err_rmse(prediction, target, dims=per_channel_dims)
    per_channel_nrmse = err_nrmse(prediction, target, dims=per_channel_dims)
    return {
        "err_RMSE": _round_float(err_rmse(prediction, target)),
        "err_nRMSE": _round_float(err_nrmse(prediction, target)),
        "per_channel": {
            "err_RMSE": [float(item) for item in per_channel_rmse.tolist()],
            "err_nRMSE": [float(item) for item in per_channel_nrmse.tolist()],
        },
        "num_eval_batches": len(predictions),
        "num_eval_pairs": int(prediction.shape[0]),
        "normalization": "train_split_stats" if stats is not None else "none",
        "metric_units": metric_units,
        "horizon": "one_step",
    }
