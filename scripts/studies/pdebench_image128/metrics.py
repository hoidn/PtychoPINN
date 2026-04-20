"""Metrics for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

from typing import Iterable

import torch

from scripts.studies.pdebench_image128.normalization import denormalize_batch


def err_rmse(prediction: torch.Tensor, target: torch.Tensor, *, dims=None) -> torch.Tensor:
    return torch.sqrt(torch.mean((prediction - target).square(), dim=dims))


def err_nrmse(prediction: torch.Tensor, target: torch.Tensor, *, dims=None, eps: float = 1e-12) -> torch.Tensor:
    numerator = torch.sum((prediction - target).square(), dim=dims)
    denominator = torch.clamp(torch.sum(target.square(), dim=dims), min=eps)
    return torch.sqrt(numerator / denominator)


def _round_float(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def static_operator_metric_payload(
    prediction_batches: Iterable[torch.Tensor],
    target_batches: Iterable[torch.Tensor],
    *,
    normalized: bool,
    target_stats: dict | None,
) -> dict:
    predictions = [batch.detach().cpu().float() for batch in prediction_batches]
    targets = [batch.detach().cpu().float() for batch in target_batches]
    if not predictions or not targets:
        raise ValueError("metric payload requires at least one prediction and target batch")
    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    metric_units = "raw_target_units"
    if normalized:
        if target_stats is None:
            raise ValueError("target_stats are required for normalized static-operator metrics")
        prediction = denormalize_batch(prediction, target_stats)
        target = denormalize_batch(target, target_stats)
        metric_units = "denormalized_target_units"

    per_channel_dims = (0, 2, 3)
    per_channel_rmse = err_rmse(prediction, target, dims=per_channel_dims)
    per_channel_nrmse = err_nrmse(prediction, target, dims=per_channel_dims)
    return {
        "err_RMSE": _round_float(err_rmse(prediction, target)),
        "err_nRMSE": _round_float(err_nrmse(prediction, target)),
        "relative_l2": _round_float(err_nrmse(prediction, target)),
        "per_channel": {
            "err_RMSE": [float(item) for item in per_channel_rmse.tolist()],
            "err_nRMSE": [float(item) for item in per_channel_nrmse.tolist()],
            "relative_l2": [float(item) for item in per_channel_nrmse.tolist()],
        },
        "num_eval_batches": len(predictions),
        "num_eval_samples": int(prediction.shape[0]),
        "normalization": "train_split_target_stats" if target_stats is not None else "none",
        "metric_units": metric_units,
        "horizon": "static_operator",
    }
