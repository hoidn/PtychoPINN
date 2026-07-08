"""MAE/RMSE/SSIM metrics for OpenFWI FlatVel-A velocity maps."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _denormalize_target(batch: torch.Tensor, stats: dict) -> torch.Tensor:
    target_stats = stats.get("target", stats)
    mean = float(target_stats["mean"])
    std = max(float(target_stats["std"]), 1e-12)
    return batch * std + mean


def _as_float(value: torch.Tensor | float) -> float:
    result = float(torch.as_tensor(value).detach().cpu().item())
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric value: {result}")
    return result


def _ssim_one(prediction: np.ndarray, target: np.ndarray) -> float:
    data_min = float(min(prediction.min(), target.min()))
    data_max = float(max(prediction.max(), target.max()))
    data_range = data_max - data_min
    if data_range <= 1e-12:
        data_range = 1.0
    min_side = min(prediction.shape[-2:])
    if min_side < 3:
        if np.array_equal(prediction, target):
            return 1.0
        mae = float(np.mean(np.abs(prediction - target)))
        return float(max(-1.0, min(1.0, 1.0 - mae / data_range)))
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    return float(structural_similarity(target, prediction, data_range=data_range, win_size=win_size))


def metric_payload(
    prediction_batches: Iterable[torch.Tensor],
    target_batches: Iterable[torch.Tensor],
    *,
    normalized: bool,
    target_stats: dict | None,
) -> dict:
    """Compute MAE/RMSE/SSIM on denormalized velocity maps when requested."""
    predictions = [batch.detach().cpu().float() for batch in prediction_batches]
    targets = [batch.detach().cpu().float() for batch in target_batches]
    if not predictions or not targets:
        raise ValueError("metric_payload requires at least one prediction and target batch")
    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"OpenFWI metrics expect shape (B,1,H,W), got {tuple(prediction.shape)}")

    metric_units = "raw_velocity_or_model_units"
    normalization = "none"
    if normalized and target_stats is not None:
        prediction = _denormalize_target(prediction, target_stats)
        target = _denormalize_target(target, target_stats)
        metric_units = "denormalized_velocity"
        normalization = "train_split_stats"

    absolute = (prediction - target).abs()
    squared = (prediction - target).square()
    per_sample_mae = absolute.flatten(1).mean(dim=1)
    per_sample_rmse = torch.sqrt(squared.flatten(1).mean(dim=1))
    ssim_values = [
        _ssim_one(prediction[index, 0].numpy(), target[index, 0].numpy())
        for index in range(prediction.shape[0])
    ]
    return {
        "MAE": _as_float(absolute.mean()),
        "RMSE": _as_float(torch.sqrt(squared.mean())),
        "SSIM": float(np.mean(ssim_values)),
        "per_sample": {
            "MAE": [float(item) for item in per_sample_mae.tolist()],
            "RMSE": [float(item) for item in per_sample_rmse.tolist()],
            "SSIM": [float(item) for item in ssim_values],
        },
        "num_eval_samples": int(prediction.shape[0]),
        "metric_units": metric_units,
        "target_normalization": normalization,
        "ssim_data_range_policy": "per_sample_combined_prediction_target_range_min_1",
    }
