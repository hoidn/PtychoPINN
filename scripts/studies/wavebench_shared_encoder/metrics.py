from __future__ import annotations

from typing import Any

import torch


def compute_reconstruction_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    skimage_metrics: Any | None = None,
) -> dict[str, float]:
    diff = predictions - targets
    mae = float(torch.mean(torch.abs(diff)).item())
    rmse = float(torch.sqrt(torch.mean(diff**2)).item())
    pred_flat = predictions.reshape(predictions.shape[0], -1)
    target_flat = targets.reshape(targets.shape[0], -1)
    rel_l2 = float(
        torch.mean(
            torch.norm(pred_flat - target_flat, dim=1)
            / torch.norm(target_flat, dim=1).clamp_min(1e-12)
        ).item()
    )

    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    ssim_values: list[float] = []
    for index in range(predictions.shape[0]):
        pred_2d = pred_np[index, 0]
        target_2d = target_np[index, 0]
        if skimage_metrics is None:
            pred_centered = pred_2d - pred_2d.mean()
            target_centered = target_2d - target_2d.mean()
            numerator = (2.0 * (pred_centered * target_centered).mean()) + 1e-8
            denominator = (pred_centered**2).mean() + (target_centered**2).mean() + 1e-8
            ssim_values.append(float(numerator / denominator))
            continue
        data_min = float(min(pred_2d.min(), target_2d.min()))
        data_max = float(max(pred_2d.max(), target_2d.max()))
        data_range = max(data_max - data_min, 1e-6)
        ssim_values.append(
            float(
                skimage_metrics.structural_similarity(
                    pred_2d,
                    target_2d,
                    data_range=data_range,
                )
            )
        )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "RelL2": rel_l2,
        "SSIM": float(sum(ssim_values) / len(ssim_values)),
    }

