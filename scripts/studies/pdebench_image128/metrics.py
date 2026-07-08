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


def _frequency_band_masks(
    *,
    height: int,
    width: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fy = torch.fft.fftshift(torch.fft.fftfreq(int(height), device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(int(width), device=device))
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    max_radius = torch.clamp(radius.max(), min=1e-12)
    normalized = radius / max_radius
    return {
        "low": normalized <= (1.0 / 3.0),
        "mid": (normalized > (1.0 / 3.0)) & (normalized <= (2.0 / 3.0)),
        "high": normalized > (2.0 / 3.0),
    }


def fourier_rmse_bands(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute FFT-domain RMSE over fixed low/mid/high radial bands.

    This local CNS convention uses `torch.fft.fft2(..., norm="ortho")` and
    three fftshifted radial-frequency thirds so the band definition is fixed
    across Hybrid ResNet, FNO, and U-Net rows.
    """
    prediction = prediction.detach().float()
    target = target.detach().float()
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.ndim != 4:
        raise ValueError("fourier_rmse_bands expects tensors shaped (N,C,H,W)")
    _, _, height, width = prediction.shape
    spectrum_error = torch.fft.fftshift(
        torch.fft.fft2(prediction - target, dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )
    power = spectrum_error.abs().square()
    masks = _frequency_band_masks(height=int(height), width=int(width), device=power.device)
    per_channel: dict[str, list[float]] = {}
    payload: dict[str, float | str | dict] = {
        "band_definition": "fftshifted_radial_frequency_thirds",
        "fft_norm": "ortho",
        "band_counts": {name: int(mask.sum().item()) for name, mask in masks.items()},
    }
    for name, mask in masks.items():
        if not bool(mask.any()):
            aggregate = torch.tensor(0.0, dtype=power.dtype, device=power.device)
            channel_values = torch.zeros(power.shape[1], dtype=power.dtype, device=power.device)
        else:
            band_power = power[..., mask]
            aggregate = torch.sqrt(torch.mean(band_power))
            channel_values = torch.sqrt(torch.mean(band_power, dim=(0, 2)))
        key = f"fRMSE_{name}"
        payload[key] = _round_float(aggregate)
        per_channel[key] = [float(item) for item in channel_values.detach().cpu().tolist()]
    payload["per_channel"] = per_channel
    return payload


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


def dynamic_state_metric_payload(
    prediction_batches: Iterable[torch.Tensor],
    target_batches: Iterable[torch.Tensor],
    *,
    normalized: bool,
    state_stats: dict | None,
) -> dict:
    predictions = [batch.detach().cpu().float() for batch in prediction_batches]
    targets = [batch.detach().cpu().float() for batch in target_batches]
    if not predictions or not targets:
        raise ValueError("metric payload requires at least one prediction and target batch")
    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    metric_units = "raw_state_units"
    if normalized:
        if state_stats is None:
            raise ValueError("state_stats are required for normalized dynamic-state metrics")
        prediction = denormalize_batch(prediction, state_stats)
        target = denormalize_batch(target, state_stats)
        metric_units = "denormalized_state_units"

    per_channel_dims = (0, 2, 3)
    per_channel_rmse = err_rmse(prediction, target, dims=per_channel_dims)
    per_channel_nrmse = err_nrmse(prediction, target, dims=per_channel_dims)
    fourier_payload = fourier_rmse_bands(prediction, target)
    return {
        "err_RMSE": _round_float(err_rmse(prediction, target)),
        "err_nRMSE": _round_float(err_nrmse(prediction, target)),
        "relative_l2": _round_float(err_nrmse(prediction, target)),
        "fRMSE_low": fourier_payload["fRMSE_low"],
        "fRMSE_mid": fourier_payload["fRMSE_mid"],
        "fRMSE_high": fourier_payload["fRMSE_high"],
        "fourier_metric_units": f"{metric_units}_fft_ortho",
        "fourier_band_definition": fourier_payload["band_definition"],
        "fourier_band_counts": fourier_payload["band_counts"],
        "per_channel": {
            "err_RMSE": [float(item) for item in per_channel_rmse.tolist()],
            "err_nRMSE": [float(item) for item in per_channel_nrmse.tolist()],
            "relative_l2": [float(item) for item in per_channel_nrmse.tolist()],
            **fourier_payload["per_channel"],
        },
        "num_eval_batches": len(predictions),
        "num_eval_samples": int(prediction.shape[0]),
        "normalization": "train_split_state_stats" if state_stats is not None else "none",
        "metric_units": metric_units,
        "horizon": "one_step",
    }
