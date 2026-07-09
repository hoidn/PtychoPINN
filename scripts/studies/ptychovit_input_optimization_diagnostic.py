#!/usr/bin/env python3
"""Stationary-point input-space diagnostic for PtychoViT bridge inputs."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def project_input(
    x: torch.Tensor,
    *,
    min_value: float = 0.0,
    max_value: float | None = None,
) -> torch.Tensor:
    """Project an input tensor into configured bounds."""
    if max_value is None:
        return torch.clamp(x, min=min_value)
    return torch.clamp(x, min=min_value, max=max_value)


def is_stationary(grad_norm: float, threshold: float) -> bool:
    """Return True when gradient norm is at or below threshold."""
    return float(grad_norm) <= float(threshold)


def _total_variation(x: torch.Tensor | None) -> torch.Tensor:
    if x is None:
        return torch.tensor(0.0)
    if x.ndim < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    tv_h = torch.mean(torch.abs(x[..., 1:, :] - x[..., :-1, :]))
    tv_w = torch.mean(torch.abs(x[..., :, 1:] - x[..., :, :-1]))
    return tv_h + tv_w


def compute_objective(
    *,
    pred_amp: torch.Tensor,
    pred_phase: torch.Tensor,
    weights: Mapping[str, float],
    pred_diff_amp: torch.Tensor | None = None,
    target_diff_amp: torch.Tensor | None = None,
    input_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted objective and report component scalars."""
    amp_var = torch.var(pred_amp)
    phase_var = torch.var(pred_phase)
    tv = _total_variation(input_tensor)

    if pred_diff_amp is not None and target_diff_amp is not None:
        forward_consistency = -torch.mean((pred_diff_amp - target_diff_amp) ** 2)
    else:
        forward_consistency = torch.tensor(0.0, device=pred_amp.device, dtype=pred_amp.dtype)

    objective = (
        float(weights.get("amp_var", 0.0)) * amp_var
        + float(weights.get("phase_var", 0.0)) * phase_var
        - float(weights.get("tv", 0.0)) * tv
        + float(weights.get("forward_consistency", 0.0)) * forward_consistency
    )

    components = {
        "amp_var": float(amp_var.detach().cpu().item()),
        "phase_var": float(phase_var.detach().cpu().item()),
        "tv": float(tv.detach().cpu().item()),
        "forward_consistency": float(forward_consistency.detach().cpu().item()),
        "total": float(objective.detach().cpu().item()),
    }
    return objective, components


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _resolve_normalization_dict_path(args: argparse.Namespace) -> Path | None:
    if args.normalization_dict_path is not None:
        return args.normalization_dict_path

    candidate = args.test_dp.parent / "normalization.pkl"
    if candidate.exists():
        return candidate
    return None


def _load_runtime_classes(ptychovit_repo: Path):
    repo_str = str(ptychovit_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    data_module = importlib.import_module("data")
    model_module = importlib.import_module("model.model")
    return data_module.PtychographyDataset, model_module.PtychoViT


def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping) and "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        return payload["state_dict"]
    if isinstance(payload, Mapping):
        return payload
    raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    y = x.detach().to(dtype=torch.float32)
    return {
        "min": float(torch.min(y).cpu().item()),
        "max": float(torch.max(y).cpu().item()),
        "mean": float(torch.mean(y).cpu().item()),
        "std": float(torch.std(y, unbiased=False).cpu().item()),
    }


def _to_2d_image(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _write_stationary_point_png(
    *,
    output_dir: Path,
    input_tensor: torch.Tensor,
    pred_amp: torch.Tensor,
    pred_phase: torch.Tensor,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stationary_point.png"

    input_img = _to_2d_image(input_tensor)
    amp_img = _to_2d_image(pred_amp)
    phase_img = _to_2d_image(pred_phase)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = (
        ("Optimized Input Diffraction Amp", input_img, "magma"),
        ("Predicted Amplitude", amp_img, "viridis"),
        ("Predicted Phase", phase_img, "twilight"),
    )
    for ax, (title, image, cmap) in zip(axes, panels):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = args.ptychovit_repo / "config.yaml"
    config = _load_yaml(cfg_path) if cfg_path.exists() else {}
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    normalization_dict_path = _resolve_normalization_dict_path(args)
    normalization_dict_arg = str(normalization_dict_path) if normalization_dict_path is not None else None

    dataset_cls, model_cls = _load_runtime_classes(args.ptychovit_repo)
    dataset = dataset_cls(
        file_path=str(args.test_dp),
        scale=float(data_cfg.get("scale", 10000.0)),
        normalization_dict_path=normalization_dict_arg,
        apply_noise=False,
        cache_object=bool(data_cfg.get("cache_object", False)),
        max_probe_modes=int(data_cfg.get("max_probe_modes", 8)),
    )
    if len(dataset) == 0:
        raise RuntimeError("Diagnostic dataset is empty")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    if len(batch) != 7:
        raise ValueError(f"Expected dataset batch of length 7, got {len(batch)}")

    diff_amp, _amp_patch, _phase_patch, probe, _probe_position, normalization, scale = batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(config=model_cfg)
    state_dict = _extract_state_dict(torch.load(args.checkpoint, map_location=device))
    load_state_result = model.load_state_dict(state_dict, strict=False)
    _ = load_state_result
    model.to(device)
    model.eval()

    input_x = diff_amp.to(device=device, dtype=torch.float32).clone().detach()
    with torch.no_grad():
        input_x.copy_(project_input(input_x, min_value=0.0, max_value=args.input_max))
    input_x.requires_grad_(True)

    probe_tensor = probe.clone().detach().to(device)
    if torch.is_complex(probe_tensor):
        probe_tensor = torch.view_as_real(probe_tensor)
    elif probe_tensor.shape[-1] != 2:
        raise ValueError(
            "Probe tensor must be complex (converted via view_as_real) or already have trailing real/imag axis"
        )

    normalization_tensor = torch.as_tensor(normalization, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([input_x], lr=float(args.lr))
    weights = {
        "amp_var": float(args.w_amp_var),
        "phase_var": float(args.w_phase_var),
        "tv": float(args.w_tv),
        "forward_consistency": float(args.w_forward_consistency),
    }

    objective_history: list[float] = []
    grad_norm_history: list[float] = []
    input_stats: list[dict[str, float]] = []
    amp_stats: list[dict[str, float]] = []
    phase_stats: list[dict[str, float]] = []
    objective_components_history: list[dict[str, float]] = []
    stationary_step: int | None = None
    final_input_x: torch.Tensor | None = None
    final_pred_amp: torch.Tensor | None = None
    final_pred_phase: torch.Tensor | None = None

    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)

        pred_diff_amp, pred_amp, pred_phase = model(input_x, probe_tensor, normalization_tensor, scale_tensor)
        pred_diff_amp = pred_diff_amp.squeeze(1)
        pred_amp = pred_amp.squeeze(1)
        pred_phase = pred_phase.squeeze(1)

        objective, components = compute_objective(
            pred_amp=pred_amp,
            pred_phase=pred_phase,
            weights=weights,
            pred_diff_amp=pred_diff_amp,
            target_diff_amp=input_x.squeeze(1),
            input_tensor=input_x.squeeze(1),
        )

        (-objective).backward()
        grad = input_x.grad
        grad_norm = float(torch.linalg.norm(grad).detach().cpu().item()) if grad is not None else 0.0

        optimizer.step()
        with torch.no_grad():
            input_x.copy_(project_input(input_x, min_value=0.0, max_value=args.input_max))

        objective_history.append(float(objective.detach().cpu().item()))
        grad_norm_history.append(grad_norm)
        input_stats.append(_tensor_stats(input_x))
        amp_stats.append(_tensor_stats(pred_amp))
        phase_stats.append(_tensor_stats(pred_phase))
        objective_components_history.append(components)
        final_input_x = input_x.detach().cpu().clone()
        final_pred_amp = pred_amp.detach().cpu().clone()
        final_pred_phase = pred_phase.detach().cpu().clone()

        if stationary_step is None and is_stationary(grad_norm, float(args.stationary_threshold)):
            stationary_step = step
            break

    if stationary_step is None:
        stationary_step = int(args.steps)

    normalization_value = float(getattr(dataset, "normalization", normalization_tensor.flatten()[0].item()))
    scale_value = float(getattr(dataset, "scale", scale_tensor.flatten()[0].item()))
    if final_input_x is None or final_pred_amp is None or final_pred_phase is None:
        raise RuntimeError("Diagnostic produced no optimization steps; unable to render stationary-point image")
    stationary_png_path = _write_stationary_point_png(
        output_dir=args.output_dir,
        input_tensor=final_input_x,
        pred_amp=final_pred_amp,
        pred_phase=final_pred_phase,
    )

    return {
        "objective_history": objective_history,
        "grad_norm_history": grad_norm_history,
        "stationary_step": int(stationary_step),
        "input_stats": input_stats,
        "amp_stats": amp_stats,
        "phase_stats": phase_stats,
        "objective_components": objective_components_history,
        "normalization_context": {
            "normalization": normalization_value,
            "scale": scale_value,
            "normalization_dict_path": normalization_dict_arg,
            "object_name": getattr(dataset, "object_name", None),
            "test_dp": str(args.test_dp),
            "test_para": str(args.test_para),
        },
        "config": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "stationary_threshold": float(args.stationary_threshold),
            "input_max": float(args.input_max) if args.input_max is not None else None,
            "weights": weights,
            "checkpoint": str(args.checkpoint),
            "ptychovit_repo": str(args.ptychovit_repo),
        },
        "artifacts": {
            "stationary_point_png": str(stationary_png_path),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PtychoViT input-space stationary-point diagnostic")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ptychovit-repo", type=Path, required=True)
    parser.add_argument("--test-dp", type=Path, required=True)
    parser.add_argument("--test-para", type=Path, required=True)
    parser.add_argument("--normalization-dict-path", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--stationary-threshold", type=float, default=1.0e-5)
    parser.add_argument("--input-max", type=float, default=100.0)
    parser.add_argument("--w-amp-var", type=float, default=1.0)
    parser.add_argument("--w-phase-var", type=float, default=1.0)
    parser.add_argument("--w-tv", type=float, default=0.1)
    parser.add_argument("--w-forward-consistency", type=float, default=1.0)
    return parser.parse_args(argv)


def _require_positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0 (got {value})")


def _require_nonnegative(name: str, value: float) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0 (got {value})")


def validate_args(args: argparse.Namespace) -> None:
    for path_name in ("checkpoint", "ptychovit_repo", "test_dp", "test_para"):
        path = getattr(args, path_name)
        if not Path(path).exists():
            raise FileNotFoundError(f"{path_name} not found: {path}")

    _require_positive("steps", float(args.steps))
    _require_positive("lr", float(args.lr))
    _require_positive("stationary-threshold", float(args.stationary_threshold))

    if args.input_max is not None:
        _require_positive("input-max", float(args.input_max))

    _require_nonnegative("w-amp-var", float(args.w_amp_var))
    _require_nonnegative("w-phase-var", float(args.w_phase_var))
    _require_nonnegative("w-tv", float(args.w_tv))
    _require_nonnegative("w-forward-consistency", float(args.w_forward_consistency))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_diagnostic(args)
    report_path = args.output_dir / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote diagnostic report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
