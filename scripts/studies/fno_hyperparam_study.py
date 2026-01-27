#!/usr/bin/env python3
"""Hyperparameter sweep for FNO/Hybrid grid-lines runs."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np

from ptycho.workflows.grid_lines_workflow import GridLinesConfig, run_grid_lines_workflow
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_grid_lines_torch


DEFAULT_N = 64
DEFAULT_GRIDSIZE = 1
DEFAULT_PROBE_NPZ = Path("datasets/Run1084_recon3_postPC_shrunk_3.npz")


def _phase_metric(metrics: Dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (list, tuple)) and len(value) > 1:
        value = value[1]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_dataset(output_dir: Path, epochs: int, nimgs_train: int, nimgs_test: int) -> Tuple[Path, Path]:
    dataset_dir = output_dir / "datasets" / f"N{DEFAULT_N}" / f"gs{DEFAULT_GRIDSIZE}"
    train_npz = dataset_dir / "train.npz"
    test_npz = dataset_dir / "test.npz"

    if train_npz.exists() and test_npz.exists():
        return train_npz, test_npz

    cfg = GridLinesConfig(
        N=DEFAULT_N,
        gridsize=DEFAULT_GRIDSIZE,
        output_dir=output_dir,
        probe_npz=DEFAULT_PROBE_NPZ,
        nimgs_train=nimgs_train,
        nimgs_test=nimgs_test,
        nepochs=epochs,
    )
    run_grid_lines_workflow(cfg)
    return train_npz, test_npz


def _grid(light: bool) -> Iterable[Tuple[str, str, int, int]]:
    architectures = ["fno"] if light else ["hybrid", "fno"]
    transforms = ["none"] if light else ["none", "log1p", "sqrt"]
    modes = [12] if light else [12, 16, 24]
    widths = [32] if light else [32, 48, 64]

    for arch in architectures:
        for transform in transforms:
            for mode in modes:
                for width in widths:
                    yield arch, transform, mode, width


def run_sweep(
    *,
    output_dir: Path,
    epochs: int = 20,
    light: bool = False,
    ensure_data: bool = True,
    nimgs_train: int = 1,
    nimgs_test: int = 1,
) -> Path:
    os.environ.setdefault("PTYCHO_DISABLE_MEMOIZE", "1")
    os.environ.setdefault("PTYCHO_MEMOIZE_KEY_MODE", "dataset")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = output_dir / "datasets" / f"N{DEFAULT_N}" / f"gs{DEFAULT_GRIDSIZE}"
    train_npz = dataset_dir / "train.npz"
    test_npz = dataset_dir / "test.npz"

    if ensure_data:
        train_npz, test_npz = _ensure_dataset(output_dir, epochs, nimgs_train, nimgs_test)

    results: List[Dict[str, Any]] = []

    for idx, (arch, transform, mode, width) in enumerate(_grid(light), start=1):
        run_root = output_dir / f"run_{idx:03d}_{arch}_m{mode}_w{width}_t{transform}"
        run_root.mkdir(parents=True, exist_ok=True)

        row: Dict[str, Any] = {
            "architecture": arch,
            "fno_input_transform": transform,
            "fno_modes": mode,
            "fno_width": width,
            "epochs": epochs,
            "model_params": None,
            "inference_time_s": None,
            "ssim_phase": None,
            "psnr_phase": None,
            "mae_phase": None,
            "error": None,
        }

        cfg = TorchRunnerConfig(
            train_npz=train_npz,
            test_npz=test_npz,
            output_dir=run_root,
            architecture=arch,
            epochs=epochs,
            fno_modes=mode,
            fno_width=width,
            fno_input_transform=transform,
        )

        try:
            result = run_grid_lines_torch(cfg)
            metrics = result.get("metrics", {})
            row.update({
                "model_params": result.get("model_params"),
                "inference_time_s": result.get("inference_time_s"),
                "ssim_phase": _phase_metric(metrics, "ssim"),
                "psnr_phase": _phase_metric(metrics, "psnr"),
                "mae_phase": _phase_metric(metrics, "mae"),
            })
        except Exception as exc:
            row["error"] = str(exc)

        results.append(row)

    csv_path = output_dir / "study_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)

    if results:
        write_pareto_plot(results, output_dir)

    return csv_path


def write_pareto_plot(results: List[Dict[str, Any]], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for row in results:
        x = row.get("model_params")
        y = row.get("ssim_phase")
        if x is None or y is None:
            continue
        try:
            x_val = float(x)
            y_val = float(y)
        except (TypeError, ValueError):
            continue
        label = row.get("fno_input_transform", "unknown")
        grouped.setdefault(str(label), []).append((x_val, y_val))

    for label, points in grouped.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Phase SSIM")
    if grouped:
        ax.legend(loc="best", fontsize="small")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    out_path = output_dir / "pareto_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run FNO/Hybrid hyperparameter sweep")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--light", action="store_true", help="Run a light sweep for validation")
    parser.add_argument("--nimgs-train", type=int, default=1)
    parser.add_argument("--nimgs-test", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_sweep(
        output_dir=args.output_dir,
        epochs=args.epochs,
        light=args.light,
        ensure_data=True,
        nimgs_train=args.nimgs_train,
        nimgs_test=args.nimgs_test,
    )


if __name__ == "__main__":
    main()
