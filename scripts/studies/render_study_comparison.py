#!/usr/bin/env python
"""Render a study comparison table from per-arm reconstruction data.

One row per arm, six columns: Amp truth | Amp recon (SSIM) | |Amp error| |
Phase truth | Phase recon (SSIM) | Wrapped phase error. Panels are rendered
from each arm's ``reconstruction.npz`` — never pasted bitmaps. Color limits
are shared per column across all rows of the same object family (matching
the reference table's convention), phase panels use a cyclic colormap
over the plane-aligned data range, and the phase-error range is symmetric
about zero at the observed error scale. Contract: stage 14 of
``docs/superpowers/plans/2026-08-07-synthetic-legacy-reproduction-repair.md``.

Usage:
    python scripts/studies/render_study_comparison.py <study_root> \
        [--output <png>] [--title <text>] [--phase-variance-floor 1e-6]

Arms are discovered at ``<study_root>/<arm>/`` (hydra study layout) or
``<study_root>/matrix_results/<arm>/`` (legacy harness layout); each arm dir
must contain ``reconstruction/reconstruction.npz``,
``reconstruction/metrics.json`` (SSIMs + alignment record), and
``datasets/test.npz`` (ground-truth ``objectGuess``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

COLUMNS = 6
_FAMILY_ORDER = {"lines": 0, "deadleaves": 1}
_ARCH_ORDER = {"hybrid_resnet": 0, "cnn": 1}
_PROFILE_ORDER = {"legacy_mae": 0, "legacy_nll": 1, "ci_nll": 2}
_FAMILY_LABELS = {"lines": "Lines", "deadleaves": "Dead leaves"}
_ARCH_LABELS = {"hybrid_resnet": "Hybrid ResNet", "cnn": "CNN"}
_PROFILE_LABELS = {
    "legacy_mae": "Legacy MAE",
    "legacy_nll": "Legacy NLL",
    "ci_nll": "CI NLL",
}


def remove_plane(phase: np.ndarray) -> np.ndarray:
    """Least-squares plane removal; matches ptycho.evaluation.fit_and_remove_plane."""
    height, width = phase.shape
    grid_y, grid_x = np.meshgrid(
        np.arange(height), np.arange(width), indexing="ij"
    )
    basis = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.ones(phase.size)]
    )
    coeffs, _, _, _ = np.linalg.lstsq(basis, phase.ravel(), rcond=None)
    return phase - (coeffs[0] * grid_x + coeffs[1] * grid_y + coeffs[2])


def _sort_key(name: str) -> tuple:
    parts = name.split("__")
    if (
        len(parts) == 3
        and parts[0] in _FAMILY_ORDER
        and parts[1] in _ARCH_ORDER
        and parts[2] in _PROFILE_ORDER
    ):
        return (
            0,
            _FAMILY_ORDER[parts[0]],
            _ARCH_ORDER[parts[1]],
            _PROFILE_ORDER[parts[2]],
            name,
        )
    return (1, 0, 0, 0, name)


def _row_label(name: str) -> str:
    parts = name.split("__")
    if len(parts) == 3:
        return "\n".join(
            (
                _FAMILY_LABELS.get(parts[0], parts[0]),
                _ARCH_LABELS.get(parts[1], parts[1]),
                _PROFILE_LABELS.get(parts[2], parts[2]),
            )
        )
    return name


def discover_arms(study_root: Path) -> list[Path]:
    root = study_root / "matrix_results"
    if not root.is_dir():
        root = study_root
    arm_dirs = [
        candidate.parent.parent
        for candidate in root.glob("*/reconstruction/reconstruction.npz")
    ]
    return sorted(arm_dirs, key=lambda path: _sort_key(path.name))


def load_row(arm_dir: Path) -> dict[str, Any]:
    reconstruction_dir = arm_dir / "reconstruction"
    metrics = json.loads((reconstruction_dir / "metrics.json").read_text())
    alignment = metrics["alignment"]
    origin_row, origin_col = (int(v) for v in alignment["truth_origin"])
    height, width = (int(v) for v in alignment["aligned_shape"])
    crop = int(alignment.get("metric_crop_border", 0))

    with np.load(
        reconstruction_dir / "reconstruction.npz", allow_pickle=False
    ) as archive:
        prediction = np.asarray(archive["complex_canvas"])
    with np.load(arm_dir / "datasets" / "test.npz", allow_pickle=False) as archive:
        truth = np.asarray(archive["objectGuess"])
    truth = truth[origin_row : origin_row + height, origin_col : origin_col + width]
    if truth.shape != prediction.shape:
        raise ValueError(
            f"{arm_dir.name}: truth slice {truth.shape} does not match "
            f"reconstruction canvas {prediction.shape}"
        )
    if crop:
        truth = truth[crop:-crop, crop:-crop]
        prediction = prediction[crop:-crop, crop:-crop]

    amp_truth = np.abs(truth)
    amp_prediction = np.abs(prediction)
    amp_prediction = amp_prediction * (
        np.mean(amp_truth) / np.mean(amp_prediction)
    )
    phase_truth = remove_plane(np.angle(truth))
    phase_prediction = remove_plane(np.angle(prediction))
    return {
        "arm": arm_dir.name,
        "label": _row_label(arm_dir.name),
        "amplitude_ssim": float(metrics["amplitude_ssim"]),
        "phase_ssim": float(metrics["phase_ssim"]),
        "arrays": {
            "amp_truth": amp_truth,
            "amp_prediction": amp_prediction,
            "amp_error": np.abs(amp_truth - amp_prediction),
            "phase_truth": phase_truth,
            "phase_prediction": phase_prediction,
            "phase_error": np.angle(
                np.exp(1j * (phase_prediction - phase_truth))
            ),
        },
    }


def robust_bounds(
    values: list[np.ndarray], *, symmetric: bool = False
) -> tuple[float, float]:
    flat = np.concatenate([np.asarray(value).ravel() for value in values])
    if symmetric:
        bound = max(float(np.percentile(np.abs(flat), 99.5)), 1e-12)
        return -bound, bound
    low, high = np.percentile(flat, [0.5, 99.5])
    if high <= low:
        high = low + 1e-12
    return float(low), float(high)


def _group_key(name: str) -> str:
    """Rows sharing a family share color limits (reference-figure convention)."""
    parts = name.split("__")
    return parts[0] if len(parts) == 3 else "all"


def column_scales(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Shared limit pair per column kind, per family group across its rows."""
    scales: dict[str, dict[str, tuple[float, float]]] = {}
    for group in {_group_key(row["arm"]) for row in rows}:
        arrays = [
            row["arrays"] for row in rows if _group_key(row["arm"]) == group
        ]
        scales[group] = {
            "amp": robust_bounds(
                [a["amp_truth"] for a in arrays]
                + [a["amp_prediction"] for a in arrays]
            ),
            "amp_error": robust_bounds([a["amp_error"] for a in arrays]),
            "phase": robust_bounds(
                [a["phase_truth"] for a in arrays]
                + [a["phase_prediction"] for a in arrays]
            ),
            "phase_error": robust_bounds(
                [a["phase_error"] for a in arrays], symmetric=True
            ),
        }
    return scales


def render(
    rows: list[dict[str, Any]],
    scales: dict[str, dict[str, tuple[float, float]]],
    output: Path,
    title: str,
    subtitle: str,
) -> None:
    figure, axes = plt.subplots(
        len(rows),
        COLUMNS,
        figsize=(19, 2.85 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    assert axes.shape == (len(rows), COLUMNS)
    for index, row in enumerate(rows):
        arrays = row["arrays"]
        group = scales[_group_key(row["arm"])]
        panels = (
            (arrays["amp_truth"], "Amp truth", "viridis", group["amp"]),
            (
                arrays["amp_prediction"],
                f"Amp recon\nSSIM {row['amplitude_ssim']:.4f}",
                "viridis",
                group["amp"],
            ),
            (arrays["amp_error"], "|Amp error|", "magma", group["amp_error"]),
            (arrays["phase_truth"], "Phase truth", "twilight", group["phase"]),
            (
                arrays["phase_prediction"],
                f"Phase recon\nSSIM {row['phase_ssim']:.4f}",
                "twilight",
                group["phase"],
            ),
            (
                arrays["phase_error"],
                "Wrapped phase error",
                "RdBu_r",
                group["phase_error"],
            ),
        )
        for column, (image, panel_title, cmap, bounds) in enumerate(panels):
            axis = axes[index, column]
            axis.imshow(image, cmap=cmap, vmin=bounds[0], vmax=bounds[1])
            axis.set_title(panel_title, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[index, 0].set_ylabel(
            row["label"],
            rotation=0,
            ha="right",
            va="center",
            labelpad=38,
            fontsize=8,
        )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default="Object-family comparison")
    parser.add_argument(
        "--subtitle",
        default="Mean-scaled amplitude, plane-aligned phase",
    )
    parser.add_argument(
        "--phase-variance-floor",
        type=float,
        default=1e-6,
        help="Minimum pixel variance across phase panels; guards against the "
        "black-phase regression",
    )
    args = parser.parse_args(argv)

    arm_dirs = discover_arms(args.study_root)
    if not arm_dirs:
        print(
            f"no arms with reconstruction data under {args.study_root}",
            file=sys.stderr,
        )
        return 1
    rows = [load_row(arm_dir) for arm_dir in arm_dirs]

    phase_pixels = np.concatenate(
        [
            row["arrays"][key].ravel()
            for row in rows
            for key in ("phase_truth", "phase_prediction")
        ]
    )
    phase_variance = float(np.var(phase_pixels))
    if phase_variance < args.phase_variance_floor:
        print(
            f"phase variance {phase_variance:.3g} is below the floor "
            f"{args.phase_variance_floor:g}: phase panels carry no structure "
            "(black-phase regression)",
            file=sys.stderr,
        )
        return 1

    scales = column_scales(rows)
    output = args.output or (args.study_root / "comparison_table.png")
    render(rows, scales, output, args.title, args.subtitle)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "columns": COLUMNS,
                "arms": [row["arm"] for row in rows],
                "phase_variance": phase_variance,
                "column_limits": {
                    group: {
                        name: [float(low), float(high)]
                        for name, (low, high) in kinds.items()
                    }
                    for group, kinds in scales.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
