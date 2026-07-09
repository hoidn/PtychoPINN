"""Matplotlib figure helpers for the VarPro/probe ablation harness (Task 1.5, F2).

Pure functions: arrays + paths only, no config objects, no torch. Extracted
out of ``varpro_probe_ablation_runner.py`` to keep that module under the
project's 500-line-per-module limit.

Two families:
  - Per-variant panels (moved here unchanged from the runner):
    ``save_recon_panel``, ``save_error_panel``.
  - Per-arm combined grids (new, F2): ``save_reconstruction_grid``,
    ``save_error_grid`` -- ONE figure per arm with truth + ALL inference
    variants side by side (amplitude row + phase row for the reconstruction
    grid; ``|recon - truth|`` per variant for the error grid), so a reviewer
    can compare every variant against ground truth and each other at a
    glance instead of opening N separate 2x2 panels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def save_recon_panel(path: Path, recon: np.ndarray, truth: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(np.abs(truth), cmap="gray")
    axes[0, 0].set_title("truth |amp|")
    axes[0, 1].imshow(np.angle(truth), cmap="twilight")
    axes[0, 1].set_title("truth phase")
    axes[1, 0].imshow(np.abs(recon), cmap="gray")
    axes[1, 0].set_title("recon |amp| (phase-aligned)")
    axes[1, 1].imshow(np.angle(recon), cmap="twilight")
    axes[1, 1].set_title("recon phase (phase-aligned)")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_error_panel(path: Path, recon_aligned: np.ndarray, truth_crop: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = np.abs(recon_aligned - truth_crop)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(err, cmap="inferno")
    ax.set_title("|recon - truth| (phase-aligned)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_reconstruction_grid(
    path: Path, truth_crop: np.ndarray, aligned_variants: Dict[str, np.ndarray],
) -> None:
    """One figure per arm: truth + every phase-aligned variant canvas, side by
    side -- amplitude row on top, phase row on bottom.

    ``aligned_variants`` maps variant name -> phase-aligned canvas, already
    cropped to the same overlap region as ``truth_crop`` (the same convention
    ``save_recon_panel`` uses per-variant -- callers typically reuse the
    ``aligned``/``truth_crop`` pair returned by ``align_global_phase`` for
    each variant against the same truth array).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["truth"] + list(aligned_variants.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    for col, name in enumerate(names):
        arr = truth_crop if name == "truth" else aligned_variants[name]
        axes[0, col].imshow(np.abs(arr), cmap="gray")
        axes[0, col].set_title(f"{name}\n|amp|")
        axes[1, col].imshow(np.angle(arr), cmap="twilight")
        axes[1, col].set_title(f"{name}\nphase")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_error_grid(
    path: Path, truth_crop: np.ndarray, aligned_variants: Dict[str, np.ndarray],
) -> None:
    """One figure per arm: ``|variant - truth|`` side by side for every variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(aligned_variants.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for col, name in enumerate(names):
        err = np.abs(aligned_variants[name] - truth_crop)
        im = axes[0, col].imshow(err, cmap="inferno")
        axes[0, col].set_title(f"{name}\n|recon - truth|")
        axes[0, col].axis("off")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
