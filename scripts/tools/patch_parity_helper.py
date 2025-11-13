#!/usr/bin/env python3
"""
Generate amplitude/phase patch comparison grids for TensorFlow vs PyTorch runs.

This helper expects two NPZ files containing reconstructed patches captured at the
same epoch. Each NPZ must provide:
    - 'amp': (N, H, W) float array of reconstructed amplitudes
    - 'phase': (N, H, W) float array of reconstructed phases
    - Optional 'sample_indices': (N,) int array of global sample ids

Example:
    python scripts/tools/patch_parity_helper.py \
        --tf-npz tmp/tf_epoch50_patches.npz \
        --torch-npz tmp/torch_epoch50_patches.npz \
        --epoch 50 \
        --num-patches 6

Outputs are written to tmp/patch_parity/{backend}_epoch{epoch}.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_backend_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    if 'amp' not in data or 'phase' not in data:
        raise ValueError(f"{path} missing required 'amp' or 'phase' arrays")
    amp = np.asarray(data['amp'])
    phase = np.asarray(data['phase'])
    if amp.shape != phase.shape:
        raise ValueError(f"{path} amplitude/phase shapes mismatch: {amp.shape} vs {phase.shape}")
    sample_indices = (
        np.asarray(data['sample_indices'])
        if 'sample_indices' in data
        else np.arange(amp.shape[0])
    )
    return amp, phase, sample_indices


def _select_shared_indices(
    tf_indices: np.ndarray,
    torch_indices: np.ndarray,
    limit: int,
) -> np.ndarray:
    shared = np.intersect1d(tf_indices, torch_indices)
    if shared.size == 0:
        raise ValueError(
            "No overlapping sample indices found between TensorFlow and PyTorch patch sets."
            " Ensure both runs consumed the same subsample_seed or provide aligned NPZ files."
        )
    return shared[:limit]


def _plot_backend_grid(
    backend_name: str,
    amp: np.ndarray,
    phase: np.ndarray,
    positions: np.ndarray,
    sample_ids: np.ndarray,
    epoch: int,
    output_dir: Path,
):
    ncols = positions.shape[0]
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 6), constrained_layout=True)
    for col, (pos, sample_id) in enumerate(zip(positions, sample_ids)):
        amp_ax = axes[0, col] if ncols > 1 else axes[0]
        phase_ax = axes[1, col] if ncols > 1 else axes[1]
        amp_ax.imshow(amp[pos], cmap='magma')
        amp_ax.set_title(f"{backend_name} amp\nidx {sample_id}")
        amp_ax.axis('off')
        phase_ax.imshow(phase[pos], cmap='twilight', vmin=-np.pi, vmax=np.pi)
        phase_ax.set_title(f"{backend_name} phase\nidx {sample_id}")
        phase_ax.axis('off')
    fig.suptitle(f"{backend_name} patch parity — epoch {epoch}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{backend_name.lower()}_epoch{epoch}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Patch parity grid generator")
    parser.add_argument("--tf-npz", type=Path, required=True, help="NPZ file with TF patches")
    parser.add_argument("--torch-npz", type=Path, required=True, help="NPZ file with Torch patches")
    parser.add_argument("--epoch", type=int, default=50, help="Epoch label for output naming")
    parser.add_argument(
        "--num-patches",
        type=int,
        default=4,
        help="Number of shared sample ids to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/patch_parity"),
        help="Directory for generated comparison grids",
    )
    args = parser.parse_args()

    tf_amp, tf_phase, tf_indices = _load_backend_npz(args.tf_npz)
    tor_amp, tor_phase, tor_indices = _load_backend_npz(args.torch_npz)

    shared_ids = _select_shared_indices(tf_indices, tor_indices, args.num_patches)

    tf_index_map = {int(idx): pos for pos, idx in enumerate(tf_indices)}
    tor_index_map = {int(idx): pos for pos, idx in enumerate(tor_indices)}
    tf_positions = np.array([tf_index_map[int(idx)] for idx in shared_ids])
    tor_positions = np.array([tor_index_map[int(idx)] for idx in shared_ids])

    tf_path = _plot_backend_grid(
        "TensorFlow", tf_amp, tf_phase, tf_positions, shared_ids, args.epoch, args.output_dir
    )
    torch_path = _plot_backend_grid(
        "PyTorch", tor_amp, tor_phase, tor_positions, shared_ids, args.epoch, args.output_dir
    )
    print(f"Wrote TF parity grid → {tf_path}")
    print(f"Wrote Torch parity grid → {torch_path}")


if __name__ == "__main__":
    main()
