"""CPU-cheap tests for the aligned-ablation montage renderer (Task 5): a
synthetic variant-grid root fixture exercises the full render path (no
mocks, no GPU, no real artifacts), and the healthy-arms-only shared-scale
logic is checked directly against a constructed flat-arm case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.studies.aligned_ablation_montage import (
    VARIANTS,
    compute_healthy_scale,
    render_montages,
)


def _write_variant(root: Path, arm: str, variant: str, canvas: np.ndarray, amp_mae: float) -> None:
    variant_dir = root / arm / "variants" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.savez(variant_dir / "canvas.npz", canvas=canvas.astype(np.complex64))
    (variant_dir / "metrics.json").write_text(json.dumps({"amp_mae": amp_mae}))


def _build_grid_root(tmp_path: Path, arms_and_canvases: dict) -> Path:
    """Write a synthetic variant-grid root: <root>/data/test.npz (ground
    truth) plus <root>/<arm>/variants/<variant>/{canvas.npz,metrics.json}
    for every arm, matching aligned_ablation_variant_grid.py's output
    contract."""
    root = tmp_path / "root"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)

    truth = np.ones((50, 50), dtype=np.complex64) * (0.5 + 0.1j)
    np.savez(data_dir / "test.npz", YY_ground_truth=truth[:, :, None])

    for arm, canvas in arms_and_canvases.items():
        for variant in VARIANTS:
            _write_variant(root, arm, variant, canvas, amp_mae=0.1234)
    return root


def test_render_montages_writes_nonempty_pngs(tmp_path: Path):
    healthy_canvas = (np.linspace(0.1, 0.9, 16 * 16).reshape(16, 16) * np.exp(1j * 0.3)).astype(np.complex64)
    arms_and_canvases = {
        "neither": healthy_canvas,
        "weight_only": healthy_canvas * 1.1,
    }
    root = _build_grid_root(tmp_path, arms_and_canvases)
    output_dir = root / "montage"

    amp_path, phase_path = render_montages(root, list(arms_and_canvases), output_dir)

    assert amp_path == output_dir / "montage_amp.png"
    assert phase_path == output_dir / "montage_phase.png"
    assert amp_path.exists() and amp_path.stat().st_size > 0
    assert phase_path.exists() and phase_path.stat().st_size > 0


def test_compute_healthy_scale_excludes_flat_arm():
    # Two "healthy" arms with genuine spread, one "flat" (collapsed) arm
    # railed to a constant far outside the healthy range. A naive scale
    # (all arms pooled) would blow vmax out to 9.0; the healthy-only scale
    # must ignore the flat arm's std-below-floor pixels.
    arm_arrays = {
        "healthy_a": np.array([0.10, 0.20, 0.30, 0.40]),
        "healthy_b": np.array([0.15, 0.25, 0.35, 0.45]),
        "flat": np.full(4, 9.0),
    }

    vmin, vmax = compute_healthy_scale(arm_arrays, floor_ratio=0.01)

    assert vmin == 0.10
    assert vmax == 0.45


def test_compute_healthy_scale_falls_back_to_all_arms_when_every_arm_flat():
    arm_arrays = {
        "flat_a": np.full(4, 2.0),
        "flat_b": np.full(4, 2.0),
    }

    vmin, vmax = compute_healthy_scale(arm_arrays, floor_ratio=0.01)

    assert vmin == 2.0
    assert vmax == 2.0
