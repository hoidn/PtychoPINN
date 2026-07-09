"""Montage renderer for the aligned-ablation inference-variant grid (Task 5).

Ports the session-scratch renderer (``scratchpad/montage/render_montage.py``,
lost to a machine reboot) into the repo so the montage is reproducible.
Reconstructed from ``docs/plans/2026-07-06-aligned-ablation-20epoch-rerun.md``
Task 5 and the 5-epoch reference artifacts under
``.artifacts/varpro_ablation/ext_matrix_aligned/montage/``.

Consumes the output contract written by
``scripts/studies/aligned_ablation_variant_grid.py``: for each arm under
``<root>``, ``variants/<variant>/{canvas.npz,metrics.json}``. Renders two
PNGs -- ``montage_amp.png`` and ``montage_phase.png`` -- with rows = arms,
columns = a leftmost ground-truth column followed by the four inference
variants (``uniform_novarpro``, ``uniform_varpro``, ``probe_novarpro``,
``probe_varpro``).

Shared color scale ("error scale from healthy arms only, collapsed arms
clip"): a training run that has collapsed to a near-uniform canvas would,
if included naively, either wash out the shared scale to its own
degenerate range or (if it saturated to an extreme value) blow the range
out for every other cell. ``compute_healthy_scale`` excludes any arm whose
pixel std falls below a small floor (``floor_ratio`` times the median std
across arms) from the vmin/vmax computation; excluded arms are still
rendered with the same shared scale, so they clip/saturate visually
instead of dictating it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# Mirrors aligned_ablation_variant_grid.py's own sys.path handling: this
# script is invoked as `python scripts/studies/aligned_ablation_montage.py`,
# so the repo root is not implicitly on sys.path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.aligned_ablation_variant_grid import (
    BORDER_TRIM_PX,
    crop_border,
    load_truth,
)

VARIANTS: Tuple[str, ...] = (
    "uniform_novarpro",
    "uniform_varpro",
    "probe_novarpro",
    "probe_varpro",
)


# ---------------------------------------------------------------------------
# Pure-numpy pieces (no matplotlib import -- CPU-cheap, unit tested directly
# by tests/studies/test_aligned_ablation_montage.py).
# ---------------------------------------------------------------------------

def compute_healthy_scale(
    arm_arrays: Dict[str, np.ndarray],
    reference: Optional[np.ndarray] = None,
    floor_ratio: float = 0.01,
) -> Tuple[float, float]:
    """Shared ``(vmin, vmax)`` display range computed from healthy arms only.

    An arm is "healthy" when its pixel std exceeds ``floor_ratio`` times the
    median std across all arms; a collapsed/flat arm (std ~ 0, or otherwise
    far below the pack) falls under that floor and is excluded from the
    range computation, so it cannot drag the shared scale down to its own
    degenerate values (or blow it out, if it rails to an extreme constant).
    Excluded arms are still rendered with the resulting scale -- they clip
    visually rather than dictating it.

    ``reference`` (typically the ground-truth column) is always folded into
    the pooled range regardless of the health check, since ground truth is
    never "collapsed" by construction.

    Falls back to every arm if none clears the floor (all flat) -- some
    scale must still be picked.
    """
    stds = {arm: float(np.std(arr)) for arm, arr in arm_arrays.items()}
    median_std = float(np.median(list(stds.values()))) if stds else 0.0
    floor = floor_ratio * median_std
    healthy = [arm for arm, std in stds.items() if std > floor] or list(arm_arrays.keys())

    pooled = [arm_arrays[arm].ravel() for arm in healthy]
    if reference is not None:
        pooled.append(np.asarray(reference).ravel())
    values = np.concatenate(pooled)
    return float(values.min()), float(values.max())


def pool_by_arm(
    canvases: Dict[Tuple[str, str], np.ndarray],
    arms: Sequence[str],
    variants: Sequence[str],
    transform: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Concatenate ``transform(canvas)`` across an arm's variants, per arm --
    the per-arm pixel pool ``compute_healthy_scale`` computes std over."""
    return {
        arm: np.concatenate([transform(canvases[(arm, variant)]).ravel() for variant in variants])
        for arm in arms
    }


# ---------------------------------------------------------------------------
# I/O + orchestration
# ---------------------------------------------------------------------------

def load_arm_variant(root: Path, arm: str, variant: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load one arm/variant's ``canvas.npz`` + ``metrics.json``, written by
    ``aligned_ablation_variant_grid.run_arm``."""
    variant_dir = root / arm / "variants" / variant
    canvas_path = variant_dir / "canvas.npz"
    metrics_path = variant_dir / "metrics.json"
    if not canvas_path.exists():
        raise FileNotFoundError(f"Missing {canvas_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")
    with np.load(canvas_path) as data:
        canvas = data["canvas"]
    metrics = json.loads(metrics_path.read_text())
    return canvas, metrics


def load_grid(
    root: Path, arms: Sequence[str], variants: Sequence[str] = VARIANTS,
) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], Dict[str, Any]]]:
    """Load the border-cropped ground truth plus every arm/variant canvas +
    metrics under ``root``.

    Ground truth is ``<root>/data/test.npz``'s ``YY_ground_truth`` array
    (``aligned_ablation_variant_grid.load_truth``; there is no literal
    ``objectGuess`` key in this pipeline's test.npz schema -- that name
    refers generically to the reconstructed object's ground truth), cropped
    by the same ``BORDER_TRIM_PX`` the grid script applies to canvases
    before scoring.
    """
    test_npz_path = root / "data" / "test.npz"
    if not test_npz_path.exists():
        raise FileNotFoundError(f"Missing {test_npz_path}")
    with np.load(test_npz_path) as data:
        truth = load_truth(data)
    truth_cropped = crop_border(truth, BORDER_TRIM_PX)

    canvases: Dict[Tuple[str, str], np.ndarray] = {}
    metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for arm in arms:
        for variant in variants:
            canvas, variant_metrics = load_arm_variant(root, arm, variant)
            canvases[(arm, variant)] = canvas
            metrics[(arm, variant)] = variant_metrics
    return truth_cropped, canvases, metrics


def render_one_montage(
    output_path: Path,
    truth_display: np.ndarray,
    canvases: Dict[Tuple[str, str], np.ndarray],
    metrics: Dict[Tuple[str, str], Dict[str, Any]],
    arms: Sequence[str],
    variants: Sequence[str],
    transform: Callable[[np.ndarray], np.ndarray],
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Render one truth + arms x variants grid figure to ``output_path``.

    Cell titles are ``arm/variant`` plus that variant's ``amp_mae`` (from
    its ``metrics.json``), per the montage spec -- both the amplitude and
    phase montages use ``amp_mae`` as the headline number.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(arms)
    n_cols = len(variants) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)
    for row, arm in enumerate(arms):
        ax_truth = axes[row][0]
        ax_truth.imshow(truth_display, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_truth.set_title("truth", fontsize=8)
        ax_truth.axis("off")
        for col, variant in enumerate(variants, start=1):
            ax = axes[row][col]
            ax.imshow(transform(canvases[(arm, variant)]), cmap=cmap, vmin=vmin, vmax=vmax)
            amp_mae = metrics[(arm, variant)]["amp_mae"]
            ax.set_title(f"{arm}/{variant}\namp_mae={amp_mae:.4f}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_montages(
    root: Path,
    arms: Sequence[str],
    output_dir: Path,
    variants: Sequence[str] = VARIANTS,
    floor_ratio: float = 0.01,
) -> Tuple[Path, Path]:
    """Render ``montage_amp.png`` + ``montage_phase.png`` for ``root``'s
    variant grid into ``output_dir``. Returns the two output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_cropped, canvases, metrics = load_grid(root, arms, variants)

    amp_pool = pool_by_arm(canvases, arms, variants, np.abs)
    amp_vmin, amp_vmax = compute_healthy_scale(
        amp_pool, reference=np.abs(truth_cropped), floor_ratio=floor_ratio,
    )
    amp_path = output_dir / "montage_amp.png"
    render_one_montage(
        amp_path, np.abs(truth_cropped), canvases, metrics, arms, variants,
        np.abs, "gray", amp_vmin, amp_vmax,
    )

    phase_pool = pool_by_arm(canvases, arms, variants, np.angle)
    phase_vmin, phase_vmax = compute_healthy_scale(
        phase_pool, reference=np.angle(truth_cropped), floor_ratio=floor_ratio,
    )
    phase_path = output_dir / "montage_phase.png"
    render_one_montage(
        phase_path, np.angle(truth_cropped), canvases, metrics, arms, variants,
        np.angle, "twilight", phase_vmin, phase_vmax,
    )
    return amp_path, phase_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, type=Path,
        help="Variant-grid root (output of aligned_ablation_variant_grid.py)",
    )
    parser.add_argument(
        "--arms", required=True, nargs="+",
        help="Arm subdirectory names under --root, in row order",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write montage_amp.png/montage_phase.png (default: <root>/montage)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir or (args.root / "montage")
    amp_path, phase_path = render_montages(args.root, args.arms, output_dir)
    print(f"Wrote {amp_path}")
    print(f"Wrote {phase_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
