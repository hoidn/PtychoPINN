"""Visual helpers for the BRDT four-row preflight bundle.

Renders fixed-sample comparison panels, error maps, and sinogram-residual
panels. Source ``.npy`` arrays are written alongside the rendered PNGs so
the follow-up summary item can regenerate plots without touching this
module.

This module deliberately avoids any heavy plotting dependency beyond
Matplotlib so the bundle stays portable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


VISUAL_MANIFEST_VERSION: str = "brdt_preflight_visuals_v1"


@dataclass
class VisualBundleEntry:
    """One sample/row visual artifact pointer."""

    sample_id: int
    row_id: str
    pred_array: str
    target_array: str
    sino_pred_array: str
    sino_obs_array: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "sample_id": int(self.sample_id),
            "row_id": self.row_id,
            "pred_array": self.pred_array,
            "target_array": self.target_array,
            "sino_pred_array": self.sino_pred_array,
            "sino_obs_array": self.sino_obs_array,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


def save_source_array(path: Path, arr: np.ndarray) -> str:
    """Write a numpy array as ``.npy`` and return the relative path string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr))
    return path.name


def render_compare_q(
    *,
    preds_by_row: Mapping[str, np.ndarray],
    target: np.ndarray,
    out_path: Path,
    sample_id: int,
) -> None:
    """Render a ``q`` comparison panel: target alongside each row prediction.

    ``preds_by_row[row_id]`` is a single-sample 2-D array shaped ``(N, N)``.
    The figure layout is one row per visualization with the ground truth
    in the leftmost column followed by each row's prediction.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    n_rows = max(1, len(preds_by_row))
    fig, axes = plt.subplots(1, n_rows + 1, figsize=(3 * (n_rows + 1), 3.2))
    if n_rows + 1 == 1:
        axes = [axes]
    target_2d = np.asarray(target).squeeze()
    vmin = float(target_2d.min())
    vmax = float(target_2d.max())
    axes[0].imshow(target_2d, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"target_q (sample={sample_id})")
    axes[0].axis("off")
    for ax, (row_id, pred) in zip(axes[1:], preds_by_row.items()):
        pred_2d = np.asarray(pred).squeeze()
        ax.imshow(pred_2d, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(row_id)
        ax.axis("off")
    fig.suptitle("BRDT preflight — physical q (decision_support_preflight_only)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_error_q(
    *,
    preds_by_row: Mapping[str, np.ndarray],
    target: np.ndarray,
    out_path: Path,
    sample_id: int,
) -> None:
    """Render the absolute-error map for each row prediction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    n_rows = max(1, len(preds_by_row))
    fig, axes = plt.subplots(1, n_rows, figsize=(3 * n_rows, 3.2))
    if n_rows == 1:
        axes = [axes]
    target_2d = np.asarray(target).squeeze()
    diffs = {row_id: np.abs(np.asarray(p).squeeze() - target_2d) for row_id, p in preds_by_row.items()}
    vmax = max((float(d.max()) for d in diffs.values()), default=0.0)
    for ax, (row_id, d) in zip(axes, diffs.items()):
        ax.imshow(d, vmin=0.0, vmax=vmax, cmap="magma")
        ax.set_title(f"|err| {row_id}")
        ax.axis("off")
    fig.suptitle(
        f"BRDT preflight — |q_pred - q_target| (sample={sample_id})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_sinogram_residual(
    *,
    sino_obs: np.ndarray,
    sino_preds_by_row: Mapping[str, np.ndarray],
    out_path: Path,
    sample_id: int,
) -> None:
    """Render the (real, imag) sinogram residuals for each row prediction.

    ``sino_obs`` shape ``(A, D, 2)``; ``sino_preds_by_row[row_id]`` shape
    ``(A, D, 2)``. The figure stacks the real and imag parts of the
    residual side-by-side per row.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    rows = list(sino_preds_by_row.items())
    n = max(1, len(rows))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.0), squeeze=False)
    obs = np.asarray(sino_obs)
    for col, (row_id, pred) in enumerate(rows):
        residual = np.asarray(pred) - obs
        for r in range(2):
            axes[r, col].imshow(residual[..., r], cmap="seismic")
            axes[r, col].set_title(f"{row_id} {'real' if r == 0 else 'imag'}")
            axes[r, col].axis("off")
    fig.suptitle(
        f"BRDT preflight — sinogram residual y_pred - y_obs (sample={sample_id})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_visual_manifest(
    path: Path,
    *,
    figures: List[str],
    entries: List[VisualBundleEntry],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": VISUAL_MANIFEST_VERSION,
        "claim_boundary": "decision_support_preflight_only",
        "figures": list(figures),
        "entries": [e.to_dict() for e in entries],
    }
    if extra:
        payload["extra"] = dict(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
