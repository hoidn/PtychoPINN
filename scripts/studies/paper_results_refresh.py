"""Generate paper-local result tables from completed NeurIPS artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.paper_model_config_table import write_model_config_table
from scripts.studies.paper_efficiency_table import write_paper_efficiency_table


NEURIPS_DIR = REPO_ROOT / "docs" / "plans" / "NEURIPS-HYBRID-RESNET-2026"
TABLES_DIR = NEURIPS_DIR / "tables"
FIGURES_DIR = NEURIPS_DIR / "figures"

BRDT_ROOT = (
    REPO_ROOT
    / ".artifacts"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-29-brdt-four-row-preflight"
)
BRDT_METRICS_JSON = BRDT_ROOT / "metrics.json"
BRDT_SOURCE_FIGURE = BRDT_ROOT / "visuals" / "brdt_compare_q.png"
BRDT_40EP_ROOT = (
    REPO_ROOT
    / ".artifacts"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-05-05-brdt-supervised-born-40ep-paper-evidence"
)
BRDT_40EP_SOURCE_ARRAYS = BRDT_40EP_ROOT / "figures" / "source_arrays"
BRDT_CONTEXT_FIGURE = FIGURES_DIR / "brdt_sample_0255_context_recon_error.png"
BRDT_MEASUREMENT_CMAP = "cividis"
BRDT_RECONSTRUCTION_CMAP = "viridis"
BRDT_ERROR_CMAP = "inferno"

CDI_UNO_METRICS_JSON = (
    REPO_ROOT
    / ".artifacts"
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-30-cdi-lines128-uno-table-extension"
    / "runs"
    / "complete_table_plus_uno_20260504T100347Z"
    / "metrics.json"
)
CDI_SUPERVISED_FFNO_METRICS_JSON = (
    REPO_ROOT
    / ".artifacts"
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-29-cdi-lines128-supervised-equivalent-rows"
    / "runs"
    / "supervised_ffno_extension_20260430T180217Z"
    / "runs"
    / "supervised_ffno"
    / "metrics.json"
)
CDI_RECONS_ROOT = (
    REPO_ROOT
    / ".artifacts"
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-30-cdi-lines128-uno-table-extension"
    / "runs"
    / "complete_table_plus_uno_20260504T100347Z"
    / "recons"
)
CDI_PHASE_ZOOM_ROWS = [
    ("gt", "Ground truth"),
    ("pinn", "CNN+PINN"),
    ("pinn_fno_vanilla", "FNO+PINN"),
    ("pinn_ffno", "FFNO-local proxy+PINN"),
    ("pinn_neuralop_uno", "U-NO+PINN"),
    ("pinn_hybrid_resnet", "SRU-Net+PINN"),
]
CDI_PHASE_ZOOM_FIGURE = FIGURES_DIR / "cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet.png"
CDI_PHASE_ZOOM_PER_PANEL_FIGURE = (
    FIGURES_DIR / "cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet_per_panel_scaled.png"
)

REQUIRED_CNS_HISTORY5_ROWS = [
    "author_ffno_cns_base",
    "spectral_resnet_bottleneck_base",
    "fno_base",
    "unet_strong",
]

KNOWN_CNS_HISTORY5_ROWS = {
    "author_ffno_cns_base": {"history_len": 5, "epochs": 40},
    "spectral_resnet_bottleneck_base": {"history_len": 5, "epochs": 40},
    "fno_base": {"history_len": 5, "epochs": 40},
    "unet_strong": {"history_len": 5, "epochs": 40},
}

CNS_REQUIRED_HEADLINE_ROWS = [
    "author_ffno_cns_base",
    "spectral_resnet_bottleneck_base",
    "fno_base",
    "unet_strong",
]

CNS_MANUSCRIPT_LABELS = {
    "author_ffno_cns_base": "FFNO",
    "spectral_resnet_bottleneck_base": "SRU-Net*",
    "fno_base": "FNO",
    "unet_strong": "U-Net",
}

CNS_H2_BUNDLE_TABLE_JSON = (
    REPO_ROOT
    / ".artifacts"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-29-cns-paper-2048cap-row-extension"
    / "bundle_2048cap"
    / "cns_paper_table_rows.json"
)
CNS_H2_SUMMARY_AUTHORITY = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md"
)
CNS_H2_CONTRACT_AUTHORITY = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md"
)
CNS_H2_FIXED_CONTRACT = {
    "history_len": 2,
    "split_counts": {"train": 2048, "val": 256, "test": 256},
    "max_windows_per_trajectory": 8,
    "epochs": 40,
    "batch_size": 4,
    "training_loss": "mse",
    "metric_family": [
        "err_RMSE",
        "err_nRMSE",
        "relative_l2",
        "fRMSE_low",
        "fRMSE_mid",
        "fRMSE_high",
    ],
    "claim_boundary": "bounded_capped_decision_support_only",
}

CNS_H5_COMPARE_JSON = (
    REPO_ROOT
    / ".artifacts"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "phase-2-pdebench-cns-history5-comparator-gap-fill"
    / "compare_40ep_against_existing.json"
)
CNS_H5_SUMMARY_AUTHORITY = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history5_comparator_gap_fill_summary.md"
)
CNS_H5_FIXED_CONTRACT = {
    "history_len": 5,
    "split_counts": {"train": 512, "val": 64, "test": 64},
    "max_windows_per_trajectory": 8,
    "epochs": 40,
    "batch_size": 4,
    "training_loss": "mse",
    "metric_family": [
        "err_RMSE",
        "err_nRMSE",
        "relative_l2",
        "fRMSE_low",
        "fRMSE_mid",
        "fRMSE_high",
    ],
    "claim_boundary": "bounded_capped_decision_support_only",
}

BRDT_LABELS = {
    "classical_born_backprop": "Model-based Born inverse",
    "unet": "U-Net",
    "fno_vanilla": "FNO",
    "hybrid_resnet": "SRU-Net",
}
BRDT_CONTEXT_ROWS = (
    (
        "classical_born_backprop",
        "Model-based Born inverse",
        "sample_0255_classical_born_backprop_q_pred.npy",
    ),
    ("ffno", "FFNO", "sample_0255_ffno_q_pred.npy"),
    ("hybrid_resnet", "SRU-Net", "sample_0255_hybrid_resnet_q_pred.npy"),
)

CDI_ROW_ORDER = [
    "pinn",
    "pinn_fno_vanilla",
    "pinn_ffno",
    "pinn_hybrid_resnet",
    "pinn_neuralop_uno",
    "baseline",
    "supervised_ffno",
    "supervised_neuralop_uno",
]

CDI_LABELS = {
    "baseline": ("CNN", "supervised"),
    "pinn": ("CNN", "PINN"),
    "pinn_fno_vanilla": ("FNO", "PINN"),
    "pinn_ffno": ("FFNO-local proxy", "PINN"),
    "supervised_ffno": ("FFNO-local proxy", "supervised"),
    "pinn_hybrid_resnet": ("SRU-Net", "PINN"),
    "pinn_neuralop_uno": ("U-NO", "PINN"),
    "supervised_neuralop_uno": ("U-NO", "supervised"),
}


@dataclass(frozen=True)
class BrdtReconPanel:
    row_id: str
    label: str
    q_pred: np.ndarray
    abs_error: np.ndarray


@dataclass(frozen=True)
class BrdtSamplePanels:
    sample_id: int
    measurement_magnitude: np.ndarray
    target_q: np.ndarray
    reconstruction_rows: tuple[BrdtReconPanel, ...]
    reconstruction_vmin: float
    reconstruction_vmax: float
    error_vmax: float


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def center_crop_bounds(shape: tuple[int, int], *, fraction: float = 0.5) -> tuple[int, int, int, int]:
    if len(shape) != 2:
        raise ValueError(f"phase image must be 2D, got shape={shape!r}")
    if not 0 < fraction <= 1:
        raise ValueError(f"crop fraction must be in (0, 1], got {fraction}")
    height, width = shape
    crop_h = max(1, int(round(height * fraction)))
    crop_w = max(1, int(round(width * fraction)))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return y0, y0 + crop_h, x0, x0 + crop_w


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2 * np.pi) - np.pi


def align_phase_to_reference(phase: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if phase.shape != reference.shape:
        raise ValueError(f"phase/reference shape mismatch: {phase.shape} vs {reference.shape}")
    offset = np.angle(np.mean(np.exp(1j * (phase - reference))))
    return wrap_phase(phase - offset)


def _load_phase_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        if "phase" not in payload:
            raise KeyError(f"{path} does not contain a 'phase' array")
        phase = np.asarray(payload["phase"], dtype=np.float32)
    if phase.ndim != 2:
        raise ValueError(f"{path} phase image must be 2D, got shape={phase.shape!r}")
    return phase


def shared_display_bounds(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    finite_values = [
        np.asarray(array, dtype=np.float64)[np.isfinite(array)]
        for array in arrays
    ]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        raise ValueError("cannot compute display bounds from all-NaN phase arrays")
    stacked = np.concatenate(finite_values)
    vmin = float(np.nanmin(stacked))
    vmax = float(np.nanmax(stacked))
    if vmin == vmax:
        pad = 1.0 if vmin == 0.0 else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def robust_display_bounds(
    array: np.ndarray,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[float, float]:
    finite_values = np.asarray(array, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("cannot compute display bounds from all-NaN array")
    vmin = float(np.nanquantile(finite_values, lower_quantile))
    vmax = float(np.nanquantile(finite_values, upper_quantile))
    if vmax <= vmin:
        return shared_display_bounds([finite_values])
    return vmin, vmax


def gt_anchored_phase_bounds(
    display_phases: Mapping[str, np.ndarray],
    crop_bounds: Sequence[int],
    *,
    reference_row: str = "gt",
    upper_quantile: float = 0.99,
) -> tuple[float, float]:
    """Use GT crop bounds so model outliers do not rescale Fig. 1."""
    y0, y1, x0, x1 = [int(value) for value in crop_bounds]
    if reference_row not in display_phases:
        return shared_display_bounds(list(display_phases.values()))
    reference_crop = np.asarray(display_phases[reference_row][y0:y1, x0:x1], dtype=np.float64)
    finite_reference = reference_crop[np.isfinite(reference_crop)]
    if finite_reference.size == 0:
        return shared_display_bounds(list(display_phases.values()))
    vmin = float(np.nanmin(finite_reference))
    vmax = float(np.nanquantile(finite_reference, upper_quantile))
    if vmax <= vmin:
        return shared_display_bounds([reference_crop])
    return vmin, vmax


srunet_matched_phase_bounds = gt_anchored_phase_bounds


def _cdi_phase_zoom_display_phases(
    *,
    recons_root: Path = CDI_RECONS_ROOT,
    crop_fraction: float = 0.5,
) -> tuple[dict[str, np.ndarray], tuple[int, int, int, int]]:
    phases = {
        row_id: _load_phase_array(recons_root / row_id / "recon.npz")
        for row_id, _ in CDI_PHASE_ZOOM_ROWS
    }
    shapes = {phase.shape for phase in phases.values()}
    if len(shapes) != 1:
        raise ValueError(f"phase arrays must share one shape, got {sorted(shapes)!r}")

    shape = next(iter(shapes))
    y0, y1, x0, x1 = center_crop_bounds(shape, fraction=crop_fraction)
    reference = phases["gt"]
    display_phases = {"gt": wrap_phase(reference)}
    for row_id in phases:
        if row_id == "gt":
            continue
        display_phases[row_id] = align_phase_to_reference(phases[row_id], reference)
    return display_phases, (y0, y1, x0, x1)


def _save_cdi_phase_zoom_figure(
    *,
    display_phases: Mapping[str, np.ndarray],
    crop_bounds: Sequence[int],
    output_path: Path,
    bounds_by_row: Mapping[str, tuple[float, float]],
) -> None:
    y0, y1, x0, x1 = [int(value) for value in crop_bounds]

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_count = len(CDI_PHASE_ZOOM_ROWS)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(1.75 * panel_count, 2.1),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (row_id, title) in zip(axes, CDI_PHASE_ZOOM_ROWS):
        phase_vmin, phase_vmax = bounds_by_row[row_id]
        axis.imshow(
            display_phases[row_id][y0:y1, x0:x1],
            cmap="twilight",
            vmin=phase_vmin,
            vmax=phase_vmax,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=8)
        axis.set_axis_off()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_cdi_phase_zoom_figure(
    *,
    recons_root: Path = CDI_RECONS_ROOT,
    output_path: Path = CDI_PHASE_ZOOM_FIGURE,
    crop_fraction: float = 0.5,
) -> dict[str, object]:
    display_phases, crop_bounds = _cdi_phase_zoom_display_phases(
        recons_root=recons_root,
        crop_fraction=crop_fraction,
    )
    y0, y1, x0, x1 = crop_bounds
    phase_vmin, phase_vmax = gt_anchored_phase_bounds(display_phases, crop_bounds)
    bounds_by_row = {
        row_id: (phase_vmin, phase_vmax)
        for row_id, _ in CDI_PHASE_ZOOM_ROWS
    }
    _save_cdi_phase_zoom_figure(
        display_phases=display_phases,
        crop_bounds=crop_bounds,
        output_path=output_path,
        bounds_by_row=bounds_by_row,
    )
    return {
        "figure": str(output_path),
        "source_recons_root": str(recons_root),
        "visible_rows": [row_id for row_id, _ in CDI_PHASE_ZOOM_ROWS],
        "display_channel": "phase",
        "crop_fraction": crop_fraction,
        "crop_bounds": [y0, y1, x0, x1],
        "phase_alignment": "global_circular_offset_to_gt_before_wrapping",
        "phase_colormap": "twilight",
        "phase_display_scale": "gt_crop_min_to_gt_crop_p99_after_alignment",
        "phase_display_bounds": [phase_vmin, phase_vmax],
    }


def write_cdi_phase_zoom_per_panel_figure(
    *,
    recons_root: Path = CDI_RECONS_ROOT,
    output_path: Path = CDI_PHASE_ZOOM_PER_PANEL_FIGURE,
    crop_fraction: float = 0.5,
) -> dict[str, object]:
    display_phases, crop_bounds = _cdi_phase_zoom_display_phases(
        recons_root=recons_root,
        crop_fraction=crop_fraction,
    )
    y0, y1, x0, x1 = crop_bounds
    bounds_by_row = {
        row_id: robust_display_bounds(display_phases[row_id][y0:y1, x0:x1])
        for row_id, _ in CDI_PHASE_ZOOM_ROWS
    }
    _save_cdi_phase_zoom_figure(
        display_phases=display_phases,
        crop_bounds=crop_bounds,
        output_path=output_path,
        bounds_by_row=bounds_by_row,
    )
    return {
        "figure": str(output_path),
        "source_recons_root": str(recons_root),
        "visible_rows": [row_id for row_id, _ in CDI_PHASE_ZOOM_ROWS],
        "display_channel": "phase",
        "crop_fraction": crop_fraction,
        "crop_bounds": [y0, y1, x0, x1],
        "phase_alignment": "global_circular_offset_to_gt_before_wrapping",
        "phase_colormap": "twilight",
        "phase_display_scale": "per_panel_p01_to_p99_after_alignment",
        "phase_display_bounds_by_row": {
            row_id: [bounds[0], bounds[1]]
            for row_id, bounds in bounds_by_row.items()
        },
        "caption_note": (
            "Each panel is contrast-normalized independently; colors are not "
            "comparable across panels."
        )
    }


def detect_cns_history5_gaps(
    available_rows: Mapping[str, Mapping[str, object]],
    *,
    required_rows: Sequence[str],
    history_len: int = 5,
    epochs: int = 40,
) -> list[str]:
    gaps: list[str] = []
    for row_id in required_rows:
        row = dict(available_rows.get(row_id, {}))
        if int(row.get("history_len", -1)) != history_len or int(row.get("epochs", -1)) != epochs:
            gaps.append(row_id)
    return gaps


def audit_cns_history5_availability() -> dict[str, object]:
    gaps = detect_cns_history5_gaps(
        KNOWN_CNS_HISTORY5_ROWS,
        required_rows=REQUIRED_CNS_HISTORY5_ROWS,
    )
    return {
        "required_rows": REQUIRED_CNS_HISTORY5_ROWS,
        "available_rows": KNOWN_CNS_HISTORY5_ROWS,
        "missing_rows": gaps,
    }


def _load_npy_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path)).squeeze()


def load_brdt_sample255_panels(
    *,
    source_arrays: Path = BRDT_40EP_SOURCE_ARRAYS,
) -> BrdtSamplePanels:
    source_arrays = Path(source_arrays)
    target_q = _load_npy_array(source_arrays / "sample_0255_q_target.npy")
    sino_obs = np.asarray(np.load(source_arrays / "sample_0255_sino_obs.npy"))
    if sino_obs.ndim < 3:
        raise ValueError(
            "sample_0255_sino_obs.npy must include a complex channel, "
            f"got {sino_obs.shape!r}"
        )
    measurement_magnitude = np.linalg.norm(sino_obs, axis=-1)

    rows = []
    for row_id, label, filename in BRDT_CONTEXT_ROWS:
        q_pred = _load_npy_array(source_arrays / filename)
        if q_pred.shape != target_q.shape:
            raise ValueError(
                f"{filename} shape {q_pred.shape!r} does not match target {target_q.shape!r}"
            )
        rows.append(
            BrdtReconPanel(
                row_id=row_id,
                label=label,
                q_pred=q_pred,
                abs_error=np.abs(q_pred - target_q),
            )
        )

    reconstruction_vmin = float(np.nanmin(target_q))
    reconstruction_vmax = float(np.nanmax(target_q))
    error_vmax = float(max(np.nanmax(row.abs_error) for row in rows))
    if reconstruction_vmax <= reconstruction_vmin:
        reconstruction_vmin, reconstruction_vmax = shared_display_bounds([target_q])
    if error_vmax <= 0:
        error_vmax = 1.0
    return BrdtSamplePanels(
        sample_id=255,
        measurement_magnitude=measurement_magnitude,
        target_q=target_q,
        reconstruction_rows=tuple(rows),
        reconstruction_vmin=reconstruction_vmin,
        reconstruction_vmax=reconstruction_vmax,
        error_vmax=error_vmax,
    )


def brdt_context_panel_titles(panels: BrdtSamplePanels) -> dict[str, list[str]]:
    prediction_titles = {
        "classical_born_backprop": r"Model-based: $\hat{q}_{Born}$",
        "ffno": r"FFNO: $\hat{q}$",
        "hybrid_resnet": r"SRU-Net: $\hat{q}$",
    }
    error_titles = {
        "classical_born_backprop": r"$|\hat{q}_{Born}-q|$",
        "ffno": r"$|\hat{q}_{FFNO}-q|$",
        "hybrid_resnet": r"$|\hat{q}_{SRU-Net}-q|$",
    }
    return {
        "top": [
            r"Target: $q$",
            *[prediction_titles[row.row_id] for row in panels.reconstruction_rows],
        ],
        "bottom": [
            r"Input: $|s_{obs}(\theta,d)|$",
            *[error_titles[row.row_id] for row in panels.reconstruction_rows],
        ],
    }


def write_brdt_context_figure(
    *,
    output_path: Path = BRDT_CONTEXT_FIGURE,
    source_arrays: Path = BRDT_40EP_SOURCE_ARRAYS,
) -> Path:
    panels = load_brdt_sample255_panels(source_arrays=source_arrays)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.5, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)

    top_axes = [fig.add_subplot(grid[0, col]) for col in range(4)]
    bottom_axes = [fig.add_subplot(grid[1, col]) for col in range(4)]
    titles = brdt_context_panel_titles(panels)
    top_panels = [(r"Target: $q$", panels.target_q)] + [
        (row.label, row.q_pred) for row in panels.reconstruction_rows
    ]
    q_im = None
    for axis, title, (_, image) in zip(top_axes, titles["top"], top_panels):
        q_im = axis.imshow(
            image,
            cmap=BRDT_RECONSTRUCTION_CMAP,
            vmin=panels.reconstruction_vmin,
            vmax=panels.reconstruction_vmax,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=8)
        axis.set_axis_off()

    context_im = bottom_axes[0].imshow(
        panels.measurement_magnitude,
        cmap=BRDT_MEASUREMENT_CMAP,
        interpolation="nearest",
        aspect="auto",
    )
    bottom_axes[0].set_title(titles["bottom"][0], fontsize=8)
    bottom_axes[0].set_xlabel("detector sample $d$", fontsize=7)
    bottom_axes[0].set_ylabel(r"angle index $\theta$", fontsize=7)
    bottom_axes[0].tick_params(labelsize=6, length=2)

    error_im = None
    for axis, title, row in zip(bottom_axes[1:], titles["bottom"][1:], panels.reconstruction_rows):
        error_im = axis.imshow(
            row.abs_error,
            cmap=BRDT_ERROR_CMAP,
            vmin=0.0,
            vmax=panels.error_vmax,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=7)
        axis.set_axis_off()

    if q_im is not None:
        q_cbar = fig.colorbar(
            q_im,
            ax=top_axes,
            shrink=0.72,
            fraction=0.025,
            pad=0.01,
        )
        q_cbar.set_label("q", fontsize=7)
    if error_im is not None:
        error_cbar = fig.colorbar(
            error_im,
            ax=bottom_axes[1:],
            shrink=0.72,
            fraction=0.025,
            pad=0.01,
        )
        error_cbar.set_label("absolute error", fontsize=7)
    measurement_cbar = fig.colorbar(
        context_im,
        ax=bottom_axes[0],
        shrink=0.72,
        fraction=0.04,
        pad=0.03,
    )
    measurement_cbar.set_label("sinogram magnitude", fontsize=7)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def _brdt_value(row: Mapping[str, object], section: str, key: str) -> float:
    section_payload = row.get(section, {})
    if not isinstance(section_payload, Mapping):
        raise KeyError(f"Expected mapping at {section}")
    return float(section_payload[key])


def normalized_brdt_rows(metrics_payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_row in metrics_payload["rows"]:  # type: ignore[index]
        row = dict(raw_row)
        row_id = str(row["row_id"])
        status = str(row.get("row_status", row.get("status", "")))
        label = str(BRDT_LABELS.get(row_id, row.get("paper_label", row_id.replace("_", " "))))
        display = {
            "row_id": row_id,
            "label": label,
            "status": status,
        }
        if status != "blocked":
            display.update(
                {
                    "image_relative_l2_phys": _brdt_value(row, "image", "image_relative_l2_phys"),
                    "meas_relative_l2": _brdt_value(row, "measurement", "meas_relative_l2"),
                    "psnr_phys": _brdt_value(row, "supporting", "psnr_phys"),
                    "ssim_phys": _brdt_value(row, "supporting", "ssim_phys"),
                }
            )
        rows.append(display)
    return rows


def render_brdt_metrics_table(metrics_payload: Mapping[str, object]) -> str:
    rows = normalized_brdt_rows(metrics_payload)
    completed_rows = [row for row in rows if row["status"] != "blocked"]
    best_values = {}
    if completed_rows:
        best_values = {
            "image_relative_l2_phys": min(float(row["image_relative_l2_phys"]) for row in completed_rows),
            "meas_relative_l2": min(float(row["meas_relative_l2"]) for row in completed_rows),
            "psnr_phys": max(float(row["psnr_phys"]) for row in completed_rows),
            "ssim_phys": max(float(row["ssim_phys"]) for row in completed_rows),
        }

    def metric_cell(row: Mapping[str, object], key: str, precision: int) -> str:
        value = float(row[key])
        formatted = f"{value:.{precision}f}"
        if key in best_values and abs(value - best_values[key]) <= 0.5 * 10 ** (-precision):
            return rf"\textbf{{{formatted}}}"
        return formatted

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Row & Image rel. $L_2$ $\downarrow$ & Meas. rel. $L_2$ $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        label = _latex_escape(row["label"])
        status = str(row["status"])
        if status == "blocked":
            lines.append(f"{label} & -- & -- & -- & -- \\\\")
            continue
        lines.append(
            f"{label} & {metric_cell(row, 'image_relative_l2_phys', 3)} & "
            f"{metric_cell(row, 'meas_relative_l2', 3)} & "
            f"{metric_cell(row, 'psnr_phys', 2)} & "
            f"{metric_cell(row, 'ssim_phys', 3)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_brdt_assets() -> dict[str, str]:
    payload = _read_json(BRDT_METRICS_JSON)
    rows = normalized_brdt_rows(payload)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = TABLES_DIR / "brdt_decision_support_metrics.tex"
    csv_path = TABLES_DIR / "brdt_decision_support_metrics.csv"
    json_path = TABLES_DIR / "brdt_decision_support_metrics.json"
    fig_path = FIGURES_DIR / "brdt_decision_support_recon.png"

    tex_path.write_text(render_brdt_metrics_table(payload), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "label",
                "status",
                "image_relative_l2_phys",
                "meas_relative_l2",
                "psnr_phys",
                "ssim_phys",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        json_path,
        {
            "claim_boundary": "decision_support_preflight_only",
            "source_metrics_json": str(BRDT_METRICS_JSON.relative_to(REPO_ROOT)),
            "source_figure": str(BRDT_SOURCE_FIGURE.relative_to(REPO_ROOT)),
            "paper_figure": str(fig_path.relative_to(NEURIPS_DIR)),
            "rows": rows,
        },
    )
    shutil.copyfile(BRDT_SOURCE_FIGURE, fig_path)
    return {
        "tex": str(tex_path),
        "csv": str(csv_path),
        "json": str(json_path),
        "figure": str(fig_path),
    }


def _pair(payload: Mapping[str, object], key: str) -> tuple[float, float]:
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise KeyError("metrics")
    raw = metrics[key]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ValueError(f"Metric {key!r} must be a two-value sequence")
    return float(raw[0]), float(raw[1])


def cdi_display_metrics(metrics_payload: Mapping[str, object]) -> list[dict[str, object]]:
    source_rows = metrics_payload.get("rows", metrics_payload)
    if not isinstance(source_rows, Mapping):
        raise ValueError("CDI metrics payload must be a row mapping or contain a 'rows' mapping")

    rows: list[dict[str, object]] = []
    for row_id in CDI_ROW_ORDER:
        if row_id not in source_rows:
            continue
        payload = dict(source_rows[row_id])
        amp_mae, phase_mae = _pair(payload, "mae")
        amp_mse, phase_mse = _pair(payload, "mse")
        amp_ssim, phase_ssim = _pair(payload, "ssim")
        model, training = CDI_LABELS[row_id]
        rows.append(
            {
                "row_id": row_id,
                "model": model,
                "training": training,
                "amp_mae": amp_mae,
                "phase_mae": phase_mae,
                "amp_mse": amp_mse,
                "phase_mse": phase_mse,
                "amp_ssim": amp_ssim,
                "phase_ssim": phase_ssim,
            }
        )
    return rows


CDI_METRIC_COLUMNS = [
    ("amp_mae", False),
    ("phase_mae", False),
    ("amp_mse", False),
    ("phase_mse", False),
    ("amp_ssim", True),
    ("phase_ssim", True),
]
CDI_OBJECTIVE_CONTROL_COLUMNS = [
    ("amp_mae", False),
    ("phase_mae", False),
    ("amp_ssim", True),
    ("phase_ssim", True),
]


def _cdi_best_values(
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[tuple[str, bool]] = CDI_METRIC_COLUMNS,
) -> dict[str, float]:
    best: dict[str, float] = {}
    for key, higher_is_better in columns:
        values = [float(row[key]) for row in rows]
        best[key] = max(values) if higher_is_better else min(values)
    return best


def _format_cdi_metric(row: Mapping[str, object], key: str, best: Mapping[str, float]) -> str:
    value = float(row[key])
    formatted = f"{value:.4f}"
    if abs(value - best[key]) <= 5e-5:
        return rf"\textbf{{{formatted}}}"
    return formatted


def _formatted_cdi_values(
    row: Mapping[str, object],
    best: Mapping[str, float],
    *,
    columns: Sequence[tuple[str, bool]] = CDI_METRIC_COLUMNS,
) -> list[str]:
    return [_format_cdi_metric(row, key, best) for key, _ in columns]


def render_cdi_pinn_metrics_table(rows: Sequence[Mapping[str, object]]) -> str:
    pinn_rows = [row for row in rows if row.get("training") == "PINN"]
    best = _cdi_best_values(pinn_rows)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Amp MAE $\downarrow$ & Phase MAE $\downarrow$ & Amp MSE $\downarrow$ & Phase MSE $\downarrow$ & Amp SSIM $\uparrow$ & Phase SSIM $\uparrow$ \\",
        r"\midrule",
    ]
    for row in pinn_rows:
        values = _formatted_cdi_values(row, best)
        lines.append(f"{_latex_escape(row['model'])} & {' & '.join(values)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_cdi_objective_comparison_table(rows: Sequence[Mapping[str, object]]) -> str:
    rows_by_model: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        model = str(row["model"])
        training = str(row["training"])
        rows_by_model.setdefault(model, {})[training] = row

    paired_models = [
        model
        for model, model_rows in rows_by_model.items()
        if "PINN" in model_rows and "supervised" in model_rows
    ]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Objective & Amp MAE & Phase MAE & Amp SSIM & Phase SSIM \\",
    ]
    for model in paired_models:
        model_rows = [rows_by_model[model]["PINN"], rows_by_model[model]["supervised"]]
        best = _cdi_best_values(model_rows, columns=CDI_OBJECTIVE_CONTROL_COLUMNS)
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{_latex_escape(model)}}}}} \\")
        lines.append(r"\midrule")
        for row in model_rows:
            label = "PINN" if row["training"] == "PINN" else "Supervised"
            values = _formatted_cdi_values(row, best, columns=CDI_OBJECTIVE_CONTROL_COLUMNS)
            lines.append(f"{label} & {' & '.join(values)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_cdi_metrics_table(rows: Sequence[Mapping[str, object]]) -> str:
    return render_cdi_pinn_metrics_table(rows)


def write_cdi_extended_assets() -> dict[str, str]:
    payload = _read_json(CDI_UNO_METRICS_JSON)
    source_rows = dict(payload["rows"])
    source_rows["supervised_ffno"] = {"metrics": _read_json(CDI_SUPERVISED_FFNO_METRICS_JSON)}
    payload = {**payload, "rows": source_rows}
    rows = cdi_display_metrics(payload)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = TABLES_DIR / "cdi_lines128_metrics_extended.tex"
    pinn_tex_path = TABLES_DIR / "cdi_lines128_pinn_metrics.tex"
    objective_tex_path = TABLES_DIR / "cdi_lines128_objective_comparison.tex"
    csv_path = TABLES_DIR / "cdi_lines128_metrics_extended.csv"
    json_path = TABLES_DIR / "cdi_lines128_metrics_extended.json"

    tex_path.write_text(render_cdi_metrics_table(rows), encoding="utf-8")
    pinn_tex_path.write_text(render_cdi_pinn_metrics_table(rows), encoding="utf-8")
    objective_tex_path.write_text(render_cdi_objective_comparison_table(rows), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        json_path,
        {
            "claim_boundary": "complete_lines128_cdi_benchmark_plus_uno_extension",
            "source_metrics_json": str(CDI_UNO_METRICS_JSON.relative_to(REPO_ROOT)),
            "supervised_ffno_source_metrics_json": str(
                CDI_SUPERVISED_FFNO_METRICS_JSON.relative_to(REPO_ROOT)
            ),
            "rows": rows,
        },
    )
    return {
        "tex": str(tex_path),
        "pinn_tex": str(pinn_tex_path),
        "objective_tex": str(objective_tex_path),
        "csv": str(csv_path),
        "json": str(json_path),
    }


def load_cns_h2_authority_rows(
    *,
    bundle_table_json: Path = CNS_H2_BUNDLE_TABLE_JSON,
) -> dict[str, object]:
    payload = _read_json(bundle_table_json)
    headline_rows = list(payload.get("headline_rows", []))
    rows_by_id = {str(row["row_id"]): dict(row) for row in headline_rows}
    return {
        "lane_id": "h2_2048_256_256_40ep",
        "contract": dict(CNS_H2_FIXED_CONTRACT),
        "summary_authority": CNS_H2_SUMMARY_AUTHORITY,
        "contract_authority": CNS_H2_CONTRACT_AUTHORITY,
        "source_bundle_json": str(bundle_table_json.relative_to(REPO_ROOT)),
        "rows_by_id": rows_by_id,
    }


def _verify_source_contract_block(
    source_label: str,
    source_contract: Mapping[str, object],
    expected_contract: Mapping[str, object],
) -> None:
    fields = (
        "history_len",
        "epochs",
        "batch_size",
        "training_loss",
        "max_windows_per_trajectory",
    )
    mismatches: list[str] = []
    for field in fields:
        if field not in source_contract:
            mismatches.append(f"{field}_missing")
            continue
        if str(source_contract[field]) != str(expected_contract[field]):
            mismatches.append(
                f"{field}_mismatch(source={source_contract[field]!r},"
                f"expected={expected_contract[field]!r})"
            )
    expected_split = {k: int(v) for k, v in dict(expected_contract["split_counts"]).items()}  # type: ignore[arg-type]
    actual_split_raw = source_contract.get("split_counts")
    if not isinstance(actual_split_raw, Mapping):
        mismatches.append("split_counts_missing")
    else:
        actual_split = {k: int(v) for k, v in dict(actual_split_raw).items()}
        if actual_split != expected_split:
            mismatches.append(
                f"split_counts_mismatch(source={actual_split},expected={expected_split})"
            )
    expected_metrics = list(expected_contract["metric_family"])  # type: ignore[arg-type]
    actual_metrics = list(source_contract.get("metric_family", []))
    if actual_metrics != expected_metrics:
        mismatches.append(
            f"metric_family_mismatch(source={actual_metrics},expected={expected_metrics})"
        )
    if mismatches:
        raise RuntimeError(
            f"{source_label} top-level contract block disagrees with the matched-condition "
            f"fixed contract: {mismatches}"
        )


def load_cns_h5_candidate_rows(
    *,
    compare_json: Path = CNS_H5_COMPARE_JSON,
) -> dict[str, object]:
    payload = _read_json(compare_json)
    source_contract = payload.get("contract")
    if not isinstance(source_contract, Mapping):
        raise RuntimeError(
            f"CNS h5 compare {compare_json} is missing a top-level 'contract' block; "
            "matched-condition selection requires the source to declare its fixed contract"
        )
    _verify_source_contract_block(
        source_label=f"CNS h5 compare {compare_json.name}",
        source_contract=source_contract,
        expected_contract=CNS_H5_FIXED_CONTRACT,
    )
    rows_by_id: dict[str, dict[str, object]] = {}
    for raw in payload.get("profile_results", []):
        row = dict(raw)
        row_id = str(row.get("profile_id", ""))
        if not row_id:
            continue
        rows_by_id[row_id] = row
    try:
        compare_json_label = str(compare_json.relative_to(REPO_ROOT))
    except ValueError:
        compare_json_label = str(compare_json)
    return {
        "lane_id": "h5_512_64_64_40ep",
        "contract": dict(CNS_H5_FIXED_CONTRACT),
        "summary_authority": CNS_H5_SUMMARY_AUTHORITY,
        "source_compare_json": compare_json_label,
        "source_contract": dict(source_contract),
        "rows_by_id": rows_by_id,
    }


def _h5_row_consistent(row: Mapping[str, object], contract: Mapping[str, object]) -> list[str]:
    issues: list[str] = []
    if int(row.get("history_len", -1)) != int(contract["history_len"]):
        issues.append("history_len_mismatch")
    if int(row.get("epochs", -1)) != int(contract["epochs"]):
        issues.append("epochs_mismatch")
    if int(row.get("batch_size", -1)) != int(contract["batch_size"]):
        issues.append("batch_size_mismatch")
    if str(row.get("training_loss", "")) != str(contract["training_loss"]):
        issues.append("training_loss_mismatch")
    if "max_windows_per_trajectory" not in row:
        issues.append("max_windows_per_trajectory_missing")
    elif int(row["max_windows_per_trajectory"]) != int(contract["max_windows_per_trajectory"]):
        issues.append("max_windows_per_trajectory_mismatch")
    expected_split = dict(contract["split_counts"])  # type: ignore[arg-type]
    actual_split = dict(row.get("split_counts", {}))
    if {k: int(v) for k, v in actual_split.items()} != {k: int(v) for k, v in expected_split.items()}:
        issues.append("split_counts_mismatch")
    expected_metrics = list(contract["metric_family"])  # type: ignore[arg-type]
    actual_metrics = list(row.get("metric_family", []))
    if actual_metrics != expected_metrics:
        issues.append("metric_family_mismatch")
    if str(row.get("status", "")) != "completed":
        issues.append("status_not_completed")
    return issues


def evaluate_cns_h5_lane(
    h5_lane: Mapping[str, object],
    *,
    required_rows: Sequence[str] = tuple(CNS_REQUIRED_HEADLINE_ROWS),
) -> dict[str, object]:
    contract = dict(h5_lane["contract"])  # type: ignore[arg-type]
    rows_by_id = dict(h5_lane["rows_by_id"])  # type: ignore[arg-type]
    missing: list[str] = []
    inconsistent: list[dict[str, object]] = []
    for row_id in required_rows:
        if row_id not in rows_by_id:
            missing.append(row_id)
            continue
        issues = _h5_row_consistent(rows_by_id[row_id], contract)
        if issues:
            inconsistent.append({"row_id": row_id, "issues": issues})
    is_complete = not missing and not inconsistent
    return {
        "lane_id": h5_lane["lane_id"],
        "is_complete_and_consistent": is_complete,
        "missing_rows": missing,
        "inconsistent_rows": inconsistent,
    }


def _h2_row_to_payload_row(
    row: Mapping[str, object],
    *,
    contract: Mapping[str, object],
) -> dict[str, object]:
    row_id = str(row["row_id"])
    return {
        "row_id": row_id,
        "manuscript_label": CNS_MANUSCRIPT_LABELS[row_id],
        "row_role": "headline",
        "row_status": "capped_decision_support",
        "claim_scope": str(contract["claim_boundary"]),
        "history_len": int(contract["history_len"]),
        "split_counts": dict(contract["split_counts"]),  # type: ignore[arg-type]
        "split_label": str(row.get("split_label", "")),
        "epochs": int(contract["epochs"]),
        "batch_size": int(contract["batch_size"]),
        "training_loss": str(contract["training_loss"]),
        "err_nRMSE": row.get("err_nRMSE"),
        "err_RMSE": row.get("err_RMSE"),
        "relative_l2": row.get("relative_l2"),
        "fRMSE_low": row.get("fRMSE_low"),
        "fRMSE_mid": row.get("fRMSE_mid"),
        "fRMSE_high": row.get("fRMSE_high"),
        "parameter_count": row.get("parameter_count"),
        "runtime_sec": row.get("runtime_sec"),
        "source_run_root": str(row.get("source_run_root", "")),
    }


def _h5_row_to_payload_row(
    row: Mapping[str, object],
    *,
    contract: Mapping[str, object],
) -> dict[str, object]:
    row_id = str(row["profile_id"])
    split_counts = dict(row.get("split_counts", contract["split_counts"]))  # type: ignore[arg-type]
    split_label = (
        f"{int(split_counts['train'])} / {int(split_counts['val'])} / "
        f"{int(split_counts['test'])}"
    )
    return {
        "row_id": row_id,
        "manuscript_label": CNS_MANUSCRIPT_LABELS[row_id],
        "row_role": "headline",
        "row_status": "capped_decision_support",
        "claim_scope": str(contract["claim_boundary"]),
        "history_len": int(row.get("history_len", contract["history_len"])),
        "split_counts": split_counts,
        "split_label": split_label,
        "epochs": int(row.get("epochs", contract["epochs"])),
        "batch_size": int(row.get("batch_size", contract["batch_size"])),
        "training_loss": str(row.get("training_loss", contract["training_loss"])),
        "err_nRMSE": row.get("err_nRMSE"),
        "err_RMSE": row.get("err_RMSE"),
        "relative_l2": row.get("relative_l2"),
        "fRMSE_low": row.get("fRMSE_low"),
        "fRMSE_mid": row.get("fRMSE_mid"),
        "fRMSE_high": row.get("fRMSE_high"),
        "parameter_count": row.get("parameter_count"),
        "runtime_sec": row.get("runtime_sec"),
        "source_run_root": str(row.get("run_root", "")),
    }


def select_cns_matched_condition(
    *,
    h2_lane: Mapping[str, object] | None = None,
    h5_lane: Mapping[str, object] | None = None,
    required_rows: Sequence[str] = tuple(CNS_REQUIRED_HEADLINE_ROWS),
) -> dict[str, object]:
    if h2_lane is None:
        h2_lane = load_cns_h2_authority_rows()
    if h5_lane is None:
        h5_lane = load_cns_h5_candidate_rows()

    h5_eval = evaluate_cns_h5_lane(h5_lane, required_rows=required_rows)
    h2_rows_by_id = dict(h2_lane["rows_by_id"])  # type: ignore[arg-type]
    h2_missing = [row_id for row_id in required_rows if row_id not in h2_rows_by_id]
    if h2_missing:
        raise RuntimeError(
            f"CNS h2 authority bundle missing required rows: {h2_missing}; cannot fall back"
        )

    h2_contract = dict(h2_lane["contract"])  # type: ignore[arg-type]
    h5_contract = dict(h5_lane["contract"])  # type: ignore[arg-type]

    if h5_eval["is_complete_and_consistent"]:
        selected_lane_id = str(h5_lane["lane_id"])
        selected_contract = h5_contract
        selected_rows = [
            _h5_row_to_payload_row(
                h5_lane["rows_by_id"][row_id],  # type: ignore[index]
                contract=h5_contract,
            )
            for row_id in required_rows
        ]
        rejected_candidate = None
        selection_reason = (
            "h5 lane has all four required rows present and internally consistent under "
            "the fixed 512/64/64, history_len=5, 40-epoch contract"
        )
        source_summary_paths = [
            str(h5_lane["summary_authority"]),
            str(h2_lane["summary_authority"]),
        ]
        same_condition_visuals_available = False
        same_condition_visuals_reason = (
            "h5 lane rows live in three different run roots; no single fixed-sample "
            "bundle covers all four manuscript rows on the same h5 trajectory"
        )
    else:
        selected_lane_id = str(h2_lane["lane_id"])
        selected_contract = h2_contract
        selected_rows = [
            _h2_row_to_payload_row(h2_rows_by_id[row_id], contract=h2_contract)
            for row_id in required_rows
        ]
        rejected_candidate = {
            "lane_id": h5_lane["lane_id"],
            "missing_rows": h5_eval["missing_rows"],
            "inconsistent_rows": h5_eval["inconsistent_rows"],
        }
        selection_reason = (
            "h5 lane was incomplete or inconsistent against the fixed 512/64/64, "
            "history_len=5, 40-epoch contract; falling back to the locked h2 "
            "2048/256/256 authority"
        )
        source_summary_paths = [
            str(h2_lane["summary_authority"]),
            str(h2_lane["contract_authority"]),
        ]
        same_condition_visuals_available = True
        same_condition_visuals_reason = (
            "h2 2048cap bundle includes a same-contract fixed-sample manifest covering "
            "all four headline rows"
        )

    return {
        "schema_version": "pdebench_cns_matched_condition_decision_v1",
        "selected_lane_id": selected_lane_id,
        "selected_contract": selected_contract,
        "selected_rows": selected_rows,
        "selected_row_ids": [str(row["row_id"]) for row in selected_rows],
        "rejected_candidate": rejected_candidate,
        "selection_reason": selection_reason,
        "h5_evaluation": h5_eval,
        "source_summary_paths": source_summary_paths,
        "same_condition_visuals_available": same_condition_visuals_available,
        "same_condition_visuals_reason": same_condition_visuals_reason,
        "claim_boundary": str(selected_contract["claim_boundary"]),
        "h2_authority_summary": str(h2_lane["summary_authority"]),
        "h5_authority_summary": str(h5_lane["summary_authority"]),
    }


def render_cns_matched_condition_tex(decision: Mapping[str, object]) -> str:
    rows = list(decision["selected_rows"])  # type: ignore[arg-type]

    def _fmt(value: object, precision: int) -> str:
        if value in (None, ""):
            return r"\textit{missing}"
        return f"{float(value):.{precision}f}"

    lines = [
        "% Auto-generated CNS matched-condition headline table",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & nRMSE $\\downarrow$ & Low RMSE $\\downarrow$ & "
        "Mid RMSE $\\downarrow$ & High RMSE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        raw_label = str(row["manuscript_label"])
        label = _latex_escape(raw_label.replace("*", r"$^*$"))
        lines.append(
            " & ".join(
                [
                    label,
                    _fmt(row.get("err_nRMSE"), 4),
                    _fmt(row.get("fRMSE_low"), 3),
                    _fmt(row.get("fRMSE_mid"), 3),
                    _fmt(row.get("fRMSE_high"), 3),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


_CNS_MATCHED_CSV_FIELDS = [
    "row_id",
    "manuscript_label",
    "row_role",
    "row_status",
    "claim_scope",
    "history_len",
    "split_label",
    "epochs",
    "batch_size",
    "training_loss",
    "err_nRMSE",
    "err_RMSE",
    "relative_l2",
    "fRMSE_low",
    "fRMSE_mid",
    "fRMSE_high",
    "parameter_count",
    "runtime_sec",
    "source_run_root",
]


def _build_cns_matched_condition_payload(decision: Mapping[str, object]) -> dict[str, object]:
    contract = dict(decision["selected_contract"])  # type: ignore[arg-type]
    return {
        "schema_version": "pdebench_cns_matched_condition_table_v1",
        "selected_lane_id": str(decision["selected_lane_id"]),
        "claim_boundary": str(contract["claim_boundary"]),
        "fixed_contract": {
            "history_len": contract["history_len"],
            "split_counts": contract["split_counts"],
            "max_windows_per_trajectory": contract["max_windows_per_trajectory"],
            "epochs": contract["epochs"],
            "batch_size": contract["batch_size"],
            "training_loss": contract["training_loss"],
            "metric_family": contract["metric_family"],
            "claim_boundary": contract["claim_boundary"],
            "source_summary_paths": list(decision["source_summary_paths"]),  # type: ignore[arg-type]
        },
        "headline_row_ids": list(decision["selected_row_ids"]),  # type: ignore[arg-type]
        "rows": list(decision["selected_rows"]),  # type: ignore[arg-type]
    }


def _build_cns_matched_condition_lineage(decision: Mapping[str, object]) -> dict[str, object]:
    rows_lineage = []
    for row in decision["selected_rows"]:  # type: ignore[arg-type]
        rows_lineage.append(
            {
                "row_id": row["row_id"],
                "manuscript_label": row["manuscript_label"],
                "source_run_root": row["source_run_root"],
            }
        )
    return {
        "schema_version": "pdebench_cns_matched_condition_lineage_v1",
        "selected_lane_id": str(decision["selected_lane_id"]),
        "h2_authority_summary": decision["h2_authority_summary"],
        "h5_authority_summary": decision["h5_authority_summary"],
        "source_summary_paths": list(decision["source_summary_paths"]),  # type: ignore[arg-type]
        "rejected_candidate": decision.get("rejected_candidate"),
        "selection_reason": decision["selection_reason"],
        "rows": rows_lineage,
    }


def _build_cns_matched_condition_figure_selection(
    decision: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "pdebench_cns_matched_condition_figure_selection_v1",
        "selected_lane_id": str(decision["selected_lane_id"]),
        "same_condition_visuals_available": bool(
            decision["same_condition_visuals_available"]
        ),
        "same_condition_visuals_reason": str(decision["same_condition_visuals_reason"]),
        "selected_row_ids": list(decision["selected_row_ids"]),  # type: ignore[arg-type]
        "manuscript_figure_role": (
            "same_condition_headline"
            if decision["same_condition_visuals_available"]
            else "adjacent_context_only"
        ),
    }


def write_cns_matched_condition_assets(
    decision: Mapping[str, object],
    *,
    output_root: Path,
) -> dict[str, str]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    decision_path = output_root / "matched_condition_decision.json"
    table_json_path = output_root / "cns_paper_table_rows.json"
    table_csv_path = output_root / "cns_paper_table_rows.csv"
    table_tex_path = output_root / "cns_paper_table_rows.tex"
    lineage_path = output_root / "source_lineage.json"
    figure_selection_path = output_root / "figure_selection.json"

    _write_json(decision_path, decision)
    payload = _build_cns_matched_condition_payload(decision)
    _write_json(table_json_path, payload)

    with table_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_CNS_MATCHED_CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({field: row.get(field, "") for field in _CNS_MATCHED_CSV_FIELDS})

    table_tex_path.write_text(render_cns_matched_condition_tex(decision), encoding="utf-8")
    _write_json(lineage_path, _build_cns_matched_condition_lineage(decision))
    _write_json(
        figure_selection_path, _build_cns_matched_condition_figure_selection(decision)
    )

    return {
        "matched_condition_decision_json": str(decision_path),
        "cns_paper_table_rows_json": str(table_json_path),
        "cns_paper_table_rows_csv": str(table_csv_path),
        "cns_paper_table_rows_tex": str(table_tex_path),
        "source_lineage_json": str(lineage_path),
        "figure_selection_json": str(figure_selection_path),
    }


def write_cns_matched_condition_paper_assets(
    decision: Mapping[str, object],
    *,
    tables_dir: Path = TABLES_DIR,
) -> dict[str, str]:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    base = "pdebench_cns_matched_condition_metrics"
    tex_path = tables_dir / f"{base}.tex"
    csv_path = tables_dir / f"{base}.csv"
    json_path = tables_dir / f"{base}.json"

    payload = _build_cns_matched_condition_payload(decision)
    _write_json(json_path, payload)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_CNS_MATCHED_CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({field: row.get(field, "") for field in _CNS_MATCHED_CSV_FIELDS})
    tex_path.write_text(render_cns_matched_condition_tex(decision), encoding="utf-8")
    return {
        "tex": str(tex_path),
        "csv": str(csv_path),
        "json": str(json_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-cns-history5", action="store_true")
    parser.add_argument(
        "--audit-cns-matched-condition",
        action="store_true",
        help="Run the matched-condition selector and emit the decision payload only.",
    )
    parser.add_argument(
        "--write-cns-matched-condition-assets",
        type=Path,
        default=None,
        help="Write matched-condition refresh assets into the given item-local root.",
    )
    parser.add_argument(
        "--write-cns-matched-condition-paper-assets",
        action="store_true",
        help="Also emit paper-local CNS matched-condition table assets under tables/.",
    )
    parser.add_argument("--write-brdt-assets", action="store_true")
    parser.add_argument(
        "--write-brdt-context-figure",
        action="store_true",
        help="Write the BRDT sample-255 measurement/reconstruction/error figure.",
    )
    parser.add_argument("--write-cdi-extended-assets", action="store_true")
    parser.add_argument("--write-cdi-phase-zoom-figure", action="store_true")
    parser.add_argument("--write-cdi-phase-zoom-per-panel-figure", action="store_true")
    parser.add_argument(
        "--write-model-config-table",
        action="store_true",
        help="Emit paper-local model configuration appendix table assets under tables/.",
    )
    parser.add_argument(
        "--write-efficiency-table",
        action="store_true",
        help="Emit paper-local efficiency table assets under tables/.",
    )
    args = parser.parse_args(argv)

    outputs: dict[str, object] = {}
    cns_matched_decision: dict[str, object] | None = None
    if args.audit_cns_history5:
        outputs["cns_history5_audit"] = audit_cns_history5_availability()
    if (
        args.audit_cns_matched_condition
        or args.write_cns_matched_condition_assets is not None
        or args.write_cns_matched_condition_paper_assets
    ):
        cns_matched_decision = select_cns_matched_condition()
        outputs["cns_matched_condition_decision"] = cns_matched_decision
    if args.write_cns_matched_condition_assets is not None and cns_matched_decision is not None:
        outputs["cns_matched_condition_assets"] = write_cns_matched_condition_assets(
            cns_matched_decision,
            output_root=args.write_cns_matched_condition_assets,
        )
    if args.write_cns_matched_condition_paper_assets and cns_matched_decision is not None:
        outputs["cns_matched_condition_paper_assets"] = (
            write_cns_matched_condition_paper_assets(cns_matched_decision)
        )
    if args.write_brdt_assets:
        outputs["brdt_assets"] = write_brdt_assets()
    if args.write_brdt_context_figure:
        outputs["brdt_context_figure"] = str(write_brdt_context_figure())
    if args.write_cdi_extended_assets:
        outputs["cdi_extended_assets"] = write_cdi_extended_assets()
    if args.write_cdi_phase_zoom_figure:
        outputs["cdi_phase_zoom_figure"] = write_cdi_phase_zoom_figure()
    if args.write_cdi_phase_zoom_per_panel_figure:
        outputs["cdi_phase_zoom_per_panel_figure"] = write_cdi_phase_zoom_per_panel_figure()
    if args.write_model_config_table:
        outputs["model_config_table"] = write_model_config_table(REPO_ROOT, TABLES_DIR)
    if args.write_efficiency_table:
        outputs["paper_efficiency_table"] = write_paper_efficiency_table(REPO_ROOT, TABLES_DIR)
    if outputs:
        print(json.dumps(outputs, indent=2))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
