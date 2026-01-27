"""Grid-based lines workflow orchestration.

Skeleton module for running probe prep → grid simulation → training → inference →
metrics for the deprecated ptycho_lines workflow.

Data contracts: see specs/data_contracts.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np

from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p
from scripts.tools.prepare_data_tool import interpolate_array, smooth_complex_array


@dataclass
class GridLinesConfig:
    """Configuration for grid-based lines workflow.

    See docs/plans/2026-01-27-grid-lines-workflow.md for parameter details.
    """

    N: int
    gridsize: int
    output_dir: Path
    probe_npz: Path
    size: int = 392
    offset: int = 4
    outer_offset_train: int = 8
    outer_offset_test: int = 20
    nimgs_train: int = 2
    nimgs_test: int = 2
    nphotons: float = 1e9
    nepochs: int = 60
    batch_size: int = 16
    nll_weight: float = 0.0
    mae_weight: float = 1.0
    realspace_weight: float = 0.0
    probe_smoothing_sigma: float = 0.5


# ---------------------------------------------------------------------------
# Probe Extraction + Upscaling Helpers (Task 2)
# ---------------------------------------------------------------------------


def load_probe_guess(npz_path: Path) -> np.ndarray:
    """Load probeGuess from NPZ file."""
    data = np.load(npz_path)
    if "probeGuess" not in data:
        raise KeyError("probeGuess missing from probe npz")
    return data["probeGuess"]


def scale_probe(probe: np.ndarray, target_N: int, smoothing_sigma: float) -> np.ndarray:
    """Resize probe to target_N and optionally smooth.

    Uses interpolate_array for cubic spline interpolation on real/imag parts,
    then smooth_complex_array for Gaussian smoothing on amp/phase.
    """
    if probe.shape[0] != probe.shape[1]:
        raise ValueError("probe must be square")
    if probe.shape[0] != target_N:
        zoom_factor = target_N / probe.shape[0]
        probe = interpolate_array(probe, zoom_factor)
    if smoothing_sigma and smoothing_sigma > 0:
        probe = smooth_complex_array(probe, smoothing_sigma)
    return probe.astype(np.complex64)


# ---------------------------------------------------------------------------
# Simulation + Dataset Persistence (Task 3)
# ---------------------------------------------------------------------------


def configure_legacy_params(cfg: GridLinesConfig, probe_np: np.ndarray) -> TrainingConfig:
    """Configure legacy params.cfg and return a TrainingConfig.

    Must be called before generate_data() to set up legacy global state.
    """
    from ptycho import probe as probe_mod

    config = TrainingConfig(
        model=ModelConfig(N=cfg.N, gridsize=cfg.gridsize),
        nphotons=cfg.nphotons,
        nepochs=cfg.nepochs,
        batch_size=cfg.batch_size,
        nll_weight=cfg.nll_weight,
        mae_weight=cfg.mae_weight,
        realspace_weight=cfg.realspace_weight,
    )
    update_legacy_dict(p.cfg, config)
    p.set("data_source", "lines")
    p.set("size", cfg.size)
    p.set("offset", cfg.offset)
    p.set("outer_offset_train", cfg.outer_offset_train)
    p.set("outer_offset_test", cfg.outer_offset_test)
    p.set("nimgs_train", cfg.nimgs_train)
    p.set("nimgs_test", cfg.nimgs_test)
    p.set("nphotons", cfg.nphotons)
    p.set("sim_jitter_scale", 0.0)
    probe_mod.set_probe_guess(probe_guess=probe_np)
    return config


def simulate_grid_data(cfg: GridLinesConfig, probe_np: np.ndarray) -> Dict[str, Any]:
    """Run simulation via data_preprocessing.generate_data and return split data."""
    from ptycho import data_preprocessing

    configure_legacy_params(cfg, probe_np)
    (
        X_tr, YI_tr, Yphi_tr,
        X_te, YI_te, Yphi_te,
        YY_gt, dataset, YY_full, norm_Y_I
    ) = data_preprocessing.generate_data()

    return {
        "train": {
            "X": X_tr,
            "Y_I": YI_tr,
            "Y_phi": Yphi_tr,
            "coords_nominal": dataset.train_data.coords_nominal,
            "coords_true": dataset.train_data.coords_true,
            "YY_full": dataset.train_data.YY_full,
            "container": dataset.train_data,
        },
        "test": {
            "X": X_te,
            "Y_I": YI_te,
            "Y_phi": Yphi_te,
            "coords_nominal": dataset.test_data.coords_nominal,
            "coords_true": dataset.test_data.coords_true,
            "YY_full": dataset.test_data.YY_full,
            "YY_ground_truth": YY_gt,
            "norm_Y_I": norm_Y_I,
            "container": dataset.test_data,
        },
        "intensity_scale": p.get("intensity_scale"),
    }


def dataset_out_dir(cfg: GridLinesConfig) -> Path:
    """Return output directory for dataset NPZ files."""
    return cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"


def save_split_npz(
    cfg: GridLinesConfig,
    split: str,
    data: Dict[str, Any],
    config: TrainingConfig
) -> Path:
    """Save train or test split as NPZ with metadata."""
    from ptycho.metadata import MetadataManager

    out_dir = dataset_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.npz"

    payload = {
        "diffraction": data["X"],
        "Y_I": data["Y_I"],
        "Y_phi": data["Y_phi"],
        "coords_nominal": data["coords_nominal"],
        "coords_true": data["coords_true"],
        "YY_full": data["YY_full"],
    }
    if data.get("probeGuess") is not None:
        payload["probeGuess"] = data["probeGuess"]
    if split == "test":
        if data.get("YY_ground_truth") is not None:
            payload["YY_ground_truth"] = data["YY_ground_truth"]
        if data.get("norm_Y_I") is not None:
            payload["norm_Y_I"] = np.array(data["norm_Y_I"])

    metadata = MetadataManager.create_metadata(
        config,
        script_name="grid_lines_workflow",
        size=cfg.size,
        offset=cfg.offset,
        outer_offset_train=cfg.outer_offset_train,
        outer_offset_test=cfg.outer_offset_test,
        nimgs_train=cfg.nimgs_train,
        nimgs_test=cfg.nimgs_test,
    )
    MetadataManager.save_with_metadata(str(path), payload, metadata)
    return path


# ---------------------------------------------------------------------------
# Stitching Helper (Task 4) - gridsize=1 safe
# ---------------------------------------------------------------------------


def stitch_predictions(predictions: np.ndarray, norm_Y_I: float, part: str = "amp") -> np.ndarray:
    """Stitch model predictions, bypassing the incorrect gridsize=1 guard.

    NOTE: This function exists because data_preprocessing.stitch_data() has an
    incorrect ValueError guard for gridsize=1. The original stitching math works
    fine for gridsize=1 (produces 1x1 grid).

    Bug ref: STITCH-GRIDSIZE-001

    Contract: STITCH-001
    - Handles both gridsize=1 and gridsize=2
    - Uses outer_offset_test from params.cfg for border clipping
    - Returns stitched array with last dimension = 1

    Args:
        predictions: Model output, shape (n_test, N, N, gridsize^2) or complex
        norm_Y_I: Normalization factor from simulation
        part: 'amp', 'phase', or 'complex'

    Returns:
        Stitched images, shape (n_test, H, W, 1)
    """
    nimgs = p.get("nimgs_test")
    outer_offset = p.get("outer_offset_test")
    N = p.cfg["N"]

    nsegments = int(np.sqrt((predictions.size / nimgs) / (N**2)))

    if part == "amp":
        getpart = np.absolute
    elif part == "phase":
        getpart = np.angle
    else:
        getpart = lambda x: x

    img_recon = np.reshape(
        norm_Y_I * getpart(predictions), (-1, nsegments, nsegments, N, N, 1)
    )

    # Border clipping (from data_preprocessing.get_clip_sizes)
    bordersize = (N - outer_offset / 2) / 2
    borderleft = int(np.ceil(bordersize))
    borderright = int(np.floor(bordersize))

    img_recon = img_recon[:, :, :, borderleft:-borderright, borderleft:-borderright, :]
    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)
    return stitched


def run_grid_lines_workflow(cfg: GridLinesConfig) -> Dict[str, Any]:
    """Orchestrate probe prep → sim → train → infer → stitch → metrics."""
    raise NotImplementedError
