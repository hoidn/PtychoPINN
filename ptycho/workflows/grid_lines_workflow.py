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


# ---------------------------------------------------------------------------
# Training + Inference Helpers (Task 5)
# ---------------------------------------------------------------------------


def train_pinn_model(train_data):
    """Train PtychoPINN model and return model + history."""
    from ptycho import train_pinn

    model, history = train_pinn.train(train_data)
    return model, history


def save_pinn_model(cfg: GridLinesConfig) -> None:
    """Save trained PINN model to output directory."""
    from ptycho import model_manager

    out_dir = cfg.output_dir / "pinn"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_manager.save(str(out_dir))


def select_baseline_channels(X, Y_I, Y_phi):
    """Select channel 0 only for baseline when gridsize > 1."""
    if X.shape[-1] > 1:
        return X[..., :1], Y_I[..., :1], Y_phi[..., :1]
    return X, Y_I, Y_phi


def train_baseline_model(X_train, Y_I_train, Y_phi_train):
    """Train baseline model (channel 0 only for gridsize > 1)."""
    from ptycho import baselines

    Xb, YIb, Yphib = select_baseline_channels(X_train, Y_I_train, Y_phi_train)
    model, history = baselines.train(Xb, YIb, Yphib)
    return model, history


def run_pinn_inference(model, X_test, coords_nominal):
    """Run PINN inference on test data.

    Returns the reconstructed complex object (first output of model.predict).
    """
    intensity_scale = p.get("intensity_scale")
    reconstructed_obj, _, _ = model.predict([X_test * intensity_scale, coords_nominal])
    return reconstructed_obj


def run_baseline_inference(model, X_test):
    """Run baseline inference on test data (channel 0 only)."""
    Xb, _, _ = select_baseline_channels(X_test, X_test, X_test)
    pred_amp, pred_phase = model.predict(Xb)
    pred_complex = pred_amp * np.exp(1j * pred_phase)
    return pred_complex


# ---------------------------------------------------------------------------
# Orchestrator + Outputs (Task 6)
# ---------------------------------------------------------------------------


def save_comparison_png(
    cfg: GridLinesConfig,
    gt_amp: np.ndarray,
    gt_phase: np.ndarray,
    pinn_amp: np.ndarray,
    pinn_phase: np.ndarray,
    base_amp: np.ndarray,
    base_phase: np.ndarray,
) -> Path:
    """Save 2x3 comparison plot (amp/phase rows x GT/PINN/Baseline cols)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 0: Amplitude
    axes[0, 0].imshow(gt_amp, cmap="viridis")
    axes[0, 0].set_title("GT Amplitude")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pinn_amp, cmap="viridis")
    axes[0, 1].set_title("PINN Amplitude")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(base_amp, cmap="viridis")
    axes[0, 2].set_title("Baseline Amplitude")
    axes[0, 2].axis("off")

    # Row 1: Phase
    axes[1, 0].imshow(gt_phase, cmap="twilight")
    axes[1, 0].set_title("GT Phase")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pinn_phase, cmap="twilight")
    axes[1, 1].set_title("PINN Phase")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(base_phase, cmap="twilight")
    axes[1, 2].set_title("Baseline Phase")
    axes[1, 2].axis("off")

    fig.suptitle(f"N={cfg.N}, gridsize={cfg.gridsize}", fontsize=14)
    plt.tight_layout()

    visuals_dir = cfg.output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    out_path = visuals_dir / "compare_amp_phase.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_grid_lines_workflow(cfg: GridLinesConfig) -> Dict[str, Any]:
    """Orchestrate probe prep → sim → train → infer → stitch → metrics.

    Steps:
    1. Load and scale probe to target N
    2. Configure legacy params and simulate grid data
    3. Save train/test datasets as NPZ
    4. Train PINN and baseline models
    5. Run inference on test data
    6. Stitch predictions and compute metrics
    7. Save comparison PNG and metrics JSON

    Returns:
        Dict with train_npz, test_npz, metrics paths and values.
    """
    from ptycho.evaluation import eval_reconstruction

    print(f"[grid_lines_workflow] Starting N={cfg.N}, gridsize={cfg.gridsize}")

    # Step 1: Probe preparation
    print("[1/7] Loading and scaling probe...")
    probe_guess = load_probe_guess(cfg.probe_npz)
    probe_scaled = scale_probe(probe_guess, cfg.N, cfg.probe_smoothing_sigma)

    # Step 2: Simulation
    print("[2/7] Running grid simulation...")
    sim = simulate_grid_data(cfg, probe_scaled)
    config = configure_legacy_params(cfg, probe_scaled)

    # Step 3: Save datasets
    print("[3/7] Saving datasets...")
    sim["train"]["probeGuess"] = probe_scaled
    sim["test"]["probeGuess"] = probe_scaled
    train_npz = save_split_npz(cfg, "train", sim["train"], config)
    test_npz = save_split_npz(cfg, "test", sim["test"], config)

    # Step 4: Training
    print("[4/7] Training PINN model...")
    pinn_model, _ = train_pinn_model(sim["train"]["container"])
    save_pinn_model(cfg)

    print("[4/7] Training Baseline model...")
    base_model, _ = train_baseline_model(
        sim["train"]["X"], sim["train"]["Y_I"], sim["train"]["Y_phi"]
    )
    base_dir = cfg.output_dir / "baseline"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_model.save(base_dir / "baseline.keras")

    # Step 5: Inference
    print("[5/7] Running inference...")
    pinn_pred = run_pinn_inference(
        pinn_model, sim["test"]["X"], sim["test"]["coords_nominal"]
    )
    base_pred = run_baseline_inference(base_model, sim["test"]["X"])

    # Step 6: Stitch and evaluate
    print("[6/7] Stitching and computing metrics...")
    norm_Y_I = sim["test"]["norm_Y_I"]
    YY_gt = sim["test"]["YY_ground_truth"]

    pinn_amp = stitch_predictions(pinn_pred, norm_Y_I, part="amp")
    pinn_phase = stitch_predictions(pinn_pred, norm_Y_I, part="phase")
    pinn_stitched = pinn_amp * np.exp(1j * pinn_phase)

    base_amp = stitch_predictions(base_pred, norm_Y_I, part="amp")
    base_phase = stitch_predictions(base_pred, norm_Y_I, part="phase")
    base_stitched = base_amp * np.exp(1j * base_phase)

    pinn_metrics = eval_reconstruction(pinn_stitched, YY_gt, label="pinn")
    base_metrics = eval_reconstruction(base_stitched, YY_gt, label="baseline")

    # Step 7: Save outputs
    print("[7/7] Saving outputs...")

    # Metrics JSON
    metrics_payload = {
        "pinn": pinn_metrics,
        "baseline": base_metrics,
    }
    metrics_path = cfg.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str))

    # Comparison PNG
    gt_amp_2d = np.abs(YY_gt[0, :, :]) if YY_gt.ndim >= 3 else np.abs(YY_gt)
    gt_phase_2d = np.angle(YY_gt[0, :, :]) if YY_gt.ndim >= 3 else np.angle(YY_gt)
    pinn_amp_2d = pinn_amp[0, :, :, 0]
    pinn_phase_2d = pinn_phase[0, :, :, 0]
    base_amp_2d = base_amp[0, :, :, 0]
    base_phase_2d = base_phase[0, :, :, 0]

    png_path = save_comparison_png(
        cfg,
        gt_amp_2d, gt_phase_2d,
        pinn_amp_2d, pinn_phase_2d,
        base_amp_2d, base_phase_2d,
    )

    print(f"[grid_lines_workflow] Complete. Outputs in {cfg.output_dir}")

    return {
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "metrics_json": str(metrics_path),
        "comparison_png": str(png_path),
        "metrics": metrics_payload,
    }
