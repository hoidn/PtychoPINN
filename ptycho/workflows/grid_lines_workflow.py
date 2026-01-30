"""Grid-based lines workflow orchestration.

Skeleton module for running probe prep → grid simulation → training → inference →
metrics for the deprecated ptycho_lines workflow.

Data contracts: see specs/data_contracts.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
import json
import numpy as np
from skimage.restoration import unwrap_phase

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
    probe_scale_mode: str = "pad_extrapolate"
    set_phi: bool = False


# ---------------------------------------------------------------------------
# Probe Extraction + Upscaling Helpers (Task 2)
# ---------------------------------------------------------------------------


def load_probe_guess(npz_path: Path) -> np.ndarray:
    """Load probeGuess from NPZ file."""
    data = np.load(npz_path)
    if "probeGuess" not in data:
        raise KeyError("probeGuess missing from probe npz")
    return data["probeGuess"]


def scale_probe(
    probe: np.ndarray,
    target_N: int,
    smoothing_sigma: float,
    scale_mode: str = "pad_extrapolate",
) -> np.ndarray:
    """Resize probe to target_N and optionally smooth.

    Modes:
        - interpolate: cubic spline interpolation on real/imag parts.
        - pad_extrapolate: edge amplitude padding + quadratic phase extrapolation.
    """
    return scale_probe_with_mode(
        probe,
        target_N,
        smoothing_sigma,
        scale_mode=scale_mode,
    )


def _fit_quadratic_phase(phase: np.ndarray) -> tuple[float, float]:
    """Fit phase ~= a * r^2 + b to an unwrapped phase map."""
    h, w = phase.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.indices((h, w))
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    A = np.stack([r2.ravel(), np.ones(r2.size)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, phase.ravel(), rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _pad_amplitude(amplitude: np.ndarray, target_N: int) -> np.ndarray:
    """Pad amplitude to target_N using nearest-neighbor (edge) padding."""
    h, w = amplitude.shape
    if h != w:
        raise ValueError("probe must be square")
    if target_N < h:
        raise ValueError("pad_extrapolate requires target_N >= probe size")
    pad_total = target_N - h
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    pad_width = ((pad_before, pad_after), (pad_before, pad_after))
    return np.pad(amplitude, pad_width=pad_width, mode="edge")


def scale_probe_with_mode(
    probe: np.ndarray,
    target_N: int,
    smoothing_sigma: float,
    scale_mode: str = "pad_extrapolate",
) -> np.ndarray:
    """Resize probe to target_N and optionally smooth using specified mode.

    Modes:
        - interpolate: cubic spline interpolation on real/imag parts.
        - pad_extrapolate: edge-pad amplitude + quadratic phase extrapolation.
    """
    if probe.shape[0] != probe.shape[1]:
        raise ValueError("probe must be square")
    if scale_mode == "interpolate":
        if probe.shape[0] != target_N:
            zoom_factor = target_N / probe.shape[0]
            probe = interpolate_array(probe, zoom_factor)
    elif scale_mode == "pad_extrapolate":
        if target_N < probe.shape[0]:
            raise ValueError("pad_extrapolate requires target_N >= probe size")
        amplitude = np.abs(probe)
        phase = unwrap_phase(np.angle(probe))
        padded_amp = _pad_amplitude(amplitude, target_N)
        coeff_a, coeff_b = _fit_quadratic_phase(phase)
        cy = (target_N - 1) / 2.0
        cx = (target_N - 1) / 2.0
        yy, xx = np.indices((target_N, target_N))
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        extrap_phase = coeff_a * r2 + coeff_b
        extrap_phase = np.angle(np.exp(1j * extrap_phase))
        probe = (padded_amp * np.exp(1j * extrap_phase)).astype(np.complex64)
    else:
        raise ValueError(f"Unknown scale_mode '{scale_mode}'")

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
        model=ModelConfig(N=cfg.N, gridsize=cfg.gridsize, object_big=False),
        nphotons=cfg.nphotons,
        nepochs=cfg.nepochs,
        batch_size=cfg.batch_size,
        nll_weight=cfg.nll_weight,
        mae_weight=cfg.mae_weight,
        realspace_weight=cfg.realspace_weight,
    )
    update_legacy_dict(p.cfg, config)
    p.set("set_phi", cfg.set_phi)
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
    - Handles both gridsize=1 and gridsize>1
    - For gridsize>1, reshapes channels into spatial grid before stitching
    - Uses outer_offset_test from params.cfg for border clipping
    - Returns stitched array with last dimension = 1

    Args:
        predictions: Model output, shape (batch, N, N, gridsize^2) or complex
        norm_Y_I: Normalization factor from simulation
        part: 'amp', 'phase', or 'complex'

    Returns:
        Stitched images, shape (n_test, H, W, 1)
    """
    nimgs = p.get("nimgs_test")
    outer_offset = p.get("outer_offset_test")
    N = p.cfg["N"]
    gridsize = p.cfg["gridsize"]

    if part == "amp":
        getpart = np.absolute
    elif part == "phase":
        getpart = np.angle
    else:
        getpart = lambda x: x

    # Apply part extraction
    processed = getpart(predictions)

    # Handle gridsize>1: reshape channels to spatial grid
    # Input: (batch, N, N, gridsize^2)
    # Output: (batch*gridsize^2, N, N, 1) with patches reordered spatially
    if gridsize > 1 and len(processed.shape) == 4 and processed.shape[-1] == gridsize**2:
        batch = processed.shape[0]
        # Reshape to (batch, N, N, gridsize, gridsize)
        processed = processed.reshape(batch, N, N, gridsize, gridsize)
        # Transpose to (batch, gridsize, gridsize, N, N)
        processed = processed.transpose(0, 3, 4, 1, 2)
        # Reshape to (batch * gridsize * gridsize, N, N, 1)
        processed = processed.reshape(batch * gridsize**2, N, N, 1)
        # Update effective number of images for stitching calculation
        nimgs_effective = nimgs * gridsize**2
    else:
        # Ensure 4D with trailing 1
        if len(processed.shape) == 3:
            processed = processed[..., np.newaxis]
        nimgs_effective = nimgs

    # Calculate number of segments
    nsegments = int(np.sqrt((processed.size / nimgs_effective) / (N**2)))

    img_recon = np.reshape(
        norm_Y_I * processed, (-1, nsegments, nsegments, N, N, 1)
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

    Returns the reconstructed complex object (first output of model.predict),
    or None if inference fails due to XLA/dynamic shape issues.

    Known issue: PINN models compiled with XLA may fail during inference with
    dynamic batch sizes. See docs/bugs/XLA_INFERENCE_BUG.md for details.
    """
    import logging
    logger = logging.getLogger(__name__)

    intensity_scale = p.get("intensity_scale")
    try:
        reconstructed_obj, _, _ = model.predict([X_test * intensity_scale, coords_nominal])
        return reconstructed_obj
    except Exception as e:
        error_msg = str(e)
        if "xla" in error_msg.lower() or "fft" in error_msg.lower() or "dynamic" in error_msg.lower():
            logger.error(f"PINN inference failed with XLA/dynamic shape error: {e}")
            logger.warning("See docs/bugs/XLA_INFERENCE_BUG.md for details. Returning None.")
            return None
        raise  # Re-raise if not an XLA-related error


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


_LABEL_TITLES = {
    "pinn": "PINN",
    "baseline": "Baseline",
    "pinn_fno": "FNO",
    "pinn_hybrid": "Hybrid",
    "gt": "GT",
}


def _safe_min_max(array: np.ndarray) -> Tuple[float, float] | None:
    if array is None or array.size == 0:
        return None
    if not np.any(np.isfinite(array)):
        return None
    vmin = np.nanmin(array)
    vmax = np.nanmax(array)
    if np.isnan(vmin) or np.isnan(vmax):
        return None
    return float(vmin), float(vmax)


def _should_share_colorbar(
    arrays: Tuple[np.ndarray, ...] | list[np.ndarray],
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> bool:
    ranges = []
    for arr in arrays:
        bounds = _safe_min_max(arr)
        if bounds is None:
            return False
        ranges.append(bounds)
    if len(ranges) <= 1:
        return True
    first_min, first_max = ranges[0]
    for vmin, vmax in ranges[1:]:
        if not (np.isclose(vmin, first_min, rtol=rtol, atol=atol) and
                np.isclose(vmax, first_max, rtol=rtol, atol=atol)):
            return False
    return True


def _add_row_colorbars(fig, axes_row, mappables, arrays) -> None:
    share = _should_share_colorbar(arrays)
    if share:
        fig.colorbar(mappables[0], ax=list(axes_row), shrink=0.8)
    else:
        for ax, mappable in zip(axes_row, mappables):
            fig.colorbar(mappable, ax=ax, shrink=0.8)


def save_recon_artifact(output_dir: Path, label: str, recon_complex: np.ndarray) -> Path:
    """Save stitched complex reconstruction as NPZ artifact."""
    recon_dir = output_dir / "recons" / label
    recon_dir.mkdir(parents=True, exist_ok=True)
    recon = np.squeeze(recon_complex)
    if recon.ndim > 2:
        recon = recon[0]
    recon = recon.astype(np.complex64)
    amp = np.abs(recon)
    phase = np.angle(recon)
    path = recon_dir / "recon.npz"
    np.savez(path, YY_pred=recon, amp=amp, phase=phase)
    return path


def save_comparison_png_dynamic(
    output_dir: Path,
    gt_amp: np.ndarray,
    gt_phase: np.ndarray,
    recons: Dict[str, Dict[str, np.ndarray]],
    order: Tuple[str, ...],
) -> Path:
    """Save comparison plot with GT plus available model reconstructions."""
    import matplotlib.pyplot as plt

    labels = [label for label in order if label in recons]
    ncols = 1 + len(labels)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), squeeze=False)

    amp_arrays = [gt_amp]
    phase_arrays = [gt_phase]

    amp_mappables = [axes[0, 0].imshow(gt_amp, cmap="viridis")]
    axes[0, 0].set_title("GT Amplitude")
    axes[0, 0].axis("off")

    phase_mappables = [axes[1, 0].imshow(gt_phase, cmap="twilight")]
    axes[1, 0].set_title("GT Phase")
    axes[1, 0].axis("off")

    for idx, label in enumerate(labels, start=1):
        amp = recons[label]["amp"]
        phase = recons[label]["phase"]
        title = _LABEL_TITLES.get(label, label)

        amp_arrays.append(amp)
        phase_arrays.append(phase)

        amp_mappables.append(axes[0, idx].imshow(amp, cmap="viridis"))
        axes[0, idx].set_title(f"{title} Amplitude")
        axes[0, idx].axis("off")

        phase_mappables.append(axes[1, idx].imshow(phase, cmap="twilight"))
        axes[1, idx].set_title(f"{title} Phase")
        axes[1, idx].axis("off")

    _add_row_colorbars(fig, axes[0], amp_mappables, amp_arrays)
    _add_row_colorbars(fig, axes[1], phase_mappables, phase_arrays)

    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    out_path = visuals_dir / "compare_amp_phase.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_amp_phase_png(
    visuals_dir: Path,
    label: str,
    amp: np.ndarray,
    phase: np.ndarray,
) -> Path:
    """Save per-model amplitude/phase visualization."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), squeeze=False)
    title = _LABEL_TITLES.get(label, label)

    amp_mappable = axes[0, 0].imshow(amp, cmap="viridis")
    axes[0, 0].set_title(f"{title} Amplitude")
    axes[0, 0].axis("off")

    phase_mappable = axes[1, 0].imshow(phase, cmap="twilight")
    axes[1, 0].set_title(f"{title} Phase")
    axes[1, 0].axis("off")

    _add_row_colorbars(fig, axes[0], [amp_mappable], [amp])
    _add_row_colorbars(fig, axes[1], [phase_mappable], [phase])

    out_path = visuals_dir / f"amp_phase_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_grid_lines_visuals(output_dir: Path, order: Tuple[str, ...]) -> Dict[str, str]:
    """Render composite and per-model visuals from recon artifacts."""
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    recons: Dict[str, Dict[str, np.ndarray]] = {}
    per_model_paths: Dict[str, str] = {}
    for label in order:
        recon_path = output_dir / "recons" / label / "recon.npz"
        if not recon_path.exists():
            continue
        with np.load(recon_path) as data:
            if "amp" not in data or "phase" not in data:
                continue
            amp = data["amp"]
            phase = data["phase"]
        recons[label] = {"amp": amp, "phase": phase}
        per_model_paths[label] = str(save_amp_phase_png(visuals_dir, label, amp, phase))

    outputs: Dict[str, str] = {}
    for label, path in per_model_paths.items():
        outputs[f"amp_phase_{label}"] = path

    gt = recons.get("gt")
    if gt is None:
        return outputs

    compare = save_comparison_png_dynamic(
        output_dir,
        gt["amp"],
        gt["phase"],
        {label: data for label, data in recons.items() if label != "gt"},
        order=tuple(label for label in order if label != "gt"),
    )
    outputs["compare"] = str(compare)
    return outputs


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
    probe_scaled = scale_probe(
        probe_guess,
        cfg.N,
        cfg.probe_smoothing_sigma,
        scale_mode=cfg.probe_scale_mode,
    )

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
    if pinn_pred is None:
        print("[5/7] WARNING: PINN inference failed (XLA issue). Skipping PINN evaluation.")

    base_pred = run_baseline_inference(base_model, sim["test"]["X"])

    # Step 6: Stitch and evaluate
    print("[6/7] Stitching and computing metrics...")
    norm_Y_I = sim["test"]["norm_Y_I"]
    YY_gt = sim["test"]["YY_ground_truth"]

    # PINN stitching (may be None if inference failed)
    if pinn_pred is not None:
        pinn_amp = stitch_predictions(pinn_pred, norm_Y_I, part="amp")
        pinn_phase = stitch_predictions(pinn_pred, norm_Y_I, part="phase")
        pinn_stitched = pinn_amp * np.exp(1j * pinn_phase)
        pinn_metrics = eval_reconstruction(pinn_stitched, YY_gt, label="pinn")
    else:
        pinn_amp = pinn_phase = pinn_stitched = None
        pinn_metrics = {"error": "PINN inference failed (XLA issue)"}

    # Baseline stitching
    base_amp = stitch_predictions(base_pred, norm_Y_I, part="amp")
    base_phase = stitch_predictions(base_pred, norm_Y_I, part="phase")
    base_stitched = base_amp * np.exp(1j * base_phase)
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

    # Comparison PNG - squeeze any singleton dims from GT
    gt_squeezed = np.squeeze(YY_gt)
    gt_amp_2d = np.abs(gt_squeezed)
    gt_phase_2d = np.angle(gt_squeezed)
    base_amp_2d = base_amp[0, :, :, 0]
    base_phase_2d = base_phase[0, :, :, 0]

    save_recon_artifact(cfg.output_dir, "gt", gt_squeezed)
    save_recon_artifact(cfg.output_dir, "baseline", base_stitched)
    if pinn_pred is not None:
        save_recon_artifact(cfg.output_dir, "pinn", pinn_stitched)

    recons = {
        "baseline": {"amp": base_amp_2d, "phase": base_phase_2d},
    }
    if pinn_amp is not None:
        recons["pinn"] = {
            "amp": pinn_amp[0, :, :, 0],
            "phase": pinn_phase[0, :, :, 0],
        }

    png_path = save_comparison_png_dynamic(
        cfg.output_dir,
        gt_amp_2d,
        gt_phase_2d,
        recons,
        order=("pinn", "baseline"),
    )

    print(f"[grid_lines_workflow] Complete. Outputs in {cfg.output_dir}")

    return {
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "metrics_json": str(metrics_path),
        "comparison_png": str(png_path),
        "metrics": metrics_payload,
    }
