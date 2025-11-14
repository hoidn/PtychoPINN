#!/usr/bin/env python3
"""
compare_models.py - Load trained PtychoPINN and baseline models, run inference,
calculate metrics, and generate comparison visualizations.
"""

import argparse
import os
import sys
import time
import tempfile
import zipfile
from pathlib import Path
import dill
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ptycho components
from ptycho.workflows.components import load_data, create_ptycho_data_container, logger, load_inference_bundle
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p
from ptycho.tf_helper import reassemble_position, _channel_to_flat
from ptycho.evaluation import eval_reconstruction
from ptycho.image.cropping import align_for_evaluation
from ptycho.image.registration import find_translation_offset, apply_shift_and_crop, register_and_align
from ptycho.cli_args import add_logging_arguments, get_logging_config
from ptycho.log_config import setup_logging

# NOTE: nbutils import is delayed until after models are loaded to prevent KeyError


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare PtychoPINN and Baseline models.")
    parser.add_argument("--pinn_dir", type=Path, required=True,
                        help="Directory of the trained PtychoPINN model.")
    parser.add_argument("--baseline_dir", type=Path, required=True,
                        help="Directory of the trained baseline model.")
    parser.add_argument("--test_data", type=Path, required=True,
                        help="Path to the test data NPZ file.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save comparison results.")
    parser.add_argument("--p_min", type=float, default=10.0,
                        help="Lower percentile for color scale (default: 10.0).")
    parser.add_argument("--p_max", type=float, default=90.0,
                        help="Upper percentile for color scale (default: 90.0).")
    parser.add_argument("--pinn_phase_vmin", type=float, default=None,
                        help="PtychoPINN phase vmin (default: auto from percentiles).")
    parser.add_argument("--pinn_phase_vmax", type=float, default=None,
                        help="PtychoPINN phase vmax (default: auto from percentiles).")
    parser.add_argument("--baseline_phase_vmin", type=float, default=None,
                        help="Baseline phase vmin (default: auto from percentiles).")
    parser.add_argument("--baseline_phase_vmax", type=float, default=None,
                        help="Baseline phase vmax (default: auto from percentiles).")
    parser.add_argument("--skip-registration", action="store_true",
                        help="Skip automatic registration before evaluation (for debugging).")
    parser.add_argument("--register-ptychi-only", action="store_true",
                        help="Apply registration ONLY to pty-chi/tike, not to PtychoPINN or Baseline.")
    parser.add_argument("--save-npz", action="store_true", default=True,
                        help="Save NPZ files containing amplitude and phase data for all reconstructions and ground truth (default: enabled).")
    parser.add_argument("--no-save-npz", action="store_true",
                        help="Disable NPZ file export to save disk space.")
    parser.add_argument("--save-npz-aligned", action="store_true", default=True,
                        help="Save post-registration aligned NPZ files (default: enabled).")
    parser.add_argument("--no-save-npz-aligned", action="store_true",
                        help="Disable aligned NPZ file export to save disk space.")
    parser.add_argument("--phase-align-method", choices=['plane', 'mean'], default='plane',
                        help="Method for phase alignment: 'plane' (fit and remove planes, default) or 'mean' (subtract mean).")
    parser.add_argument("--frc-sigma", type=float, default=0.0,
                        help="Gaussian smoothing sigma for FRC calculation (0 = no smoothing, default: 0.0).")
    parser.add_argument("--save-debug-images", action="store_true",
                        help="Save debug images for MS-SSIM and FRC preprocessing visualization.")
    parser.add_argument("--ms-ssim-sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma for MS-SSIM amplitude calculation (default: 1.0).")
    parser.add_argument("--n-test-groups", type=int, default=None,
                        help="Number of test groups to load from the file (default: all).")
    parser.add_argument("--n-test-subsample", type=int, default=None,
                        help="Number of images to subsample from test data for evaluation.")
    parser.add_argument("--test-subsample-seed", type=int, default=None,
                        help="Random seed for reproducible test subsampling")
    # Legacy compatibility
    parser.add_argument("--n-test-images", type=int, default=None,
                        help="DEPRECATED: Use --n-test-groups instead.")
    parser.add_argument("--n-subsample", type=int, default=None,
                        help="DEPRECATED: Use --n-test-subsample instead.")
    parser.add_argument("--subsample-seed", type=int, default=None,
                        help="DEPRECATED: Use --test-subsample-seed instead.")
    parser.add_argument("--tike_recon_path", type=Path, default=None,
                        help="Path to Tike reconstruction NPZ file for three-way comparison (optional).")
    parser.add_argument("--stitch-crop-size", type=int, default=20,
                        help="Crop size M for patch stitching (must be 0 < M <= N, default: 20).")
    parser.add_argument("--baseline-debug-limit", type=int, default=None,
                        help="Limit baseline inference to first N groups for debugging (default: all groups).")
    parser.add_argument("--baseline-debug-dir", type=Path, default=None,
                        help="Directory to save baseline debug artifacts (NPZ with inputs/outputs/offsets, JSON with stats).")
    parser.add_argument("--baseline-chunk-size", type=int, default=None,
                        help="Process baseline inference in chunks of N groups to reduce GPU memory (default: None = all at once).")
    parser.add_argument("--baseline-predict-batch-size", type=int, default=32,
                        help="Batch size for baseline model.predict() within each chunk (default: 32).")

    # Add logging arguments
    add_logging_arguments(parser)
    
    return parser.parse_args()


def load_bundle_submodel(model_dir: Path, model_name: str) -> tf.keras.Model:
    """Load a specific submodel (autoencoder/diffraction_to_obj) from a wts.h5.zip archive."""
    model_dir = Path(model_dir)
    archive = model_dir / "wts.h5.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Model archive not found at: {archive}")

    logger.info(f"Extracting {model_name} from bundle {archive}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(tmp_dir)

        subdir = Path(tmp_dir) / model_name
        model_path = subdir / "model.keras"
        params_path = subdir / "params.dill"
        custom_objects_path = subdir / "custom_objects.dill"

        if not model_path.exists():
            raise FileNotFoundError(f"{model_name}/model.keras not found inside {archive}")

        with open(params_path, "rb") as fh:
            loaded_params = dill.load(fh)
        loaded_params.pop("_version", None)
        p.cfg.update(loaded_params)

        with open(custom_objects_path, "rb") as fh:
            custom_objects = dill.load(fh)

        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    logger.info(f"Loaded {model_name} model from {model_dir}")
    return model


def load_pinn_model(model_dir: Path) -> tf.keras.Model:
    """Load the diffraction_to_obj inference model for PINN comparisons."""
    model, _ = load_inference_bundle(model_dir)
    return model


def load_baseline_model(baseline_dir: Path) -> tf.keras.Model:
    """Load the baseline model.

    Priority:
    1) legacy Keras file baseline_model.h5 within baseline_dir
    2) TF bundle via load_inference_bundle(baseline_dir) (expects wts.h5.zip)
    """
    logger.info(f"Loading Baseline model from {baseline_dir}...")

    # Try legacy Keras path first
    baseline_model_path = None
    for root, dirs, files in os.walk(baseline_dir):
        for file in files:
            if file == "baseline_model.h5":
                baseline_model_path = Path(root) / file
                break
        if baseline_model_path:
            break

    if baseline_model_path and baseline_model_path.exists():
        logger.info(f"Found baseline model at: {baseline_model_path}")
        return tf.keras.models.load_model(baseline_model_path)

    # Fallback: treat baseline_dir as a TF bundle directory (wts.h5.zip)
    logger.info("baseline_model.h5 not found; attempting to load TF bundle (wts.h5.zip)")
    baseline_model, _ = load_inference_bundle(baseline_dir)
    return baseline_model


def tensor_to_numpy(value, dtype=None):
    """Convert TensorFlow tensors or arrays to NumPy with optional dtype casting."""
    if tf.is_tensor(value):
        array = value.numpy()
    else:
        array = np.asarray(value)
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype, copy=False)
    return array


def _compute_channel_offsets(container) -> tf.Tensor:
    """Return combined offsets (global + local) in channel format."""
    total_channels = int(container.X.shape[-1])
    global_offsets = tf.convert_to_tensor(container.global_offsets, dtype=tf.float32)
    if total_channels == 1:
        return global_offsets
    local_offsets = getattr(container, "local_offsets", None)
    if local_offsets is None:
        return tf.tile(global_offsets, [1, 1, 1, total_channels])
    local_offsets = tf.convert_to_tensor(local_offsets, dtype=tf.float32)
    if local_offsets.shape[-1] != total_channels:
        divisor = int(local_offsets.shape[-1])
        if divisor == 0 or total_channels % divisor != 0:
            raise ValueError(
                f"Incompatible local offset channels (got {divisor}) for total_channels={total_channels}"
            )
        repeats = total_channels // divisor
        local_offsets = tf.tile(local_offsets, [1, 1, 1, repeats])
    return global_offsets + local_offsets


def prepare_baseline_inference_data(container):
    """Flatten grouped diffraction + offsets for baseline stitching when needed.

    For grouped runs (total_channels > 1), validates that channel count is a perfect
    square (gridsize²), computes and logs the resolved gridsize, then flattens both
    diffraction and offset tensors for baseline model inference.

    Returns:
        tuple: (baseline_input, baseline_offsets) - both as numpy arrays ready for
               baseline_model.predict([baseline_input, baseline_offsets], ...)
    """
    total_channels = int(container.X.shape[-1])
    if total_channels > 1:
        # Assert perfect square channel count for grouped runs
        import math
        sqrt_channels = math.sqrt(total_channels)
        if not sqrt_channels.is_integer():
            raise ValueError(
                f"Grouped diffraction channel count ({total_channels}) must be a perfect square "
                f"(gridsize²). Got non-integer sqrt: {sqrt_channels}"
            )

        resolved_gridsize = int(sqrt_channels)
        logger.info(
            "Flattening grouped diffraction for baseline model: X %s → channels merged; "
            "resolved gridsize=%d (from %d channels)",
            container.X.shape,
            resolved_gridsize,
            total_channels,
        )

        # Force params.cfg gridsize sync for downstream Translation/reassembly
        from ptycho import params as p
        p.set('gridsize', resolved_gridsize)
        logger.info(f"Forced params.cfg['gridsize']={resolved_gridsize} for baseline grouped inference")

        flattened_input = _channel_to_flat(container.X)
        offsets_channel = _compute_channel_offsets(container)
        flattened_offsets = _channel_to_flat(offsets_channel)

        # Center offsets to zero-mean for baseline model stability
        flattened_offsets_np = tensor_to_numpy(flattened_offsets, dtype=np.float64)
        offset_mean = flattened_offsets_np.mean()
        centered_offsets = flattened_offsets_np - offset_mean

        logger.info(
            "Centered baseline offsets: original mean=%.2f, std=%.2f → centered mean=%.6f, std=%.2f",
            offset_mean,
            flattened_offsets_np.std(),
            centered_offsets.mean(),
            centered_offsets.std()
        )

        return (
            tensor_to_numpy(flattened_input, dtype=np.float32),
            centered_offsets,
        )
    return (
        tensor_to_numpy(container.X, dtype=np.float32),
        tensor_to_numpy(container.global_offsets, dtype=np.float64),
    )


def load_tike_reconstruction(tike_path: Path) -> tuple:
    """Load iterative reconstruction from standardized NPZ format (Tike or Pty-chi).
    
    Args:
        tike_path: Path to reconstruction NPZ file (tike_reconstruction.npz or ptychi_reconstruction.npz)
        
    Returns:
        tuple: (reconstructed_object, computation_time, algorithm_name)
            - reconstructed_object: Complex array containing the reconstructed object
            - computation_time: Computation time in seconds (None if not available)
            - algorithm_name: Name of algorithm used ("Tike" or "Pty-chi (algorithm)")
        
    Raises:
        KeyError: If required keys are missing from NPZ file
        ValueError: If data format is invalid
    """
    logger.info(f"Loading iterative reconstruction from {tike_path}...")
    
    with np.load(tike_path, allow_pickle=True) as data:
        # Check for required key
        if 'reconstructed_object' not in data:
            available_keys = list(data.keys())
            raise KeyError(f"Missing 'reconstructed_object' key in reconstruction NPZ file: {tike_path}. "
                          f"Available keys: {available_keys}. Expected format generated by 'run_tike_reconstruction.py' or 'run_ptychi_reconstruction.py'.")
        
        reconstructed_object = data['reconstructed_object']
        
        # Validate data format
        if not np.iscomplexobj(reconstructed_object):
            raise ValueError(f"Reconstruction must be complex-valued, got {reconstructed_object.dtype}. "
                           f"Expected complex64 or complex128 from ptychographic reconstruction.")
        
        if reconstructed_object.ndim != 2:
            raise ValueError(f"Reconstruction must be 2D, got shape {reconstructed_object.shape}. "
                           f"Expected (height, width) array representing the reconstructed object.")
        
        # Extract computation time and algorithm info from metadata if available
        computation_time = None
        algorithm_name = "Tike"  # Default to Tike for backward compatibility
        
        # First check for direct 'algorithm' field (used by pty-chi scripts)
        if 'algorithm' in data:
            try:
                algorithm_str = str(data['algorithm'].item() if hasattr(data['algorithm'], 'item') else data['algorithm'])
                # Map algorithm names to canonical IDs for metrics reporting (METRICS-NAMING-001)
                # Use "PtyChi" as canonical ID, not "Pty-chi (algorithm)"
                if algorithm_str.lower() in ['epie', 'pie', 'dm', 'ml'] or algorithm_str.lower().startswith('ptychi'):
                    algorithm_name = "PtyChi"
                    logger.debug(f"Detected pty-chi algorithm '{algorithm_str}', using canonical ID: PtyChi")
                elif algorithm_str.lower() == 'tike':
                    algorithm_name = "Tike"
                else:
                    # Unknown algorithm, default to PtyChi for pty-chi namespace
                    algorithm_name = "PtyChi"
                    logger.debug(f"Unknown algorithm '{algorithm_str}', defaulting to canonical ID: PtyChi")
                logger.debug(f"Detected algorithm from 'algorithm' field: {algorithm_name}")
            except Exception as e:
                logger.debug(f"Could not extract algorithm field: {e}")
        
        # Also check for metadata field (alternative format)
        if 'metadata' in data:
            try:
                metadata = data['metadata'].item()  # Extract from numpy array
                if computation_time is None:
                    computation_time = metadata.get('computation_time_seconds', None)
                
                # Detect algorithm type from metadata (overrides direct algorithm field if present)
                if 'algorithm' in metadata:
                    algorithm = metadata.get('algorithm', 'tike')
                    # Use canonical IDs for metrics reporting (METRICS-NAMING-001)
                    if algorithm.startswith('ptychi') or 'ptychi' in algorithm.lower():
                        algorithm_name = "PtyChi"
                    elif algorithm == 'tike':
                        algorithm_name = "Tike"
                    else:
                        # Default to PtyChi for unknown algorithms in pty-chi namespace
                        algorithm_name = "PtyChi"
                
                if computation_time is not None:
                    logger.debug(f"Extracted {algorithm_name} computation time: {computation_time:.2f}s")
            except Exception as e:
                logger.warning(f"Could not extract metadata: {e}")
        
        # Extract reconstruction_time if computation_time not found
        if computation_time is None and 'reconstruction_time' in data:
            try:
                computation_time = float(data['reconstruction_time'].item() if hasattr(data['reconstruction_time'], 'item') else data['reconstruction_time'])
            except Exception as e:
                logger.debug(f"Could not extract reconstruction_time: {e}")
        
        logger.info(f"Loaded {algorithm_name} reconstruction: {reconstructed_object.shape} ({reconstructed_object.dtype})")
        
        return reconstructed_object, computation_time, algorithm_name


def create_comparison_plot(pinn_obj, baseline_obj, ground_truth_obj, output_path, 
                          p_min=10.0, p_max=90.0, 
                          pinn_phase_vmin=None, pinn_phase_vmax=None,
                          baseline_phase_vmin=None, baseline_phase_vmax=None,
                          pinn_offset=None, baseline_offset=None,
                          tike_obj=None, tike_phase_vmin=None, tike_phase_vmax=None, tike_offset=None,
                          algorithm_name=None):
    """Create a 2x3 or 2x4 subplot comparing reconstructions.
    
    Generates a dynamic comparison plot: 2x3 for two-way comparison (PtychoPINN vs. Baseline)
    or 2x4 for three-way comparison (PtychoPINN vs. Baseline vs. iterative reconstruction) when iterative data is provided.
    
    Args:
        pinn_obj: PtychoPINN reconstruction
        baseline_obj: Baseline reconstruction  
        ground_truth_obj: Ground truth object (optional)
        output_path: Path to save the plot
        p_min: Lower percentile for color scale (default: 10.0)
        p_max: Upper percentile for color scale (default: 90.0)
        pinn_phase_vmin: PtychoPINN phase vmin (default: auto from percentiles)
        pinn_phase_vmax: PtychoPINN phase vmax (default: auto from percentiles)
        baseline_phase_vmin: Baseline phase vmin (default: auto from percentiles)
        baseline_phase_vmax: Baseline phase vmax (default: auto from percentiles)
        pinn_offset: Translation offset detected for PtychoPINN (dy, dx)
        baseline_offset: Translation offset detected for Baseline (dy, dx)
        tike_obj: Iterative reconstruction (optional, triggers 2x4 plot)
        tike_phase_vmin: Iterative reconstruction phase vmin (default: auto from percentiles)
        tike_phase_vmax: Iterative reconstruction phase vmax (default: auto from percentiles)
        tike_offset: Translation offset detected for iterative reconstruction (dy, dx)
        algorithm_name: Name of the iterative algorithm (e.g., "Pty-chi (ePIE)", "Tike")
    """
    # Dynamic subplot grid: 2x3 for two-way, 2x4 for three-way comparison
    if tike_obj is not None:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
        iterative_label = algorithm_name if algorithm_name else "Iterative"
        fig.suptitle(f"PtychoPINN vs. Baseline vs. {iterative_label} Reconstruction", fontsize=16)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
        fig.suptitle("PtychoPINN vs. Baseline Reconstruction", fontsize=16)

    # Set titles with offset information
    pinn_title = "PtychoPINN"
    if pinn_offset is not None:
        pinn_title += f"\n(offset: ({pinn_offset[0]:.2f}, {pinn_offset[1]:.2f}))"
    axes[0, 0].set_title(pinn_title)
    
    baseline_title = "Baseline"
    if baseline_offset is not None:
        baseline_title += f"\n(offset: ({baseline_offset[0]:.2f}, {baseline_offset[1]:.2f}))"
    axes[0, 1].set_title(baseline_title)
    
    # Add iterative reconstruction title if present
    if tike_obj is not None:
        tike_title = algorithm_name if algorithm_name else "Iterative"
        if tike_offset is not None:
            tike_title += f"\n(offset: ({tike_offset[0]:.2f}, {tike_offset[1]:.2f}))"
        axes[0, 2].set_title(tike_title)
        axes[0, 3].set_title("Ground Truth")  # Ground Truth moves to column 3
    else:
        axes[0, 2].set_title("Ground Truth")  # Ground Truth stays in column 2
    
    # Set row labels
    axes[0, 0].set_ylabel("Phase", fontsize=14)
    axes[1, 0].set_ylabel("Amplitude", fontsize=14)

    # --- Percentile-based color scaling ---
    
    # Calculate per-panel amplitude limits
    pinn_amps = np.abs(pinn_obj).ravel()
    pinn_v_amp_min, pinn_v_amp_max = np.percentile(pinn_amps, [p_min, p_max])
    logger.info(f"PtychoPINN amplitude color scale (vmin, vmax) set to: ({pinn_v_amp_min:.3f}, {pinn_v_amp_max:.3f}) using {p_min}/{p_max} percentiles [per-panel].")
    
    baseline_amps = np.abs(baseline_obj).ravel()
    baseline_v_amp_min, baseline_v_amp_max = np.percentile(baseline_amps, [p_min, p_max])
    logger.info(f"Baseline amplitude color scale (vmin, vmax) set to: ({baseline_v_amp_min:.3f}, {baseline_v_amp_max:.3f}) using {p_min}/{p_max} percentiles [per-panel].")
    
    # Determine phase limits for PtychoPINN
    if pinn_phase_vmin is not None and pinn_phase_vmax is not None:
        pinn_v_phase_min, pinn_v_phase_max = pinn_phase_vmin, pinn_phase_vmax
        logger.info(f"PtychoPINN phase color scale (vmin, vmax) set to: ({pinn_v_phase_min:.3f}, {pinn_v_phase_max:.3f}) [manual].")
    else:
        pinn_phases = np.angle(pinn_obj).ravel()
        pinn_v_phase_min, pinn_v_phase_max = np.percentile(pinn_phases, [p_min, p_max])
        logger.info(f"PtychoPINN phase color scale (vmin, vmax) set to: ({pinn_v_phase_min:.3f}, {pinn_v_phase_max:.3f}) using {p_min}/{p_max} percentiles [per-panel].")
    
    # Determine phase limits for Baseline
    if baseline_phase_vmin is not None and baseline_phase_vmax is not None:
        baseline_v_phase_min, baseline_v_phase_max = baseline_phase_vmin, baseline_phase_vmax
        logger.info(f"Baseline phase color scale (vmin, vmax) set to: ({baseline_v_phase_min:.3f}, {baseline_v_phase_max:.3f}) [manual].")
    else:
        baseline_phases = np.angle(baseline_obj).ravel()
        baseline_v_phase_min, baseline_v_phase_max = np.percentile(baseline_phases, [p_min, p_max])
        logger.info(f"Baseline phase color scale (vmin, vmax) set to: ({baseline_v_phase_min:.3f}, {baseline_v_phase_max:.3f}) using {p_min}/{p_max} percentiles [per-panel].")

    # Determine color limits for iterative reconstruction if present
    if tike_obj is not None:
        # Calculate iterative reconstruction amplitude limits
        tike_amps = np.abs(tike_obj).ravel()
        tike_v_amp_min, tike_v_amp_max = np.percentile(tike_amps, [p_min, p_max])
        iterative_label = algorithm_name if algorithm_name else "Iterative"
        logger.info(f"{iterative_label} amplitude color scale (vmin, vmax) set to: ({tike_v_amp_min:.3f}, {tike_v_amp_max:.3f}) using {p_min}/{p_max} percentiles [per-panel].")
        
        # Determine phase limits for iterative reconstruction
        if tike_phase_vmin is not None and tike_phase_vmax is not None:
            tike_v_phase_min, tike_v_phase_max = tike_phase_vmin, tike_phase_vmax
            iterative_label = algorithm_name if algorithm_name else "Iterative"
            logger.info(f"{iterative_label} phase color scale (vmin, vmax) set to: ({tike_v_phase_min:.3f}, {tike_v_phase_max:.3f}) [manual].")
        else:
            tike_phases = np.angle(tike_obj).ravel()
            tike_v_phase_min, tike_v_phase_max = np.percentile(tike_phases, [p_min, p_max])
            iterative_label = algorithm_name if algorithm_name else "Iterative"
            logger.info(f"{iterative_label} phase color scale (vmin, vmax) set to: ({tike_v_phase_min:.3f}, {tike_v_phase_max:.3f}) using {p_min}/{p_max} percentiles [per-panel].")

    # Plot PtychoPINN
    im1 = axes[0, 0].imshow(np.angle(pinn_obj), vmin=pinn_v_phase_min, vmax=pinn_v_phase_max)
    im2 = axes[1, 0].imshow(np.abs(pinn_obj), cmap='gray', vmin=pinn_v_amp_min, vmax=pinn_v_amp_max)
    
    # Plot Baseline
    im3 = axes[0, 1].imshow(np.angle(baseline_obj), vmin=baseline_v_phase_min, vmax=baseline_v_phase_max)
    im4 = axes[1, 1].imshow(np.abs(baseline_obj), cmap='gray', vmin=baseline_v_amp_min, vmax=baseline_v_amp_max)
    
    # Plot iterative reconstruction if present
    if tike_obj is not None:
        im5 = axes[0, 2].imshow(np.angle(tike_obj), vmin=tike_v_phase_min, vmax=tike_v_phase_max)
        im6 = axes[1, 2].imshow(np.abs(tike_obj), cmap='gray', vmin=tike_v_amp_min, vmax=tike_v_amp_max)
        gt_col = 3  # Ground truth is in column 3 when iterative reconstruction is present
    else:
        gt_col = 2  # Ground truth is in column 2 when no iterative reconstruction
    
    # Plot Ground Truth (use its own phase and amplitude scales when auto-scaling)
    if ground_truth_obj is not None:
        # Remove extra dimensions if present
        gt_obj = ground_truth_obj.squeeze()
        
        # Calculate ground truth amplitude limits
        gt_amps = np.abs(gt_obj).ravel()
        gt_v_amp_min, gt_v_amp_max = np.percentile(gt_amps, [p_min, p_max])
        logger.info(f"Ground Truth amplitude color scale (vmin, vmax) set to: ({gt_v_amp_min:.3f}, {gt_v_amp_max:.3f}) using {p_min}/{p_max} percentiles [per-panel].")
        
        # Calculate ground truth phase limits
        gt_phases = np.angle(gt_obj).ravel()
        gt_v_phase_min, gt_v_phase_max = np.percentile(gt_phases, [p_min, p_max])
        logger.info(f"Ground Truth phase color scale (vmin, vmax) set to: ({gt_v_phase_min:.3f}, {gt_v_phase_max:.3f}) using {p_min}/{p_max} percentiles [per-panel].")
        
        im_gt_phase = axes[0, gt_col].imshow(np.angle(gt_obj), vmin=gt_v_phase_min, vmax=gt_v_phase_max)
        im_gt_amp = axes[1, gt_col].imshow(np.abs(gt_obj), cmap='gray', vmin=gt_v_amp_min, vmax=gt_v_amp_max)
    else:
        for ax in [axes[0, gt_col], axes[1, gt_col]]:
            ax.text(0.5, 0.5, "No Ground Truth", ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_facecolor('lightgray')

    # Remove ticks for cleaner appearance
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add individual colorbars for phase plots (top row)
    cbar_pinn_phase = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar_pinn_phase.set_label('Phase (rad)', rotation=270, labelpad=15)
    
    cbar_baseline_phase = plt.colorbar(im3, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar_baseline_phase.set_label('Phase (rad)', rotation=270, labelpad=15)
    
    if ground_truth_obj is not None:
        cbar_gt_phase = plt.colorbar(im_gt_phase, ax=axes[0, gt_col], fraction=0.046, pad=0.04)
        cbar_gt_phase.set_label('Phase (rad)', rotation=270, labelpad=15)
    
    # Add individual colorbars for amplitude plots (bottom row)
    cbar_pinn_amp = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar_pinn_amp.set_label('Amplitude', rotation=270, labelpad=15)
    
    cbar_baseline_amp = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar_baseline_amp.set_label('Amplitude', rotation=270, labelpad=15)
    
    if ground_truth_obj is not None:
        cbar_gt_amp = plt.colorbar(im_gt_amp, ax=axes[1, gt_col], fraction=0.046, pad=0.04)
        cbar_gt_amp.set_label('Amplitude', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Visual comparison saved to {output_path}")


def save_frc_curves(frc_tuple, output_path, model_name):
    """
    Save raw FRC curves to a CSV file for detailed analysis.
    
    Args:
        frc_tuple: Tuple of (amplitude_frc_curve, phase_frc_curve) 
        output_path: Path to save the CSV file
        model_name: Name of the model for labeling
    """
    amp_frc, phase_frc = frc_tuple
    
    # Create data for CSV
    data = []
    max_length = max(len(amp_frc) if amp_frc is not None else 0, 
                     len(phase_frc) if phase_frc is not None else 0)
    
    for i in range(max_length):
        row = {
            'model': model_name,
            'frequency_bin': i,
            'amplitude_frc': amp_frc[i] if amp_frc is not None and i < len(amp_frc) else np.nan,
            'phase_frc': phase_frc[i] if phase_frc is not None and i < len(phase_frc) else np.nan
        }
        data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.6f')
    logger.info(f"FRC curves saved to {output_path}")


def save_metrics_csv(pinn_metrics, baseline_metrics, output_path, pinn_offset=None, baseline_offset=None, tike_metrics=None, tike_offset=None, pinn_time=None, baseline_time=None, tike_time=None, algorithm_name=None):
    """Save metrics to a CSV file in a tidy format.
    
    Supports 2-way (PtychoPINN vs Baseline) or 3-way comparison (includes iterative reconstruction)
    when tike_metrics is provided. Includes computation times and registration offsets.
    
    Args:
        pinn_metrics: PtychoPINN model metrics dictionary
        baseline_metrics: Baseline model metrics dictionary  
        output_path: Path to save CSV file
        pinn_offset: PtychoPINN registration offset (dy, dx)
        baseline_offset: Baseline registration offset (dy, dx)  
        tike_metrics: Iterative reconstruction metrics (optional)
        tike_offset: Iterative reconstruction registration offset (dy, dx) (optional)
        pinn_time: PtychoPINN inference time in seconds (optional)
        baseline_time: Baseline inference time in seconds (optional)
        tike_time: Iterative reconstruction computation time in seconds (optional)
        algorithm_name: Name of the iterative algorithm (e.g., "Pty-chi (ePIE)", "Tike") (optional)
    """
    data = []
    
    # Helper function to add metrics to data list
    def add_metrics(model_name, metrics_dict):
        for metric_name, metric_value in metrics_dict.items():
            if metric_name == 'frc':  # Skip FRC arrays
                continue
            if isinstance(metric_value, tuple) and len(metric_value) == 2:
                # Metrics that return (amplitude, phase) tuples
                data.append({
                    'model': model_name,
                    'metric': metric_name,
                    'amplitude': metric_value[0],
                    'phase': metric_value[1]
                })
            else:
                # Single-value metrics
                data.append({
                    'model': model_name,
                    'metric': metric_name,
                    'value': metric_value
                })
    
    # Add metrics for all models
    if pinn_metrics:
        add_metrics('PtychoPINN', pinn_metrics)
    if baseline_metrics:
        add_metrics('Baseline', baseline_metrics)
    if tike_metrics:
        # Use the actual algorithm name if provided, otherwise default to 'Tike'
        iterative_name = algorithm_name if algorithm_name else 'Tike'
        add_metrics(iterative_name, tike_metrics)
    
    # Add registration offset information
    if pinn_offset is not None:
        data.append({
            'model': 'PtychoPINN',
            'metric': 'registration_offset_dy',
            'value': float(pinn_offset[0])
        })
        data.append({
            'model': 'PtychoPINN',
            'metric': 'registration_offset_dx',
            'value': float(pinn_offset[1])
        })
    
    if baseline_offset is not None:
        data.append({
            'model': 'Baseline',
            'metric': 'registration_offset_dy',
            'value': float(baseline_offset[0])
        })
        data.append({
            'model': 'Baseline',
            'metric': 'registration_offset_dx',
            'value': float(baseline_offset[1])
        })
    
    if tike_offset is not None:
        data.append({
            'model': iterative_name,
            'metric': 'registration_offset_dy',
            'value': float(tike_offset[0])
        })
        data.append({
            'model': iterative_name,
            'metric': 'registration_offset_dx',
            'value': float(tike_offset[1])
        })
    
    # Add computation time information
    if pinn_time is not None:
        data.append({
            'model': 'PtychoPINN',
            'metric': 'computation_time_s',
            'value': float(pinn_time)
        })
    
    if baseline_time is not None:
        data.append({
            'model': 'Baseline',
            'metric': 'computation_time_s',
            'value': float(baseline_time)
        })
    
    if tike_time is not None:
        data.append({
            'model': iterative_name,
            'metric': 'computation_time_s',
            'value': float(tike_time)
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.6f')
    logger.info(f"Metrics saved to {output_path}")
    
    # Print summary
    print("\n--- Comparison Metrics ---")
    print(df.to_string(index=False))


def save_reconstruction_npz(pinn_recon, baseline_recon, ground_truth_obj, output_dir):
    """
    Save a single unified NPZ file containing amplitude, phase, and complex data for all reconstructions
    before any registration correction is applied.
    
    Args:
        pinn_recon: PtychoPINN reconstruction (complex array)
        baseline_recon: Baseline reconstruction (complex array)  
        ground_truth_obj: Ground truth object (complex array, can be None)
        output_dir: Directory to save NPZ files
    """
    output_dir = Path(output_dir)
    
    def extract_amp_phase_complex(complex_array):
        """Extract amplitude, phase, and complex from complex array."""
        if complex_array is None:
            return None, None, None
        # Squeeze to remove batch/channel dimensions
        squeezed = np.squeeze(complex_array)
        amplitude = np.abs(squeezed)
        phase = np.angle(squeezed)
        return amplitude, phase, squeezed
    
    # Extract amplitude, phase, and complex for each reconstruction
    pinn_amp, pinn_phase, pinn_complex = extract_amp_phase_complex(pinn_recon)
    baseline_amp, baseline_phase, baseline_complex = extract_amp_phase_complex(baseline_recon)
    gt_amp, gt_phase, gt_complex = extract_amp_phase_complex(ground_truth_obj)
    
    # Create unified data dictionary
    unified_data = {
        'ptychopinn_amplitude': pinn_amp,
        'ptychopinn_phase': pinn_phase,
        'ptychopinn_complex': pinn_complex,
        'baseline_amplitude': baseline_amp,
        'baseline_phase': baseline_phase,
        'baseline_complex': baseline_complex,
    }
    
    # Add ground truth if available
    if ground_truth_obj is not None:
        unified_data.update({
            'ground_truth_amplitude': gt_amp,
            'ground_truth_phase': gt_phase,
            'ground_truth_complex': gt_complex
        })
    
    # Save single unified file
    unified_path = output_dir / "reconstructions.npz"
    np.savez_compressed(unified_path, **unified_data)
    logger.info(f"Unified reconstructions saved to {unified_path}")
    
    # Also create a metadata file describing the contents
    metadata_path = output_dir / "reconstructions_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("NPZ File Contents:\n")
        f.write("==================\n\n")
        f.write("Arrays saved in reconstructions.npz:\n")
        for key in unified_data.keys():
            if unified_data[key] is not None:
                shape = unified_data[key].shape
                dtype = unified_data[key].dtype
                f.write(f"- {key}: {shape} {dtype}\n")
            else:
                f.write(f"- {key}: None (not available)\n")
        f.write("\nDescription:\n")
        f.write("- *_amplitude: Real-valued amplitude data\n")
        f.write("- *_phase: Real-valued phase data in radians\n")
        f.write("- *_complex: Complex-valued reconstruction data\n")
        f.write("- Data saved BEFORE registration correction\n")
    
    logger.info(f"Metadata saved to {metadata_path}")
    logger.info("Unified NPZ reconstruction file saved successfully!")
    
    return {
        'unified_path': unified_path,
        'metadata_path': metadata_path
    }


def save_aligned_reconstruction_npz(pinn_aligned, baseline_aligned, gt_aligned, pinn_offset, baseline_offset, output_dir):
    """
    Save a single unified NPZ file containing amplitude, phase, and complex data for aligned reconstructions
    after registration correction has been applied.
    
    Args:
        pinn_aligned: Aligned PtychoPINN reconstruction (complex array)
        baseline_aligned: Aligned Baseline reconstruction (complex array)  
        gt_aligned: Aligned ground truth object (complex array, can be None)
        pinn_offset: Detected offset for PtychoPINN as (dy, dx) tuple
        baseline_offset: Detected offset for Baseline as (dy, dx) tuple
        output_dir: Directory to save NPZ files
    """
    output_dir = Path(output_dir)
    
    def extract_amp_phase_complex(complex_array):
        """Extract amplitude, phase, and complex from complex array."""
        if complex_array is None:
            return None, None, None
        # No squeezing needed as these are already 2D from alignment
        amplitude = np.abs(complex_array)
        phase = np.angle(complex_array)
        return amplitude, phase, complex_array
    
    # Extract amplitude, phase, and complex for each aligned reconstruction
    pinn_amp, pinn_phase, pinn_complex = extract_amp_phase_complex(pinn_aligned)
    baseline_amp, baseline_phase, baseline_complex = extract_amp_phase_complex(baseline_aligned)
    gt_amp, gt_phase, gt_complex = extract_amp_phase_complex(gt_aligned)
    
    # Create unified aligned data dictionary
    unified_data = {
        'ptychopinn_amplitude': pinn_amp,
        'ptychopinn_phase': pinn_phase,
        'ptychopinn_complex': pinn_complex,
        'baseline_amplitude': baseline_amp,
        'baseline_phase': baseline_phase,
        'baseline_complex': baseline_complex,
        'pinn_offset_dy': float(pinn_offset[0]) if pinn_offset is not None else None,
        'pinn_offset_dx': float(pinn_offset[1]) if pinn_offset is not None else None,
        'baseline_offset_dy': float(baseline_offset[0]) if baseline_offset is not None else None,
        'baseline_offset_dx': float(baseline_offset[1]) if baseline_offset is not None else None,
    }
    
    # Add aligned ground truth if available
    if gt_aligned is not None:
        unified_data.update({
            'ground_truth_amplitude': gt_amp,
            'ground_truth_phase': gt_phase,
            'ground_truth_complex': gt_complex
        })
    
    # Save single unified aligned file
    unified_path = output_dir / "reconstructions_aligned.npz"
    np.savez_compressed(unified_path, **unified_data)
    logger.info(f"Unified aligned reconstructions saved to {unified_path}")
    
    # Also create a metadata file describing the contents
    metadata_path = output_dir / "reconstructions_aligned_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("Aligned NPZ File Contents:\n")
        f.write("==========================\n\n")
        f.write("Arrays saved in reconstructions_aligned.npz:\n")
        for key in unified_data.keys():
            if unified_data[key] is not None:
                if isinstance(unified_data[key], (int, float)):
                    f.write(f"- {key}: {unified_data[key]} (scalar)\n")
                else:
                    shape = unified_data[key].shape
                    dtype = unified_data[key].dtype
                    f.write(f"- {key}: {shape} {dtype}\n")
            else:
                f.write(f"- {key}: None (not available)\n")
        f.write("\nDescription:\n")
        f.write("- *_amplitude: Real-valued amplitude data\n")
        f.write("- *_phase: Real-valued phase data in radians\n")
        f.write("- *_complex: Complex-valued reconstruction data\n")
        f.write("- *_offset_dy, *_offset_dx: Registration offsets in pixels\n")
        f.write("- Data saved AFTER registration correction and alignment\n")
    
    logger.info(f"Aligned metadata saved to {metadata_path}")
    logger.info("Unified aligned NPZ reconstruction file saved successfully!")
    
    return {
        'unified_path': unified_path,
        'metadata_path': metadata_path
    }


def main():
    """
    Main comparison workflow with automatic registration.
    
    This function loads trained PtychoPINN and baseline models, runs inference on test data,
    performs automatic image registration to align reconstructions before evaluation,
    and generates comparison metrics and visualizations. Optionally supports three-way
    comparison including Tike iterative reconstruction when --tike_recon_path is provided.
    
    The alignment process consists of two stages:
    1. Coordinate-based alignment: Crops images to the scanned region based on scan coordinates
    2. Fine-scale registration: Detects and corrects pixel-level misalignments using cross-correlation
    
    Registration can be disabled using the --skip-registration flag for debugging purposes.
    Detected translation offsets are logged and included in both the CSV output and plot annotations.
    
    Outputs:
    - Two-way mode: 2x3 comparison plot (PtychoPINN vs Baseline vs Ground Truth)
    - Three-way mode: 2x4 comparison plot (PtychoPINN vs Baseline vs Tike vs Ground Truth)
    - CSV file with metrics and computation times for all included models
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate Tike reconstruction path if provided
    if args.tike_recon_path and not args.tike_recon_path.exists():
        raise FileNotFoundError(f"Tike reconstruction file not found: {args.tike_recon_path}")
    
    # Set up enhanced centralized logging
    logging_config = get_logging_config(args) if hasattr(args, 'quiet') else {}
    setup_logging(Path(args.output_dir) / "logs", **logging_config)

    # Handle NPZ flag combinations
    if args.no_save_npz:
        args.save_npz = False
    if args.no_save_npz_aligned:
        args.save_npz_aligned = False
    
    # Log configuration
    logger.info(f"Configuration: phase_align_method='{args.phase_align_method}', frc_sigma={args.frc_sigma}")
    logger.info(f"Registration: {'disabled' if args.skip_registration else 'enabled'}")
    logger.info(f"NPZ output: raw={'enabled' if args.save_npz else 'disabled'}, aligned={'enabled' if args.save_npz_aligned else 'disabled'}")

    # CRITICAL FIX: Initialize configuration BEFORE loading data
    # This ensures params.cfg is properly set up when load_data() checks gridsize
    # Load models FIRST to restore their saved configuration (including gridsize)
    logger.info("Loading PtychoPINN inference model (diffraction_to_obj) to restore configuration...")
    pinn_model = load_pinn_model(args.pinn_dir)
    
    # The model loading should have restored the correct gridsize to params.cfg
    restored_gridsize = p.cfg.get('gridsize', 1)
    restored_N = p.cfg.get('N', 64)
    logger.info(f"Restored configuration from model: gridsize={restored_gridsize}, N={restored_N}")
    
    # Load test data using the restored gridsize configuration
    logger.info(f"Loading test data from {args.test_data} with restored gridsize={restored_gridsize}...")
    
    # Handle backward compatibility for deprecated arguments
    if args.n_test_images is not None:
        logger.warning("--n-test-images is deprecated. Use --n-test-groups instead.")
        if args.n_test_groups is None:
            args.n_test_groups = args.n_test_images
    
    if args.n_subsample is not None:
        logger.warning("--n-subsample is deprecated. Use --n-test-subsample instead.")
        if args.n_test_subsample is None:
            args.n_test_subsample = args.n_subsample
    
    if args.subsample_seed is not None:
        logger.warning("--subsample-seed is deprecated. Use --test-subsample-seed instead.")
        if args.test_subsample_seed is None:
            args.test_subsample_seed = args.subsample_seed
    
    # Parameter interpretation: Handle backward compatibility and independent control
    if args.n_test_subsample is not None:
        logger.info(f"Using independent sampling control: n_subsample={args.n_test_subsample}, n_groups={args.n_test_groups}")
        if args.n_test_groups is not None and args.n_test_subsample > args.n_test_groups:
            logger.warning(f"n_test_subsample ({args.n_test_subsample}) > n_test_groups ({args.n_test_groups}), may use more data than expected")
    else:
        if args.n_test_groups is not None:
            logger.info(f"Using n_test_groups={args.n_test_groups} (controls grouping)")
        else:
            logger.info("Using all test data (no subsampling or grouping limit)")
    
    test_data_raw = load_data(str(args.test_data), 
                             n_images=args.n_test_groups,  # n_images in load_data corresponds to groups
                             n_subsample=args.n_test_subsample,
                             subsample_seed=args.test_subsample_seed)
    
    # Update only non-architecture parameters while preserving gridsize
    # This ensures we don't overwrite the critical architecture configuration
    # Use n_test_groups if specified, otherwise use number of available images
    n_groups_to_use = args.n_test_groups if args.n_test_groups else test_data_raw.diff3d.shape[0]
    final_config = TrainingConfig(
        model=ModelConfig(N=test_data_raw.probeGuess.shape[0], gridsize=restored_gridsize),  # Use restored gridsize!
        train_data_file=Path("dummy.npz"),
        n_groups=n_groups_to_use,  # Use requested test groups for oversampling
        neighbor_count=7  # Enable K-choose-C oversampling
    )
    # CRITICAL FIX: Update legacy params BEFORE creating data container
    # This ensures generate_grouped_data() sees the correct gridsize in params.cfg
    update_legacy_dict(p.cfg, final_config)
    logger.info(f"Final configuration: gridsize={restored_gridsize}, N={test_data_raw.probeGuess.shape[0]}, n_images={test_data_raw.diff3d.shape[0]}")
    
    # Validate stitch_crop_size parameter
    N = test_data_raw.probeGuess.shape[0]
    if not (0 < args.stitch_crop_size <= N):
        raise ValueError(f"Invalid stitch_crop_size: {args.stitch_crop_size}. Must be 0 < M <= N={N}")
    logger.info(f"Using stitch_crop_size M={args.stitch_crop_size} (N={N})")
    
    # Create data container AFTER legacy params are updated
    test_container = create_ptycho_data_container(test_data_raw, final_config)
    
    # Extract ground truth if available
    ground_truth_obj = test_data_raw.objectGuess[None, ..., None] if test_data_raw.objectGuess is not None else None

    # Load baseline model (pinn_model was already loaded above to restore config)
    baseline_model = load_baseline_model(args.baseline_dir)

    # Load Tike/Pty-chi reconstruction if provided
    if args.tike_recon_path:
        tike_recon, tike_computation_time, algorithm_name = load_tike_reconstruction(args.tike_recon_path)
        logger.info(f"{algorithm_name} reconstruction loaded for three-way comparison")
        if tike_computation_time is not None:
            logger.info(f"{algorithm_name} computation time: {tike_computation_time:.2f}s")
    else:
        tike_recon, tike_computation_time, algorithm_name = None, None, None
        logger.info("Running two-way comparison (PtychoPINN vs. Baseline)")

    # Run inference for PtychoPINN
    logger.info("Running inference with PtychoPINN (diffraction_to_obj)...")

    # CRITICAL FIX: Force gridsize from channel count before inference
    # When test data has grouped diffraction (e.g., 4 channels from gs=2),
    # params.cfg['gridsize'] must match sqrt(channel_count) to prevent
    # Translation layer batch dimension mismatch (B vs B·C crash)
    import math
    diffraction_channels = int(test_container.X.shape[-1])
    required_gridsize = int(math.isqrt(diffraction_channels))
    current_gridsize = p.cfg.get('gridsize', 1)

    if current_gridsize != required_gridsize:
        logger.warning(f"GRIDSIZE MISMATCH: params.cfg['gridsize']={current_gridsize} but diffraction has {diffraction_channels} channels (requires gridsize={required_gridsize})")
        logger.warning(f"Forcing params.cfg['gridsize']={required_gridsize} before inference to prevent Translation layer crash")
        p.cfg['gridsize'] = required_gridsize
    else:
        logger.info(f"Gridsize validated: params.cfg['gridsize']={current_gridsize} matches {diffraction_channels} channels")

    pinn_start = time.time()
    pinn_patches = pinn_model.predict(
        [test_container.X * p.get('intensity_scale'), test_container.coords_nominal],
        batch_size=32,
        verbose=1
    )
    pinn_inference_time = time.time() - pinn_start
    logger.info(f"PtychoPINN inference completed in {pinn_inference_time:.2f}s")

    if isinstance(pinn_patches, (list, tuple)) and pinn_patches:
        pinn_patches = pinn_patches[0]
    pinn_patches = np.asarray(pinn_patches)

    # Log PINN patches shape before reassembly
    logger.info(f"PINN patches shape before reassembly: {pinn_patches.shape}, dtype: {pinn_patches.dtype}")

    # Reassemble PINN patches using global offsets (not coords_nominal which is channel-formatted)
    logger.info("Reassembling PINN patches...")
    # Use global_offsets for reassembly - expects shape (B, 1, 2, 1)
    pinn_offsets = np.asarray(test_container.global_offsets, dtype=np.float64)
    logger.info(f"PINN offsets shape before reassembly: {pinn_offsets.shape}, dtype: {pinn_offsets.dtype}")
    pinn_recon = reassemble_position(pinn_patches, pinn_offsets, M=args.stitch_crop_size)
    logger.info(f"PINN reconstruction shape after reassembly: {pinn_recon.shape}, dtype: {pinn_recon.dtype}")

    # Run inference for Baseline
    logger.info("Running inference with Baseline model...")
    baseline_input, baseline_offsets = prepare_baseline_inference_data(test_container)

    # Apply debug limit if requested
    if args.baseline_debug_limit is not None:
        original_shape = baseline_input.shape
        baseline_input = baseline_input[:args.baseline_debug_limit]
        baseline_offsets = baseline_offsets[:args.baseline_debug_limit]
        logger.info(f"Applied --baseline-debug-limit={args.baseline_debug_limit}: limited {original_shape} → {baseline_input.shape}")

    logger.info(f"Baseline inference input shape: {baseline_input.shape}")
    logger.info(f"Baseline inference offsets shape: {baseline_offsets.shape}")

    # DIAGNOSTIC: Check baseline input data stats
    baseline_input_mean = baseline_input.mean()
    baseline_input_max = baseline_input.max()
    baseline_input_nonzero = np.count_nonzero(baseline_input)
    logger.info(f"DIAGNOSTIC baseline_input stats: mean={baseline_input_mean:.6f}, max={baseline_input_max:.6f}, nonzero_count={baseline_input_nonzero}/{baseline_input.size}")
    if baseline_input_mean == 0.0:
        logger.error("CRITICAL: Baseline input data is all zeros! Data preparation may have failed.")

    # Warn if offsets are missing (shouldn't happen after prepare_baseline_inference_data)
    if baseline_offsets is None or baseline_offsets.size == 0:
        logger.warning(
            "Baseline offsets are missing or empty. Baseline model expects both diffraction and offsets."
        )

    baseline_start = time.time()

    # Chunked inference to reduce GPU memory footprint
    n_groups = baseline_input.shape[0]
    chunk_size = args.baseline_chunk_size if args.baseline_chunk_size is not None else n_groups

    if chunk_size < n_groups:
        logger.info(f"Using chunked baseline inference: {n_groups} groups in chunks of {chunk_size} (batch_size={args.baseline_predict_batch_size})")
        baseline_output_chunks = []

        for chunk_start in range(0, n_groups, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_groups)
            chunk_idx = chunk_start // chunk_size
            logger.info(f"Processing baseline chunk {chunk_idx+1}/{(n_groups + chunk_size - 1) // chunk_size}: groups [{chunk_start}:{chunk_end}]")

            chunk_input = baseline_input[chunk_start:chunk_end]
            chunk_offsets = baseline_offsets[chunk_start:chunk_end]

            # Clear TF session before each chunk to release memory
            if chunk_idx > 0:
                tf.keras.backend.clear_session()

            try:
                chunk_output = baseline_model.predict(
                    [chunk_input, chunk_offsets],
                    batch_size=args.baseline_predict_batch_size,
                    verbose=1
                )

                # DIAGNOSTIC: Per-chunk stats
                chunk_output_np = np.asarray(chunk_output)
                chunk_mean = np.abs(chunk_output_np).mean()
                chunk_max = np.abs(chunk_output_np).max()
                chunk_nonzero = np.count_nonzero(chunk_output_np)
                logger.info(f"DIAGNOSTIC chunk {chunk_idx+1} baseline_output stats: mean={chunk_mean:.6f}, max={chunk_max:.6f}, nonzero_count={chunk_nonzero}/{chunk_output_np.size}")

                baseline_output_chunks.append(chunk_output_np)

            except tf.errors.ResourceExhaustedError as e:
                logger.error(f"ResourceExhaustedError in chunk {chunk_idx+1} at [{chunk_start}:{chunk_end}]: {e}")
                logger.error(f"Try reducing --baseline-chunk-size (currently {chunk_size}) or --baseline-predict-batch-size (currently {args.baseline_predict_batch_size})")
                raise

        # Concatenate all chunks
        baseline_output = np.concatenate(baseline_output_chunks, axis=0)
        logger.info(f"Concatenated {len(baseline_output_chunks)} chunks into final baseline_output shape: {baseline_output.shape}")
    else:
        # Single-shot inference (original path)
        logger.info(f"Using single-shot baseline inference: {n_groups} groups (batch_size={args.baseline_predict_batch_size})")
        baseline_output = baseline_model.predict(
            [baseline_input, baseline_offsets],
            batch_size=args.baseline_predict_batch_size,
            verbose=1
        )

    baseline_inference_time = time.time() - baseline_start
    logger.info(f"Baseline inference completed in {baseline_inference_time:.2f}s")

    # DIAGNOSTIC: Check if baseline output contains actual values
    baseline_output_np_check = np.asarray(baseline_output)
    baseline_output_mean = np.abs(baseline_output_np_check).mean()
    baseline_output_max = np.abs(baseline_output_np_check).max()
    baseline_output_nonzero = np.count_nonzero(baseline_output_np_check)
    logger.info(f"DIAGNOSTIC baseline_output stats: mean={baseline_output_mean:.6f}, max={baseline_output_max:.6f}, nonzero_count={baseline_output_nonzero}/{baseline_output_np_check.size}")

    # Save debug artifacts if requested
    if args.baseline_debug_dir is not None:
        import json
        debug_dir = Path(args.baseline_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        debug_stats = {
            "input": {
                "shape": list(baseline_input.shape),
                "mean": float(baseline_input_mean),
                "max": float(baseline_input_max),
                "nonzero_count": int(baseline_input_nonzero),
                "size": int(baseline_input.size)
            },
            "offsets": {
                "shape": list(baseline_offsets.shape),
                "mean": float(baseline_offsets.mean()),
                "std": float(baseline_offsets.std()),
                "min": float(baseline_offsets.min()),
                "max": float(baseline_offsets.max())
            },
            "output": {
                "shape": list(baseline_output_np_check.shape),
                "mean": float(baseline_output_mean),
                "max": float(baseline_output_max),
                "nonzero_count": int(baseline_output_nonzero),
                "size": int(baseline_output_np_check.size)
            },
            "inference_time_seconds": float(baseline_inference_time)
        }

        debug_npz_path = debug_dir / "baseline_debug.npz"
        debug_json_path = debug_dir / "baseline_debug_stats.json"

        np.savez_compressed(
            debug_npz_path,
            baseline_input=baseline_input,
            baseline_offsets=baseline_offsets,
            baseline_output=baseline_output_np_check
        )
        with open(debug_json_path, 'w') as f:
            json.dump(debug_stats, f, indent=2)

        logger.info(f"Saved baseline debug artifacts to {debug_dir}:")
        logger.info(f"  NPZ: {debug_npz_path}")
        logger.info(f"  Stats: {debug_json_path}")

    # CRITICAL ASSERTION: Halt execution if Baseline outputs are all zeros
    if baseline_output_mean == 0.0 or baseline_output_nonzero == 0:
        logger.error("CRITICAL: Baseline model returned all-zero predictions! Check model weights and input data.")
        logger.error(f"Baseline input stats: mean={baseline_input_mean:.6f}, max={baseline_input_max:.6f}, nonzero={baseline_input_nonzero}")
        logger.error(f"Baseline output stats: mean={baseline_output_mean:.6f}, max={baseline_output_max:.6f}, nonzero={baseline_output_nonzero}")
        raise RuntimeError(
            f"Baseline model inference failed: outputs are all zeros (mean={baseline_output_mean:.6f}, "
            f"nonzero={baseline_output_nonzero}/{baseline_output_np_check.size}). "
            f"This indicates a TensorFlow/model runtime issue. "
            f"Inputs were valid (mean={baseline_input_mean:.6f}, nonzero={baseline_input_nonzero}). "
            f"Investigation required before Phase G can proceed."
        )

    # Log baseline output shape before conversion
    if isinstance(baseline_output, list):
        logger.info(f"Baseline output format: list with {len(baseline_output)} elements")
        for idx, elem in enumerate(baseline_output):
            logger.info(f"  Element {idx} shape: {np.asarray(elem).shape}, dtype: {np.asarray(elem).dtype}")
    else:
        logger.info(f"Baseline output format: single tensor, shape: {np.asarray(baseline_output).shape}, dtype: {np.asarray(baseline_output).dtype}")

    # Handle different output formats
    if isinstance(baseline_output, list) and len(baseline_output) == 2:
        # Legacy format: [amplitude, phase]
        logger.info("Converting baseline output from [amplitude, phase] list format")
        baseline_patches_I, baseline_patches_phi = baseline_output
        baseline_patches_I = np.asarray(baseline_patches_I, dtype=np.float32)
        baseline_patches_phi = np.asarray(baseline_patches_phi, dtype=np.float32)
        baseline_patches_complex = baseline_patches_I.astype(np.complex64) * \
                                   np.exp(1j * baseline_patches_phi.astype(np.complex64))
    elif isinstance(baseline_output, np.ndarray) or hasattr(baseline_output, 'numpy'):
        # Single complex tensor format
        logger.info("Converting baseline output from single complex tensor format")
        baseline_output_np = np.asarray(baseline_output)
        if np.iscomplexobj(baseline_output_np):
            # Already complex
            baseline_patches_complex = baseline_output_np.astype(np.complex64)
            # Extract amplitude and phase for logging
            baseline_patches_I = np.abs(baseline_patches_complex).astype(np.float32)
            baseline_patches_phi = np.angle(baseline_patches_complex).astype(np.float32)
        else:
            # Single real tensor - interpret as complex (this shouldn't happen but handle it)
            logger.warning("Baseline output is single real tensor, cannot convert to amplitude/phase")
            raise ValueError(f"Unexpected baseline model output: single real tensor with shape {baseline_output_np.shape}")
    else:
        raise ValueError(f"Unexpected baseline model output format: {type(baseline_output)}")

    # Log shapes after conversion
    logger.info(f"After conversion - amplitude shape: {baseline_patches_I.shape}, phase shape: {baseline_patches_phi.shape}")
    logger.info(f"Complex patches shape: {baseline_patches_complex.shape}, dtype: {baseline_patches_complex.dtype}")

    # DIAGNOSTIC: Log first patch statistics to help diagnose zero-output issues
    if baseline_patches_I.size > 0:
        first_patch_amp = baseline_patches_I[0]
        first_patch_phase = baseline_patches_phi[0]
        logger.info(f"DIAGNOSTIC first patch amplitude: shape={first_patch_amp.shape}, mean={first_patch_amp.mean():.6f}, "
                   f"max={first_patch_amp.max():.6f}, nonzero={np.count_nonzero(first_patch_amp)}/{first_patch_amp.size}")
        logger.info(f"DIAGNOSTIC first patch phase: shape={first_patch_phase.shape}, mean={first_patch_phase.mean():.6f}, "
                   f"std={first_patch_phase.std():.6f}, min={first_patch_phase.min():.6f}, max={first_patch_phase.max():.6f}")

    # Reassemble patches
    logger.info("Reassembling baseline patches...")
    
    baseline_recon = reassemble_position(baseline_patches_complex, baseline_offsets, M=args.stitch_crop_size)

    # Save NPZ files of reconstructions (before any alignment/registration) if requested
    if args.save_npz:
        logger.info("Saving NPZ files of raw reconstructions...")
        npz_paths = save_reconstruction_npz(pinn_recon, baseline_recon, ground_truth_obj, args.output_dir)
    else:
        logger.info("Raw NPZ export disabled (use --save-npz to enable)")
        npz_paths = None

    # Evaluate reconstructions
    pinn_metrics = {}
    baseline_metrics = {}
    cropped_gt = None
    
    # Track registration offsets for visualization
    pinn_offset = None
    baseline_offset = None
    
    # Coordinate-based ground truth alignment and evaluation
    if ground_truth_obj is not None:
        logger.info("Performing coordinate-based alignment of ground truth...")
        
        # Squeeze ground truth to 2D
        gt_obj_squeezed = ground_truth_obj.squeeze()
        logger.info(f"Ground truth original shape: {gt_obj_squeezed.shape}")
        
        # --- COORDINATE-BASED ALIGNMENT + REGISTRATION WORKFLOW ---
        
        # 1. Define the stitching parameter
        M_STITCH_SIZE = args.stitch_crop_size

        # 2. Extract scan coordinates in (y, x) format
        global_offsets = test_container.global_offsets
        scan_coords_xy = np.squeeze(global_offsets)
        scan_coords_yx = scan_coords_xy[:, [1, 0]]

        # 3. First stage: Coordinate-based alignment (crop to scanned region)
        pinn_recon_cropped, gt_cropped_for_pinn = align_for_evaluation(
            reconstruction_image=pinn_recon,
            ground_truth_image=ground_truth_obj,
            scan_coords_yx=scan_coords_yx,
            stitch_patch_size=M_STITCH_SIZE
        )
        
        baseline_recon_cropped, gt_cropped_for_baseline = align_for_evaluation(
            reconstruction_image=baseline_recon,
            ground_truth_image=ground_truth_obj,
            scan_coords_yx=scan_coords_yx,
            stitch_patch_size=M_STITCH_SIZE
        )
        
        # Crop Tike reconstruction if available
        if tike_recon is not None:
            tike_recon_cropped, gt_cropped_for_tike = align_for_evaluation(
                reconstruction_image=tike_recon,
                ground_truth_image=ground_truth_obj,
                scan_coords_yx=scan_coords_yx,
                stitch_patch_size=M_STITCH_SIZE
            )
        else:
            tike_recon_cropped = None
        
        # Use the first cropped GT (should be identical for both)
        cropped_gt = gt_cropped_for_pinn
        
        # 4. Second stage: Fine-scale registration (correct pixel-level shifts)
        if not args.skip_registration:
            if args.register_ptychi_only:
                logger.info("Performing selective registration (ONLY for pty-chi/tike)...")
                
                # Don't register PINN or Baseline
                pinn_recon_aligned = pinn_recon_cropped
                baseline_recon_aligned = baseline_recon_cropped
                pinn_offset = None
                baseline_offset = None
                logger.info("PtychoPINN and Baseline: skipping registration (as intended)")
                
                # Only register Tike/Pty-chi if available
                if tike_recon_cropped is not None:
                    try:
                        tike_offset = find_translation_offset(tike_recon_cropped, cropped_gt, upsample_factor=50)
                        logger.info(f"{algorithm_name} detected offset: ({tike_offset[0]:.3f}, {tike_offset[1]:.3f})")
                        tike_recon_aligned, gt_aligned_for_tike = apply_shift_and_crop(
                            tike_recon_cropped, cropped_gt, tike_offset, border_crop=2
                        )
                        # Keep original cropped_gt for PINN/Baseline, store aligned GT separately
                        gt_for_tike_eval = gt_aligned_for_tike
                    except Exception as e:
                        logger.warning(f"Registration failed for {algorithm_name}: {e}. Using unregistered version.")
                        tike_recon_aligned = tike_recon_cropped
                        tike_offset = None
                        gt_for_tike_eval = cropped_gt  # Use original GT if registration fails
                else:
                    tike_recon_aligned = None
                    tike_offset = None
                    gt_for_tike_eval = cropped_gt
            else:
                logger.info("Performing fine-scale registration to correct pixel-level misalignments...")
                
                try:
                    # Register PINN reconstruction against ground truth  
                    pinn_offset = find_translation_offset(pinn_recon_cropped, cropped_gt, upsample_factor=50)
                    logger.info(f"PtychoPINN detected offset: ({pinn_offset[0]:.3f}, {pinn_offset[1]:.3f})")
                    pinn_recon_aligned, gt_aligned_for_pinn = apply_shift_and_crop(
                        pinn_recon_cropped, cropped_gt, pinn_offset, border_crop=2
                    )
                    
                    # Register Baseline reconstruction against ground truth
                    baseline_offset = find_translation_offset(baseline_recon_cropped, cropped_gt, upsample_factor=50)
                    logger.info(f"Baseline detected offset: ({baseline_offset[0]:.3f}, {baseline_offset[1]:.3f})")
                    baseline_recon_aligned, gt_aligned_for_baseline = apply_shift_and_crop(
                        baseline_recon_cropped, cropped_gt, baseline_offset, border_crop=2
                    )
                    
                    # Register Tike reconstruction against ground truth if available
                    if tike_recon_cropped is not None:
                        tike_offset = find_translation_offset(tike_recon_cropped, cropped_gt, upsample_factor=50)
                        logger.info(f"Tike detected offset: ({tike_offset[0]:.3f}, {tike_offset[1]:.3f})")
                        tike_recon_aligned, gt_aligned_for_tike = apply_shift_and_crop(
                            tike_recon_cropped, cropped_gt, tike_offset, border_crop=2
                        )
                    else:
                        tike_recon_aligned = None
                        tike_offset = None
                    
                    # Use the GT aligned with PINN (both should be nearly identical)
                    cropped_gt = gt_aligned_for_pinn
                    
                    # Log registration results
                    if tike_recon_aligned is not None:
                        logger.info(f"Registration completed. PtychoPINN offset: {pinn_offset}, Baseline offset: {baseline_offset}, {algorithm_name} offset: {tike_offset}")
                        logger.info(f"Final aligned shapes - PINN: {pinn_recon_aligned.shape}, Baseline: {baseline_recon_aligned.shape}, {algorithm_name}: {tike_recon_aligned.shape}, GT: {cropped_gt.shape}")
                    else:
                        logger.info(f"Registration completed. PtychoPINN offset: {pinn_offset}, Baseline offset: {baseline_offset}")
                        logger.info(f"Final aligned shapes - PINN: {pinn_recon_aligned.shape}, Baseline: {baseline_recon_aligned.shape}, GT: {cropped_gt.shape}")
                    
                except Exception as e:
                    logger.warning(f"Registration failed: {e}. Continuing with coordinate-aligned images.")
                    pinn_recon_aligned = pinn_recon_cropped
                    baseline_recon_aligned = baseline_recon_cropped
                    tike_recon_aligned = tike_recon_cropped
                    pinn_offset = baseline_offset = tike_offset = None
                    # cropped_gt already set above
        else:
            logger.info("Skipping registration (--skip-registration specified)")
            pinn_recon_aligned = pinn_recon_cropped
            baseline_recon_aligned = baseline_recon_cropped
            tike_recon_aligned = tike_recon_cropped
            pinn_offset = baseline_offset = tike_offset = None
            # cropped_gt already set above
        
        if tike_recon_aligned is not None:
            logger.info(f"Final evaluation shapes: PINN {pinn_recon_aligned.shape}, Baseline {baseline_recon_aligned.shape}, {algorithm_name} {tike_recon_aligned.shape}, GT {cropped_gt.shape}")
        else:
            logger.info(f"Final evaluation shapes: PINN {pinn_recon_aligned.shape}, Baseline {baseline_recon_aligned.shape}, GT {cropped_gt.shape}")
        
        # Evaluate with aligned arrays (add back dimensions for eval function)
        # eval_reconstruction expects: stitched_obj=(batch, H, W, channels), ground_truth_obj=(H, W, channels)
        try:
            pinn_metrics = eval_reconstruction(
                pinn_recon_aligned[None, ..., None],  # (1, H, W, 1)
                cropped_gt[..., None],                 # (H, W, 1) - no batch dimension!
                label="PtychoPINN",
                phase_align_method=args.phase_align_method,
                frc_sigma=args.frc_sigma,
                debug_save_images=args.save_debug_images,
                ms_ssim_sigma=args.ms_ssim_sigma
            )
            logger.info(f"PtychoPINN evaluation complete. SSIM: amp={pinn_metrics['ssim'][0]:.3f}, phase={pinn_metrics['ssim'][1]:.3f}, MS-SSIM: amp={pinn_metrics['ms_ssim'][0]:.3f}, phase={pinn_metrics['ms_ssim'][1]:.3f}")
        except Exception as e:
            logger.error(f"PtychoPINN evaluation failed: {e}")
            # Create dummy metrics with NaN values to allow comparison to continue
            pinn_metrics = {
                'mae': (np.nan, np.nan), 'mse': (np.nan, np.nan), 
                'psnr': (np.nan, np.nan), 'ssim': (np.nan, np.nan),
                'ms_ssim': (np.nan, np.nan),
                'frc50': (np.nan, np.nan), 'frc': (None, None)
            }
        
        try:
            baseline_metrics = eval_reconstruction(
                baseline_recon_aligned[None, ..., None],  # (1, H, W, 1)
                cropped_gt[..., None],                     # (H, W, 1) - no batch dimension!
                label="Baseline",
                phase_align_method=args.phase_align_method,
                frc_sigma=args.frc_sigma,
                debug_save_images=args.save_debug_images,
                ms_ssim_sigma=args.ms_ssim_sigma
            )
            logger.info(f"Baseline evaluation complete. SSIM: amp={baseline_metrics['ssim'][0]:.3f}, phase={baseline_metrics['ssim'][1]:.3f}, MS-SSIM: amp={baseline_metrics['ms_ssim'][0]:.3f}, phase={baseline_metrics['ms_ssim'][1]:.3f}")
        except Exception as e:
            logger.error(f"Baseline evaluation failed: {e}")
            # Create dummy metrics with NaN values to allow comparison to continue
            baseline_metrics = {
                'mae': (np.nan, np.nan), 'mse': (np.nan, np.nan), 
                'psnr': (np.nan, np.nan), 'ssim': (np.nan, np.nan),
                'ms_ssim': (np.nan, np.nan),
                'frc50': (np.nan, np.nan), 'frc': (None, None)
            }
        
        # Evaluate Tike reconstruction if available
        if tike_recon_aligned is not None:
            try:
                # Use appropriate GT based on registration mode
                if args.register_ptychi_only and 'gt_for_tike_eval' in locals():
                    eval_gt = gt_for_tike_eval
                else:
                    eval_gt = cropped_gt
                    
                tike_metrics = eval_reconstruction(
                    tike_recon_aligned[None, ..., None],  # (1, H, W, 1)
                    eval_gt[..., None],                    # (H, W, 1) - no batch dimension!
                    label=algorithm_name,
                    phase_align_method=args.phase_align_method,
                    frc_sigma=args.frc_sigma,
                    debug_save_images=args.save_debug_images,
                    ms_ssim_sigma=args.ms_ssim_sigma
                )
                logger.info(f"{algorithm_name} evaluation complete. SSIM: amp={tike_metrics['ssim'][0]:.3f}, phase={tike_metrics['ssim'][1]:.3f}, MS-SSIM: amp={tike_metrics['ms_ssim'][0]:.3f}, phase={tike_metrics['ms_ssim'][1]:.3f}")
            except Exception as e:
                logger.error(f"{algorithm_name} evaluation failed: {e}")
                # Create dummy metrics with NaN values to allow comparison to continue
                tike_metrics = {
                    'mae': (np.nan, np.nan), 'mse': (np.nan, np.nan), 
                    'psnr': (np.nan, np.nan), 'ssim': (np.nan, np.nan),
                    'ms_ssim': (np.nan, np.nan),
                    'frc50': (np.nan, np.nan), 'frc': (None, None)
                }
        else:
            tike_metrics = None
        
        # Save scalar metrics to CSV
        metrics_path = args.output_dir / "comparison_metrics.csv"
        save_metrics_csv(pinn_metrics, baseline_metrics, metrics_path, pinn_offset, baseline_offset, tike_metrics, tike_offset, pinn_inference_time, baseline_inference_time, tike_computation_time, algorithm_name)
        
        # Save raw FRC curves as separate files for detailed analysis
        if pinn_metrics['frc'][0] is not None:
            pinn_frc_path = args.output_dir / "pinn_frc_curves.csv"
            save_frc_curves(pinn_metrics['frc'], pinn_frc_path, "PtychoPINN")
        
        if baseline_metrics['frc'][0] is not None:
            baseline_frc_path = args.output_dir / "baseline_frc_curves.csv"
            save_frc_curves(baseline_metrics['frc'], baseline_frc_path, "Baseline")
        
        # Save aligned NPZ files if requested
        if args.save_npz_aligned:
            logger.info("Saving NPZ files of aligned reconstructions...")
            aligned_npz_paths = save_aligned_reconstruction_npz(
                pinn_recon_aligned, baseline_recon_aligned, cropped_gt, 
                pinn_offset, baseline_offset, args.output_dir
            )
        else:
            logger.info("Aligned NPZ export disabled (use --save-npz-aligned to enable)")
            aligned_npz_paths = None
    else:
        logger.warning("No ground truth object found in test data. Skipping metric evaluation.")
        cropped_gt = None
        pinn_recon_aligned = np.squeeze(pinn_recon)
        baseline_recon_aligned = np.squeeze(baseline_recon)
        
        # For cases without ground truth, we can still save aligned reconstructions if registration was performed
        if args.save_npz_aligned and not args.skip_registration:
            logger.info("Saving NPZ files of aligned reconstructions (no ground truth available)...")
            aligned_npz_paths = save_aligned_reconstruction_npz(
                pinn_recon_aligned, baseline_recon_aligned, None, 
                pinn_offset, baseline_offset, args.output_dir
            )
        else:
            aligned_npz_paths = None

    # Create comparison plot
    plot_path = args.output_dir / "comparison_plot.png"
    create_comparison_plot(pinn_recon_aligned, baseline_recon_aligned, cropped_gt, plot_path, 
                          p_min=args.p_min, p_max=args.p_max,
                          pinn_phase_vmin=args.pinn_phase_vmin, pinn_phase_vmax=args.pinn_phase_vmax,
                          baseline_phase_vmin=args.baseline_phase_vmin, baseline_phase_vmax=args.baseline_phase_vmax,
                          pinn_offset=pinn_offset, baseline_offset=baseline_offset,
                          tike_obj=tike_recon_aligned, tike_offset=tike_offset, algorithm_name=algorithm_name)
    
    logger.info("\nComparison complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
