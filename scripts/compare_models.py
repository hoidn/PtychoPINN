#!/usr/bin/env python3
"""
compare_models.py - Load trained PtychoPINN and baseline models, run inference,
calculate metrics, and generate comparison visualizations.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ptycho components
from ptycho.model_manager import ModelManager
from ptycho.workflows.components import load_data, create_ptycho_data_container, logger
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p
from ptycho.tf_helper import reassemble_position
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
    
    # Add logging arguments
    add_logging_arguments(parser)
    
    return parser.parse_args()


def load_pinn_model(model_dir: Path) -> tf.keras.Model:
    """Load the inference-only PtychoPINN model."""
    logger.info(f"Loading PtychoPINN model from {model_dir}...")
    
    # ModelManager expects the path *without* the .zip extension
    model_zip_base_path = model_dir / "wts.h5"
    
    if not (model_dir / "wts.h5.zip").exists():
        raise FileNotFoundError(f"PtychoPINN model not found at {model_dir}/wts.h5.zip")
    
    models = ModelManager.load_multiple_models(str(model_zip_base_path))
    pinn_model = models.get('diffraction_to_obj')
    
    if pinn_model is None:
        raise ValueError("Could not find 'diffraction_to_obj' model in PtychoPINN archive.")
    
    return pinn_model


def load_baseline_model(baseline_dir: Path) -> tf.keras.Model:
    """Load the Keras baseline model."""
    logger.info(f"Loading Baseline model from {baseline_dir}...")
    
    # Find the baseline model file
    baseline_model_path = None
    for root, dirs, files in os.walk(baseline_dir):
        for file in files:
            if file == "baseline_model.h5":
                baseline_model_path = Path(root) / file
                break
        if baseline_model_path:
            break
    
    if baseline_model_path is None or not baseline_model_path.exists():
        raise FileNotFoundError(f"Baseline model not found in: {baseline_dir}")
    
    logger.info(f"Found baseline model at: {baseline_model_path}")
    return tf.keras.models.load_model(baseline_model_path)


def create_comparison_plot(pinn_obj, baseline_obj, ground_truth_obj, output_path, 
                          p_min=10.0, p_max=90.0, 
                          pinn_phase_vmin=None, pinn_phase_vmax=None,
                          baseline_phase_vmin=None, baseline_phase_vmax=None,
                          pinn_offset=None, baseline_offset=None):
    """Create a 2x3 subplot comparing reconstructions.
    
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
    """
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
    
    axes[0, 2].set_title("Ground Truth")
    
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

    # Plot PtychoPINN
    im1 = axes[0, 0].imshow(np.angle(pinn_obj), vmin=pinn_v_phase_min, vmax=pinn_v_phase_max)
    im2 = axes[1, 0].imshow(np.abs(pinn_obj), cmap='gray', vmin=pinn_v_amp_min, vmax=pinn_v_amp_max)
    
    # Plot Baseline
    im3 = axes[0, 1].imshow(np.angle(baseline_obj), vmin=baseline_v_phase_min, vmax=baseline_v_phase_max)
    im4 = axes[1, 1].imshow(np.abs(baseline_obj), cmap='gray', vmin=baseline_v_amp_min, vmax=baseline_v_amp_max)
    
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
        
        im5 = axes[0, 2].imshow(np.angle(gt_obj), vmin=gt_v_phase_min, vmax=gt_v_phase_max)
        im6 = axes[1, 2].imshow(np.abs(gt_obj), cmap='gray', vmin=gt_v_amp_min, vmax=gt_v_amp_max)
    else:
        for ax in [axes[0, 2], axes[1, 2]]:
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
        cbar_gt_phase = plt.colorbar(im5, ax=axes[0, 2], fraction=0.046, pad=0.04)
        cbar_gt_phase.set_label('Phase (rad)', rotation=270, labelpad=15)
    
    # Add individual colorbars for amplitude plots (bottom row)
    cbar_pinn_amp = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar_pinn_amp.set_label('Amplitude', rotation=270, labelpad=15)
    
    cbar_baseline_amp = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar_baseline_amp.set_label('Amplitude', rotation=270, labelpad=15)
    
    if ground_truth_obj is not None:
        cbar_gt_amp = plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
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


def save_metrics_csv(pinn_metrics, baseline_metrics, output_path, pinn_offset=None, baseline_offset=None):
    """Save metrics to a CSV file in a tidy format."""
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
    
    # Add metrics for both models
    if pinn_metrics:
        add_metrics('PtychoPINN', pinn_metrics)
    if baseline_metrics:
        add_metrics('Baseline', baseline_metrics)
    
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
    and generates comparison metrics and visualizations.
    
    The alignment process consists of two stages:
    1. Coordinate-based alignment: Crops images to the scanned region based on scan coordinates
    2. Fine-scale registration: Detects and corrects pixel-level misalignments using cross-correlation
    
    Registration can be disabled using the --skip-registration flag for debugging purposes.
    Detected translation offsets are logged and included in both the CSV output and plot annotations.
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
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

    # Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    test_data_raw = load_data(str(args.test_data))
    
    # Load models first to inspect their requirements
    pinn_model = load_pinn_model(args.pinn_dir)
    baseline_model = load_baseline_model(args.baseline_dir)
    
    # Determine gridsize requirements for each model
    baseline_channels = baseline_model.input_shape[-1]
    baseline_gridsize = int(baseline_channels ** 0.5)  # gridsize^2 = channels
    
    # For PINN model, check if it has dual inputs (diffraction + positions)
    if isinstance(pinn_model.input_shape, list):
        pinn_channels = pinn_model.input_shape[0][-1]  # First input is diffraction
    else:
        pinn_channels = pinn_model.input_shape[-1]
    pinn_gridsize = int(pinn_channels ** 0.5)
    
    logger.info(f"Baseline model expects {baseline_channels} channels (gridsize={baseline_gridsize})")
    logger.info(f"PINN model expects {pinn_channels} channels (gridsize={pinn_gridsize})")
    
    # Create separate data containers for each model based on their native gridsize
    # This ensures each model gets data in the format it expects
    
    # Create container for PINN model
    pinn_config = TrainingConfig(
        model=ModelConfig(N=test_data_raw.probeGuess.shape[0], gridsize=pinn_gridsize),
        train_data_file=Path("dummy.npz"),
        n_images=test_data_raw.diff3d.shape[0]
    )
    update_legacy_dict(p.cfg, pinn_config)
    pinn_container = create_ptycho_data_container(test_data_raw, pinn_config)
    logger.info(f"Created PINN container with shape: {pinn_container.X.shape}")
    
    # Create container for baseline model  
    baseline_config = TrainingConfig(
        model=ModelConfig(N=test_data_raw.probeGuess.shape[0], gridsize=baseline_gridsize),
        train_data_file=Path("dummy.npz"),
        n_images=test_data_raw.diff3d.shape[0]
    )
    update_legacy_dict(p.cfg, baseline_config)
    baseline_container = create_ptycho_data_container(test_data_raw, baseline_config)
    logger.info(f"Created baseline container with shape: {baseline_container.X.shape}")
    
    # GEMINI'S DIAGNOSTIC: Log channel semantics info (images already saved)
    if baseline_container.X.shape[-1] == 4:
        logger.info("GEMINI DIAGNOSTIC: Baseline model using 4-channel data (gridsize=2 neighboring patches)")
    
    if pinn_container.X.shape[-1] == 4:
        logger.info("GEMINI DIAGNOSTIC: PINN model using 4-channel data (gridsize=2 neighboring patches)")
    elif pinn_container.X.shape[-1] == 1:
        logger.info("GEMINI DIAGNOSTIC: PINN model using 1-channel data (gridsize=1 single patches)")
    
    # Use the PINN container for ground truth extraction (since both should be equivalent)
    test_container = pinn_container
    
    # Extract ground truth if available
    ground_truth_obj = test_data_raw.objectGuess[None, ..., None] if test_data_raw.objectGuess is not None else None


    # Run inference for PtychoPINN using its native container
    logger.info("Running inference with PtychoPINN...")
    logger.info(f"PINN input shape: {pinn_container.X.shape} -> model expects: {pinn_model.input_shape}")
    
    # Use PINN's native container - no shape adaptation needed
    pinn_patches = pinn_model.predict(
        [pinn_container.X * p.get('intensity_scale'), pinn_container.coords_nominal],
        batch_size=32,
        verbose=1
    )
    logger.info("PtychoPINN inference completed successfully!")
    
    # Reassemble patches using PINN container's offsets
    logger.info("Reassembling PtychoPINN patches...")
    pinn_recon = reassemble_position(pinn_patches, pinn_container.global_offsets, M=20)

    # Run inference for Baseline using its native container
    logger.info("Running inference with Baseline model...")
    logger.info(f"Baseline input shape: {baseline_container.X.shape} -> model expects: {baseline_model.input_shape}")
    
    # Use baseline's native container - no shape adaptation needed
    baseline_output = baseline_model.predict(baseline_container.X, batch_size=32, verbose=1)
    logger.info("Baseline inference completed successfully!")
    
    # Handle different output formats
    if isinstance(baseline_output, list) and len(baseline_output) == 2:
        baseline_patches_I, baseline_patches_phi = baseline_output
    else:
        # Some baseline models might output a single array
        raise ValueError("Unexpected baseline model output format")
    
    # Convert to complex representation
    baseline_patches_complex = tf.cast(baseline_patches_I, tf.complex64) * \
                               tf.exp(1j * tf.cast(baseline_patches_phi, tf.complex64))
    
    # Reassemble patches using baseline container's offsets
    logger.info("Reassembling baseline patches...")
    baseline_recon = reassemble_position(baseline_patches_complex, baseline_container.global_offsets, M=20)

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
        M_STITCH_SIZE = 20

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
        
        # Use the first cropped GT (should be identical for both)
        cropped_gt = gt_cropped_for_pinn
        
        # 4. Second stage: Fine-scale registration (correct pixel-level shifts)
        if not args.skip_registration:
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
                
                # Use the GT aligned with PINN (both should be nearly identical)
                cropped_gt = gt_aligned_for_pinn
                
                # Log registration results
                logger.info(f"Registration completed. PtychoPINN offset: {pinn_offset}, Baseline offset: {baseline_offset}")
                logger.info(f"Final aligned shapes - PINN: {pinn_recon_aligned.shape}, Baseline: {baseline_recon_aligned.shape}, GT: {cropped_gt.shape}")
                
            except Exception as e:
                logger.warning(f"Registration failed: {e}. Continuing with coordinate-aligned images.")
                pinn_recon_aligned = pinn_recon_cropped
                baseline_recon_aligned = baseline_recon_cropped
                # cropped_gt already set above
        else:
            logger.info("Skipping registration (--skip-registration specified)")
            pinn_recon_aligned = pinn_recon_cropped
            baseline_recon_aligned = baseline_recon_cropped
            # cropped_gt already set above
        
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
        
        # Save scalar metrics to CSV
        metrics_path = args.output_dir / "comparison_metrics.csv"
        save_metrics_csv(pinn_metrics, baseline_metrics, metrics_path, pinn_offset, baseline_offset)
        
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
                          pinn_offset=pinn_offset, baseline_offset=baseline_offset)
    
    logger.info("\nComparison complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()