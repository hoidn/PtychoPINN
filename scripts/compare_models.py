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
from ptycho.image.cropping import crop_to_scan_area

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


def create_comparison_plot(pinn_obj, baseline_obj, ground_truth_obj, output_path):
    """Create a 2x3 subplot comparing reconstructions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle("PtychoPINN vs. Baseline Reconstruction", fontsize=16)

    # Set titles
    axes[0, 0].set_title("PtychoPINN")
    axes[0, 1].set_title("Baseline")
    axes[0, 2].set_title("Ground Truth")
    
    # Set row labels
    axes[0, 0].set_ylabel("Phase", fontsize=14)
    axes[1, 0].set_ylabel("Amplitude", fontsize=14)

    # Calculate shared amplitude scale
    amp_pinn = np.abs(pinn_obj)
    amp_baseline = np.abs(baseline_obj)
    
    if ground_truth_obj is not None:
        amp_gt = np.abs(ground_truth_obj.squeeze())
        v_amp_min = min(np.min(amp_pinn), np.min(amp_baseline), np.min(amp_gt))
        v_amp_max = max(np.max(amp_pinn), np.max(amp_baseline), np.max(amp_gt))
    else:
        v_amp_min = min(np.min(amp_pinn), np.min(amp_baseline))
        v_amp_max = max(np.max(amp_pinn), np.max(amp_baseline))

    # Plot PtychoPINN
    im1 = axes[0, 0].imshow(np.angle(pinn_obj), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    im2 = axes[1, 0].imshow(amp_pinn, cmap='gray', vmin=v_amp_min, vmax=v_amp_max)
    
    # Plot Baseline
    im3 = axes[0, 1].imshow(np.angle(baseline_obj), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    im4 = axes[1, 1].imshow(amp_baseline, cmap='gray', vmin=v_amp_min, vmax=v_amp_max)
    
    # Plot Ground Truth
    if ground_truth_obj is not None:
        # Remove extra dimensions if present
        gt_obj = ground_truth_obj.squeeze()
        im5 = axes[0, 2].imshow(np.angle(gt_obj), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        im6 = axes[1, 2].imshow(amp_gt, cmap='gray', vmin=v_amp_min, vmax=v_amp_max)
    else:
        for ax in [axes[0, 2], axes[1, 2]]:
            ax.text(0.5, 0.5, "No Ground Truth", ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_facecolor('lightgray')

    # Remove ticks for cleaner appearance
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axes[0, :], fraction=0.046, pad=0.04)
    cbar1.set_label('Phase (radians)', rotation=270, labelpad=20)
    
    if ground_truth_obj is not None:
        cbar2 = plt.colorbar(im6, ax=axes[1, :], fraction=0.046, pad=0.04)
        cbar2.set_label('Amplitude', rotation=270, labelpad=20)
    else:
        cbar2 = plt.colorbar(im4, ax=axes[1, :], fraction=0.046, pad=0.04)
        cbar2.set_label('Amplitude', rotation=270, labelpad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Visual comparison saved to {output_path}")


def save_metrics_csv(pinn_metrics, baseline_metrics, output_path):
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
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.6f')
    logger.info(f"Metrics saved to {output_path}")
    
    # Print summary
    print("\n--- Comparison Metrics ---")
    print(df.to_string(index=False))


def main():
    """Main comparison workflow."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    test_data_raw = load_data(str(args.test_data))
    
    # Create a minimal config for data container creation
    # The actual model parameters will come from the saved models
    dummy_config = TrainingConfig(
        model=ModelConfig(N=test_data_raw.probeGuess.shape[0]),
        train_data_file=Path("dummy.npz"),
        n_images=test_data_raw.diff3d.shape[0]
    )
    update_legacy_dict(p.cfg, dummy_config)
    
    # Create data container
    test_container = create_ptycho_data_container(test_data_raw, dummy_config)
    
    # Extract ground truth if available
    ground_truth_obj = test_data_raw.objectGuess[None, ..., None] if test_data_raw.objectGuess is not None else None

    # Load models
    pinn_model = load_pinn_model(args.pinn_dir)
    baseline_model = load_baseline_model(args.baseline_dir)

    # TODO: This import is delayed because ptycho.nbutils -> ptycho.model has
    # import-time dependencies on global params.cfg keys ('probe', 'intensity_scale').
    # Loading the models first populates these keys. A better long-term fix
    # would be to refactor ptycho.model to remove module-level state access.
    from ptycho.nbutils import crop_to_non_uniform_region_with_buffer

    # Run inference for PtychoPINN
    logger.info("Running inference with PtychoPINN...")
    # PtychoPINN model requires both diffraction patterns and position coordinates
    # Scale the input data by intensity_scale to match training normalization
    pinn_patches = pinn_model.predict(
        [test_container.X * p.get('intensity_scale'), test_container.coords_nominal],
        batch_size=32,
        verbose=1
    )
    
    # Reassemble patches
    logger.info("Reassembling PtychoPINN patches...")
    pinn_recon = reassemble_position(pinn_patches, test_container.global_offsets, M=20)

    # Run inference for Baseline
    logger.info("Running inference with Baseline model...")
    # Baseline model outputs amplitude and phase separately
    baseline_output = baseline_model.predict(test_container.X, batch_size=32, verbose=1)
    
    # Handle different output formats
    if isinstance(baseline_output, list) and len(baseline_output) == 2:
        baseline_patches_I, baseline_patches_phi = baseline_output
    else:
        # Some baseline models might output a single array
        raise ValueError("Unexpected baseline model output format")
    
    # Convert to complex representation
    baseline_patches_complex = tf.cast(baseline_patches_I, tf.complex64) * \
                               tf.exp(1j * tf.cast(baseline_patches_phi, tf.complex64))
    
    # Reassemble patches
    logger.info("Reassembling baseline patches...")
    baseline_recon = reassemble_position(baseline_patches_complex, test_container.global_offsets, M=20)

    # Evaluate reconstructions
    pinn_metrics = {}
    baseline_metrics = {}
    cropped_gt = None
    
    # Coordinate-based ground truth alignment and evaluation
    if ground_truth_obj is not None:
        logger.info("Performing coordinate-based alignment of ground truth...")
        
        # Squeeze ground truth to 2D
        gt_obj_squeezed = ground_truth_obj.squeeze()
        logger.info(f"Ground truth original shape: {gt_obj_squeezed.shape}")
        
        # Extract scan coordinates from test container  
        scan_coords_yx = None
        if hasattr(test_container, 'global_offsets') and test_container.global_offsets is not None:
            # global_offsets has shape (n_images, 1, 2, 1) with [x, y] coordinates
            scan_coords_xy = np.squeeze(test_container.global_offsets)  # Remove singleton dimensions -> (n_images, 2)
            # Convert from [x, y] to [y, x] format for cropping function
            scan_coords_yx = scan_coords_xy[:, [1, 0]]  # Swap x and y columns
        elif hasattr(test_container, 'coords_nominal') and test_container.coords_nominal is not None:
            scan_coords_xy = test_container.coords_nominal
            scan_coords_yx = scan_coords_xy[:, [1, 0]]  # Swap x and y columns
        else:
            logger.error("No scan coordinates found in test data - cannot perform coordinate-based alignment")
        
        if scan_coords_yx is not None:
            # Use M=20 from reassemble_position to define effective bounds
            M = 20  # Patch size used in reassemble_position
            effective_radius = M // 2
            
            logger.info(f"Using {len(scan_coords_yx)} scan positions for coordinate-based cropping")
            logger.info(f"Patch size M={M}, effective radius: {effective_radius} pixels")
            logger.info(f"Scan coordinate range: y=[{scan_coords_yx[:, 0].min():.1f}, {scan_coords_yx[:, 0].max():.1f}], x=[{scan_coords_yx[:, 1].min():.1f}, {scan_coords_yx[:, 1].max():.1f}]")
            
            # Calculate the reconstruction bounds directly
            min_y, min_x = scan_coords_yx.min(axis=0)
            max_y, max_x = scan_coords_yx.max(axis=0)
            
            # The reconstruction bounds are determined by scan range + effective patch contribution
            recon_start_row = max(0, int(min_y) - effective_radius)
            recon_end_row = min(gt_obj_squeezed.shape[0], int(max_y) + effective_radius)
            recon_start_col = max(0, int(min_x) - effective_radius)
            recon_end_col = min(gt_obj_squeezed.shape[1], int(max_x) + effective_radius)
            
            # Crop ground truth to match reconstruction bounds
            cropped_gt = gt_obj_squeezed[recon_start_row:recon_end_row, recon_start_col:recon_end_col]
            logger.info(f"Cropped ground truth from {gt_obj_squeezed.shape} to {cropped_gt.shape}")
            
            # Crop reconstructions to same area (remove extra dimensions first)
            pinn_recon_2d = np.squeeze(pinn_recon)  # Remove any extra dimensions
            baseline_recon_2d = np.squeeze(baseline_recon)  # Remove any extra dimensions
            
            # The reconstructions should already be at the right size, but let's verify
            logger.info(f"Reconstruction shapes before alignment: PINN {pinn_recon_2d.shape}, Baseline {baseline_recon_2d.shape}")
            
            # In most cases, reconstructions are already the correct size
            # But we'll ensure they match the cropped ground truth exactly
            pinn_recon_aligned = pinn_recon_2d
            baseline_recon_aligned = baseline_recon_2d
            
            # Ensure all arrays have exactly the same shape by center-cropping to smallest
            target_h = min(pinn_recon_aligned.shape[0], baseline_recon_aligned.shape[0], cropped_gt.shape[0])
            target_w = min(pinn_recon_aligned.shape[1], baseline_recon_aligned.shape[1], cropped_gt.shape[1])
            
            def center_crop(img, target_h, target_w):
                h, w = img.shape
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                return img[start_h:start_h + target_h, start_w:start_w + target_w]
            
            pinn_recon_aligned = center_crop(pinn_recon_aligned, target_h, target_w)
            baseline_recon_aligned = center_crop(baseline_recon_aligned, target_h, target_w)
            cropped_gt = center_crop(cropped_gt, target_h, target_w)
            
            logger.info(f"Final evaluation shapes: PINN {pinn_recon_aligned.shape}, Baseline {baseline_recon_aligned.shape}, GT {cropped_gt.shape}")
            
            # Evaluate with aligned arrays (add back dimensions for eval function)
            # eval_reconstruction expects: stitched_obj=(batch, H, W, channels), ground_truth_obj=(H, W, channels)
            pinn_metrics = eval_reconstruction(
                pinn_recon_aligned[None, ..., None],  # (1, H, W, 1)
                cropped_gt[..., None]                 # (H, W, 1) - no batch dimension!
            )
            baseline_metrics = eval_reconstruction(
                baseline_recon_aligned[None, ..., None],  # (1, H, W, 1)
                cropped_gt[..., None]                     # (H, W, 1) - no batch dimension!
            )
            
            # Save metrics
            metrics_path = args.output_dir / "comparison_metrics.csv"
            save_metrics_csv(pinn_metrics, baseline_metrics, metrics_path)
        else:
            # Coordinate extraction failed, set fallback values
            logger.warning("Coordinate extraction failed - using uncropped reconstructions")
            cropped_gt = None
            pinn_recon_aligned = np.squeeze(pinn_recon)
            baseline_recon_aligned = np.squeeze(baseline_recon)
    else:
        logger.warning("No ground truth object found in test data. Skipping metric evaluation.")
        cropped_gt = None
        pinn_recon_aligned = np.squeeze(pinn_recon)
        baseline_recon_aligned = np.squeeze(baseline_recon)

    # Create comparison plot
    plot_path = args.output_dir / "comparison_plot.png"
    create_comparison_plot(pinn_recon_aligned, baseline_recon_aligned, cropped_gt, plot_path)
    
    logger.info("\nComparison complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()