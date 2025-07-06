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
from ptycho.nbutils import crop_to_non_uniform_region_with_buffer


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

    # Run inference for PtychoPINN
    logger.info("Running inference with PtychoPINN...")
    # PtychoPINN model requires both diffraction patterns and position coordinates
    pinn_patches = pinn_model.predict(
        [test_container.X, test_container.coords_nominal],
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
    
    if ground_truth_obj is not None:
        logger.info("Evaluating reconstructions against ground truth...")
        
        try:
            # Separate the ground truth into amplitude and phase
            gt_obj_squeezed = ground_truth_obj.squeeze()
            gt_amplitude = np.abs(gt_obj_squeezed)
            gt_phase = np.angle(gt_obj_squeezed)
            
            logger.info(f"Ground truth original shape: {gt_obj_squeezed.shape}")
            
            # Crop ground truth to non-uniform region with buffer=-20
            gt_amplitude_cropped = crop_to_non_uniform_region_with_buffer(gt_amplitude, buffer=-20)
            gt_phase_cropped = crop_to_non_uniform_region_with_buffer(gt_phase, buffer=-20)
            
            # Recombine into complex cropped_gt array
            cropped_gt = gt_amplitude_cropped * np.exp(1j * gt_phase_cropped)
            logger.info(f"Cropped ground truth from {gt_obj_squeezed.shape} to {cropped_gt.shape}")
            
            # Final alignment crop: slice reconstructions to exact shape of cropped ground truth
            target_shape = gt_amplitude_cropped.shape
            pinn_recon_aligned = pinn_recon[:target_shape[0], :target_shape[1]]
            baseline_recon_aligned = baseline_recon[:target_shape[0], :target_shape[1]]
            
            logger.info(f"Aligned reconstruction shapes: PINN {pinn_recon_aligned.shape}, Baseline {baseline_recon_aligned.shape}")
            
            # Evaluate PtychoPINN using cropped_gt
            pinn_metrics = eval_reconstruction(
                pinn_recon_aligned[None, ...], 
                cropped_gt[None, ..., None]
            )
            
            # Evaluate Baseline using cropped_gt
            baseline_metrics = eval_reconstruction(
                baseline_recon_aligned[None, ...], 
                cropped_gt[None, ..., None]
            )
            
            # Save metrics
            metrics_path = args.output_dir / "comparison_metrics.csv"
            save_metrics_csv(pinn_metrics, baseline_metrics, metrics_path)
            
        except Exception as e:
            logger.warning(f"Could not crop ground truth for evaluation: {e}")
            logger.warning("Skipping metric evaluation.")
            cropped_gt = None
    else:
        logger.warning("No ground truth object found in test data. Skipping metric evaluation.")

    # Create comparison plot using cropped_gt (can handle None)
    plot_path = args.output_dir / "comparison_plot.png"
    create_comparison_plot(pinn_recon, baseline_recon, cropped_gt, plot_path)
    
    logger.info("\nComparison complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()