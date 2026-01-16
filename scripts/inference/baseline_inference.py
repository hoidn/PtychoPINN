import argparse
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Import ptycho components
from ptycho import params as p
from ptycho.tf_helper import reassemble_position # Import the correct reassembly function
from ptycho.loader import RawData
from ptycho.workflows.components import (
    load_data,
    create_ptycho_data_container,
    logger
)
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict

def parse_baseline_inference_args():
    """Parse command-line arguments for baseline model inference."""
    parser = argparse.ArgumentParser(description="Inference script for the supervised baseline model.")
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to the trained baseline model (.h5 file).")
    parser.add_argument("--test_data_file", type=Path, required=True,
                        help="Path to the .npz file containing the test data.")
    parser.add_argument("--output_dir", type=Path, default=Path("baseline_inference_outputs"),
                        help="Directory to save the reconstructed images and plots.")
    parser.add_argument("--n_images", type=int, default=None,
                        help="Number of images from the test data file to use for inference. Defaults to all.")
    parser.add_argument("--gridsize", type=int, default=1,
                        help="Grid size used during training (e.g., 1 or 2). This affects the model's input shape.")
    parser.add_argument("--no_comparison_plot", action="store_true",
                        help="Disable the generation of the comparison plot with ground truth.")
    parser.add_argument("--N", type=int, default=64, help="Patch size N used during training.")
    return parser.parse_args()


def load_baseline_model(model_path: Path) -> tf.keras.Model:
    """Loads a trained Keras baseline model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    logger.info(f"Loading baseline model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully.")
    return model


def main():
    """Main function to run baseline model inference with unified reassembly."""
    args = parse_baseline_inference_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Configuration ---
    model_config = ModelConfig(N=args.N, gridsize=args.gridsize)
    training_config = TrainingConfig(model=model_config,
                                     train_data_file=Path("dummy_train.npz"),
                                     test_data_file=Path("dummy_test.npz"))
    update_legacy_dict(p.cfg, training_config)
    
    logger.info("--- Starting Baseline Inference with Unified Reassembly---")
    
    # 1. --- Load Model and Data ---
    model = load_baseline_model(args.model_path)
    logger.info(f"Loading test data from: {args.test_data_file}")
    test_data_raw = load_data(str(args.test_data_file), n_images=args.n_images)
    
    # The baseline model's input shape depends on the number of channels (gridsize**2)
    if args.gridsize == 1: n_channels = 1
    elif args.gridsize == 2: n_channels = 4
    else: raise ValueError(f"This script only supports gridsize 1 or 2, but got {args.gridsize}.")
        
    test_container = create_ptycho_data_container(test_data_raw, training_config)
    X_test_in = test_container.X[..., :n_channels]
    
    # Crucially, get the global offsets for the test data
    global_offsets = test_container.global_offsets
    logger.info(f"Input data shape: {X_test_in.shape}, Global offsets shape: {global_offsets.shape}")

    # 2. --- Perform Inference to get patches ---
    logger.info("Performing inference to get reconstructed patches...")
    pred_I_patches, pred_phi_patches = model.predict(X_test_in)
    reconstructed_patches_complex = tf.cast(pred_I_patches, tf.complex64) * tf.exp(1j * tf.cast(pred_phi_patches, tf.complex64))

    # 3. --- UNIFIED REASSEMBLY ---
    logger.info("Reassembling patches using global scan positions...")
    # This is the key change: use the same reassembly method as the PINN.
    # We are treating each predicted patch as the "object" at that scan location.
    # Note: `reassemble_position` expects a 4D tensor (batch, H, W, channels).
    # Since our patches are already (batch, H, W, 1), the shape is correct.
    obj_image = reassemble_position(reconstructed_patches_complex, global_offsets, M=20)
    
    recon_amp = np.abs(obj_image)
    recon_phase = np.angle(obj_image)
    logger.info(f"Reassembled object shape: {obj_image.shape}")
            
    # 4. --- Save Outputs ---
    logger.info(f"Saving outputs to: {args.output_dir}")
    plt.imsave(args.output_dir / "baseline_reconstructed_amplitude.png", np.squeeze(recon_amp), cmap='gray', vmin=np.min(recon_amp), vmax=np.max(recon_amp))
    plt.imsave(args.output_dir / "baseline_reconstructed_phase.png", np.squeeze(recon_phase), cmap='viridis')
    logger.info("Reconstructed amplitude and phase images saved.")

    # 5. --- Optional: Comparison Plot ---
    ground_truth_obj = test_data_raw.objectGuess
    if not args.no_comparison_plot and ground_truth_obj is not None:
        logger.info("Generating comparison plot with ground truth.")
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(np.squeeze(recon_phase), cmap='viridis'); axes[0, 0].set_title("Reconstructed Phase")
        axes[1, 0].imshow(np.squeeze(recon_amp), cmap='gray', vmin=np.min(recon_amp), vmax=np.max(recon_amp)); axes[1, 0].set_title("Reconstructed Amplitude")
        axes[0, 1].imshow(np.angle(ground_truth_obj), cmap='viridis'); axes[0, 1].set_title("Ground Truth Phase")
        axes[1, 1].imshow(np.abs(ground_truth_obj), cmap='gray', vmin=np.min(np.abs(ground_truth_obj)), vmax=np.max(np.abs(ground_truth_obj))); axes[1, 1].set_title("Ground Truth Amplitude")
        for ax in axes.flat: ax.axis('off')
        plt.tight_layout()
        plt.savefig(args.output_dir / "baseline_comparison_plot.png")
        logger.info("Comparison plot saved.")

    logger.info("--- Baseline inference script finished successfully. ---")

if __name__ == '__main__':
    main()
