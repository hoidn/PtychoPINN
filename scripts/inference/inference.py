#!/usr/bin/env python
# coding: utf-8
# TODO needs to be updated to use the new-style config dataclasses
# MAYBE only generate the comparison plot when ground truth object is provided
# MAYBE save output to npz file, not just image

"""
Inference script for ptychography reconstruction.

This script loads a trained model and test data, performs inference,
and saves the reconstructed image comparison and optionally a probe visualization.

Usage:
    python inference_script.py --model_prefix <model_prefix> --test_data <test_data_file> [--output_path <output_path>]
                               [--visualize_probe] [--K <K>] [--nsamples <nsamples>]

Arguments:
    --model_prefix: Path prefix for the saved model and its configuration
    --test_data: Path to the .npz file containing test data
    --output_path: Path prefix for saving output files and images (default: './')
    --visualize_probe: Flag to generate and save probe visualization
    --K: Number of nearest neighbors for grouped data generation (default: 7)
    --nsamples: Number of samples for grouped data generation (default: 1)
"""

import argparse
import logging
import os
import sys
import time
import signal
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ptycho import probe, params, train_pinn
from ptycho.model_manager import ModelManager
from ptycho.raw_data import RawData
from ptycho.workflows.components import load_data, setup_configuration, parse_arguments
from ptycho.config.config import InferenceConfig, validate_inference_config, update_legacy_dict

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('inference.log')
                    ])
logger = logging.getLogger(__name__)

# Redirect print statements to logger
print = logger.info

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print(f"Received signal {signum}. Initiating graceful shutdown...")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ptychography Inference Script")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the saved model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--output_dir", type=str, default='inference_outputs',
                       help="Directory for saving output files and images")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    return parser.parse_args()

def setup_inference_configuration(args: argparse.Namespace, yaml_path: str) -> InferenceConfig:
    """Setup inference configuration from arguments and YAML file."""
    base_config = setup_configuration(args, yaml_path)
    
    # Create InferenceConfig from base config and CLI args
    inference_config = InferenceConfig(
        model=base_config.model,
        model_path=Path(args.model_path),
        debug=args.debug,
        output_dir=Path(args.output_dir)
    )
    
    # Validate the configuration
    validate_inference_config(inference_config)
    
    return inference_config


def load_model(model_path: Path) -> tuple:
    """Load the saved model and its configuration."""
    try:
        # Load the model
        model = ModelManager.load_model(str(model_path))
        config = params.cfg  # ModelManager updates global config when loading

        print(f"Successfully loaded model from {model_path}")
        print(f"Model configuration: {config}")

        return model, config

    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

def perform_inference(model: tf.keras.Model, test_data: RawData, config: dict, K: int, nsamples: int) -> tuple:
    """
    Perform inference using the loaded model and test data.

    Args:
        model (tf.keras.Model): The loaded TensorFlow model.
        test_data (RawData): The RawData object containing test data.
        config (dict): The model's configuration dictionary.
        K (int): Number of nearest neighbors for grouped data generation.
        nsamples (int): Number of samples for grouped data generation.

    Returns:
        tuple: (np.ndarray, np.ndarray, np.ndarray, np.ndarray) - Reconstructed amplitude, 
               reconstructed phase, ePIE amplitude, and ePIE phase.

    Raises:
        ValueError: If there's an error during inference.
    """
    from ptycho.nbutils import reconstruct_image, crop_to_non_uniform_region_with_buffer
    try:
        # Set probe guess
        probe.set_probe_guess(None, test_data.probeGuess)

        # Set random seeds
        tf.random.set_seed(45)
        np.random.seed(45)

        # Generate grouped data
        test_dataset = test_data.generate_grouped_data(config['N'], K=K, nsamples=nsamples)
        
        # Create PtychoDataContainer
        from ptycho import loader
        test_data_container = loader.load(lambda: test_dataset, test_data.probeGuess, which=None, create_split=False)
        
        # Perform reconstruction
        start_time = time.time()
        obj_tensor_full, global_offsets = reconstruct_image(test_data_container, diffraction_to_obj=model)
        reconstruction_time = time.time() - start_time
        print(f"Reconstruction completed in {reconstruction_time:.2f} seconds")

        # Process the reconstructed image
        from ptycho.tf_helper import reassemble_position
        obj_image = reassemble_position(obj_tensor_full, global_offsets, M=20)
        
        # Extract amplitude and phase
        reconstructed_amplitude = np.abs(obj_image)
        reconstructed_phase = np.angle(obj_image)

        # Process ePIE results for comparison
        epie_phase = crop_to_non_uniform_region_with_buffer(np.angle(test_data.objectGuess), buffer=-20)
        epie_amplitude = crop_to_non_uniform_region_with_buffer(np.abs(test_data.objectGuess), buffer=-20)

        print(f"Reconstructed amplitude shape: {reconstructed_amplitude.shape}")
        print(f"Reconstructed phase shape: {reconstructed_phase.shape}")
        print(f"ePIE amplitude shape: {epie_amplitude.shape}")
        print(f"ePIE phase shape: {epie_phase.shape}")

        return reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise ValueError(f"Error during inference: {str(e)}")

def save_comparison_image(reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase, output_path):
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create the comparison figure with a smaller size
        fig, axs = plt.subplots(2, 2, figsize=(4, 4))
        
        # PtychoPINN phase
        im_pinn_phase = axs[0, 0].imshow(reconstructed_phase, cmap='gray')
        axs[0, 0].set_title('PtychoPINN Phase')
        fig.colorbar(im_pinn_phase, ax=axs[0, 0], fraction=0.046, pad=0.04)
        
        # ePIE phase
        im_epie_phase = axs[0, 1].imshow(epie_phase, cmap='gray')
        axs[0, 1].set_title('ePIE Phase')
        fig.colorbar(im_epie_phase, ax=axs[0, 1], fraction=0.046, pad=0.04)
        
        # PtychoPINN amplitude
        im_pinn_amp = axs[1, 0].imshow(reconstructed_amplitude, cmap='viridis')
        axs[1, 0].set_title('PtychoPINN Amplitude')
        fig.colorbar(im_pinn_amp, ax=axs[1, 0], fraction=0.046, pad=0.04)
        
        # ePIE amplitude
        im_epie_amp = axs[1, 1].imshow(epie_amplitude, cmap='viridis')
        axs[1, 1].set_title('ePIE Amplitude')
        fig.colorbar(im_epie_amp, ax=axs[1, 1], fraction=0.046, pad=0.04)
        
        # Remove axis ticks
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Adjust layout with specific padding
        plt.tight_layout(pad=1.5)
        
        # Save the figure with adjusted DPI and ensuring the entire figure is saved
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)

        print(f"Comparison image saved to: {output_path}")

    except Exception as e:
        print(f"Error saving comparison image: {str(e)}")

def save_probe_visualization(test_data: RawData, output_path: str):
    """
    Generate and save the probe visualization.

    Args:
        test_data (RawData): The RawData object containing test data.
        output_path (str): Path to save the probe visualization.

    Raises:
        OSError: If there's an error creating the output directory or saving the image.
    """
    from ptycho.nbutils import probeshow
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate the probe visualization
        fig = probeshow(test_data.probeGuess, test_data)
        
        # Save the figure
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Probe visualization saved to: {output_path}")

    except OSError as e:
        raise OSError(f"Error saving probe visualization: {str(e)}")

def main(model_prefix: str, test_data_file: str, output_path: str, visualize_probe: bool, K: int, nsamples: int) -> None:
    """
    Main function to orchestrate the inference process.

    Args:
        model_prefix (str): Path prefix for the saved model and its configuration.
        test_data_file (str): Path to the .npz file containing test data.
        output_path (str): Path prefix for saving output files and images.
        visualize_probe (bool): Flag to generate and save probe visualization.
        K (int): Number of nearest neighbors for grouped data generation.
        nsamples (int): Number of samples for grouped data generation.

    Raises:
        Exception: If any error occurs during the inference process.
    """
    print("Starting inference process...")
    start_time = time.time()

    try:
        # Load model
        print("Loading model...")
        model, config = load_model(model_prefix)

        # Load test data
        print("Loading test data...")
        test_data = load_data(test_data_file)

        # Check for shutdown request
        if shutdown_requested:
            print("Shutdown requested. Stopping inference process.")
            return

        # Perform inference
        print(f"Performing inference with K={K} and nsamples={nsamples}...")
        reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase = perform_inference(model, test_data, config, K, nsamples)

        # Check for shutdown request
        if shutdown_requested:
            print("Shutdown requested. Stopping before saving results.")
            return

        # Save comparison image
        print("Saving comparison image...")
        output_image_path = os.path.join(output_path, "reconstruction_comparison.png")
        save_comparison_image(reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase, output_image_path)

        # Save probe visualization if requested
        if visualize_probe:
            print("Generating and saving probe visualization...")
            probe_output_path = os.path.join(output_path, "probe_visualization.png")
            save_probe_visualization(test_data, probe_output_path)

        print("Inference process completed successfully.")

    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        raise
    except ValueError as e:
        print(f"Value error: {str(e)}")
        raise
    except OSError as e:
        print(f"OS error: {str(e)}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise
    finally:
        # Perform any necessary cleanup
        print("Cleaning up resources...")
        tf.keras.backend.clear_session()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        print("Starting ptychography inference script...")
        args = parse_arguments()
        config = setup_inference_configuration(args, args.config)
        
        # Update global params with new-style config
        update_legacy_dict(params.cfg, config)

        # Load model
        print("Loading model...")
        model, _ = load_model(config.model_path)

        # Load test data
        print("Loading test data...")
        test_data = load_data(args.test_data)

        # Perform inference
        print("Performing inference...")
        # TODO might want to reduce K
        reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase = perform_inference(
            model, test_data, params.cfg, K=7, nsamples=1)

        # Save comparison image
        print("Saving comparison image...")
        output_image_path = config.output_dir / "reconstruction_comparison.png"
        save_comparison_image(reconstructed_amplitude, reconstructed_phase, 
                            epie_amplitude, epie_phase, output_image_path)

        print("Inference process completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Script execution failed: {str(e)}")
        sys.exit(1)
    finally:
        print("Cleaning up resources...")
        tf.keras.backend.clear_session()
