#!/usr/bin/env python
# coding: utf-8
# TODO only generate the comparison plot when ground truth object is provided
# TODO save output to npz file, not just image

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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ptycho import tf_helper, probe, loader, params, train_pinn
from ptycho.model_manager import ModelManager
from ptycho import loader, params

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
    parser.add_argument("--model_prefix", type=str, required=True,
                        help="Path prefix for the saved model and its configuration")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to the .npz file containing test data")
    parser.add_argument("--output_path", type=str, default='./',
                        help="Path prefix for saving output files and images")
    parser.add_argument("--visualize_probe", action="store_true",
                        help="Generate and save probe visualization")
    parser.add_argument("--K", type=int, default=7,
                        help="Number of nearest neighbors for grouped data generation")
    parser.add_argument("--nsamples", type=int, default=1,
                        help="Number of samples for grouped data generation")
    return parser.parse_args()

def load_test_data(file_path: str) -> loader.RawData:
    """
    Load test data from a .npz file.

    Args:
        file_path (str): Path to the .npz file containing test data.

    Returns:
        loader.RawData: A RawData object containing the loaded test data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the required arrays are missing from the .npz file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    try:
        data = np.load(file_path)
        required_keys = ['xcoords', 'ycoords', 'xcoords_start', 'ycoords_start', 'diffraction', 'probeGuess', 'objectGuess']
        if not all(key in data for key in required_keys):
            raise ValueError(f"Missing required arrays in the test data file. Required: {required_keys}")

        xcoords = data['xcoords']
        ycoords = data['ycoords']
        xcoords_start = data['xcoords_start']
        ycoords_start = data['ycoords_start']
        diff3d = np.transpose(data['diffraction'], [2, 0, 1])
        probeGuess = data['probeGuess']
        objectGuess = data['objectGuess']
        
        # Create a dummy scan_index (all zeros) with the same length as xcoords
        scan_index = np.zeros(len(xcoords), dtype=int)

        test_data = loader.RawData(xcoords, ycoords, xcoords_start, ycoords_start,
                                   diff3d, probeGuess, scan_index, objectGuess=objectGuess)

        print(f"Loaded test data: {test_data}")
        return test_data

    except Exception as e:
        raise ValueError(f"Error loading test data: {str(e)}")

def load_model(model_prefix: str) -> tuple:
    """
    Load the saved model and its configuration.

    Args:
        model_prefix (str): Path prefix for the saved model and its configuration.

    Returns:
        tuple: (tf.keras.Model, dict) - The loaded TensorFlow model and its configuration.

    Raises:
        FileNotFoundError: If the model files are not found.
        ValueError: If there's an error loading the model.
    """
    try:
        # Construct the base path
        base_path = os.path.join(model_prefix, "wts.h5")
        
        # Define the model name
        model_name = "diffraction_to_obj"

        # Check if the directory exists
        full_path = f"{base_path}_{model_name}"
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model directory not found: {full_path}")

        # Load the model
        model = ModelManager.load_model(base_path, model_name)
        config = params.cfg  # The ModelManager updates the global config when loading

        print(f"Successfully loaded model from {full_path}")
        print(f"Model configuration: {config}")

        return model, config

    except FileNotFoundError as e:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

def perform_inference(model: tf.keras.Model, test_data: loader.RawData, config: dict, K: int, nsamples: int) -> tuple:
    """
    Perform inference using the loaded model and test data.

    Args:
        model (tf.keras.Model): The loaded TensorFlow model.
        test_data (loader.RawData): The RawData object containing test data.
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
        test_data_container = loader.load(lambda: test_dataset, test_data.probeGuess,
                                          which=None, create_split=False)
        
        # Perform reconstruction
        start_time = time.time()
        obj_tensor_full, global_offsets = reconstruct_image(test_data_container, diffraction_to_obj=model)
        reconstruction_time = time.time() - start_time
        print(f"Reconstruction completed in {reconstruction_time:.2f} seconds")

        # Process the reconstructed image
        obj_image = loader.reassemble_position(obj_tensor_full, global_offsets, M=20)
        
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

def save_probe_visualization(test_data: loader.RawData, output_path: str):
    """
    Generate and save the probe visualization.

    Args:
        test_data (loader.RawData): The RawData object containing test data.
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
        test_data = load_test_data(test_data_file)

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
        main(args.model_prefix + '/', args.test_data, args.output_path, args.visualize_probe, args.K, args.nsamples)
        print("Script execution completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Script execution failed: {str(e)}")
        sys.exit(1)
