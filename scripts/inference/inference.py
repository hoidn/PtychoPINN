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

from typing import Optional
import argparse
import logging
import os
import sys
import time
import signal
from pathlib import Path
from dataclasses import fields
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ptycho import probe, params
from ptycho.model_manager import ModelManager
from ptycho.raw_data import RawData
from ptycho.workflows.components import load_data, setup_configuration, parse_arguments
from ptycho.config.config import InferenceConfig, ModelConfig, validate_inference_config, update_legacy_dict
from ptycho.log_config import setup_logging

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
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to the test data file")
    parser.add_argument("--config", type=str, required=False, default=None,
                       help="Optional path to YAML configuration file to override defaults")
    parser.add_argument("--output_dir", type=str, default='inference_outputs',
                       help="Directory for saving output files and images")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--comparison_plot", action="store_true",
                       help="Generate original comparison plot (only if ground truth is available)")
    parser.add_argument("--n_images", type=int, required=False, default=None,
                       help="Number of images/groups to process. Interpretation depends on gridsize: "
                            "gridsize=1 means individual images, gridsize>1 means number of groups")
    parser.add_argument("--phase_vmin", type=float, required=False, default=None,
                       help="Minimum value for phase color scale (default: auto)")
    parser.add_argument("--phase_vmax", type=float, required=False, default=None,
                       help="Maximum value for phase color scale (default: auto)")
    return parser.parse_args()

def setup_inference_configuration(args: argparse.Namespace, yaml_path: Optional[str]) -> InferenceConfig:
    """
    Correctly sets up inference configuration by prioritizing YAML file settings.
    """
    from ptycho.config.config import load_yaml_config

    # Start with default ModelConfig values
    model_defaults = {f.name: f.default for f in fields(ModelConfig)}
    
    # Load and merge YAML config if provided
    if yaml_path:
        print(f"Loading configuration from YAML: {yaml_path}")
        yaml_data = load_yaml_config(Path(yaml_path))
        
        # The YAML might have a nested 'model' structure
        model_yaml_config = yaml_data.get('model', {})
        
        # Update defaults with YAML values
        model_defaults.update(model_yaml_config)
        print(f"Loaded gridsize={model_defaults.get('gridsize')} from config.")

    # Create the final ModelConfig object
    final_model_config = ModelConfig(**model_defaults)

    # Create the InferenceConfig object
    inference_config = InferenceConfig(
        model=final_model_config,
        model_path=Path(args.model_path),
        test_data_file=Path(args.test_data),
        debug=args.debug,
        output_dir=Path(args.output_dir)
    )
    
    validate_inference_config(inference_config)
    print(f"Final inference config - gridsize: {inference_config.model.gridsize}")
    return inference_config


def load_model(model_path: Path) -> tuple:
    """Load the saved model and its configuration."""
    try:
        print(f"Attempting to load model from: {model_path}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Check if the path is a directory and contains wts.h5.zip
        model_zip = os.path.join(model_path, "wts.h5")
        if not os.path.exists(f"{model_zip}.zip"):
            raise ValueError(f"Model archive not found at: {model_zip}.zip")
            
        # Load multiple models
        models_dict = ModelManager.load_multiple_models(model_zip)
        
        # Get the diffraction_to_obj model which is what we need for inference
        if 'diffraction_to_obj' not in models_dict:
            raise ValueError("No diffraction_to_obj model found in saved models")
            
        model = models_dict['diffraction_to_obj']
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
        print(f"DEBUG: Using gridsize={config.get('gridsize', 'NOT_SET')} for data generation")
        test_dataset = test_data.generate_grouped_data(config['N'], K=K, nsamples=nsamples)
        
        # Debug: check the shape of the generated data
        if 'diffraction' in test_dataset:
            print(f"DEBUG: Generated diffraction data shape: {test_dataset['diffraction'].shape}")
        if 'Y' in test_dataset and test_dataset['Y'] is not None:
            print(f"DEBUG: Generated Y data shape: {test_dataset['Y'].shape}")
        
        # Create PtychoDataContainer
        from ptycho import loader
        test_data_container = loader.load(lambda: test_dataset, test_data.probeGuess, which=None, create_split=False)
        
        # Debug: check the data container
        print(f"DEBUG: PtychoDataContainer shapes - X (diffraction): {test_data_container.X.shape}, Y: {test_data_container.Y.shape if test_data_container.Y is not None else 'None'}")
        
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

        # Check if ground truth object is available and valid
        has_ground_truth = False
        if hasattr(test_data, 'objectGuess') and test_data.objectGuess is not None:
            # Check if the object is all zeros or very close to zero
            if not np.allclose(test_data.objectGuess, 0, atol=1e-10):
                # Check if the object is uniform (all values are the same)
                obj_complex = test_data.objectGuess
                if not (np.allclose(obj_complex.real, obj_complex.real.flat[0], atol=1e-10) and 
                        np.allclose(obj_complex.imag, obj_complex.imag.flat[0], atol=1e-10)):
                    has_ground_truth = True
                    epie_phase = crop_to_non_uniform_region_with_buffer(np.angle(test_data.objectGuess), buffer=-20)
                    epie_amplitude = crop_to_non_uniform_region_with_buffer(np.abs(test_data.objectGuess), buffer=-20)
#                    epie_phase = np.angle(test_data.objectGuess)
#                    epie_amplitude = np.abs(test_data.objectGuess)
                    print(f"Ground truth available - ePIE amplitude shape: {epie_amplitude.shape}")
                    print(f"Ground truth available - ePIE phase shape: {epie_phase.shape}")
                else:
                    print("Ground truth object is uniform (all same value), skipping ground truth processing")
                    epie_phase = None
                    epie_amplitude = None
            else:
                print("Ground truth object is all zeros, skipping ground truth processing")
                epie_phase = None
                epie_amplitude = None
        else:
            print("No ground truth object available")
            epie_phase = None
            epie_amplitude = None

        print(f"Reconstructed amplitude shape: {reconstructed_amplitude.shape}")
        print(f"Reconstructed phase shape: {reconstructed_phase.shape}")

        return reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise ValueError(f"Error during inference: {str(e)}")

def save_comparison_plot(reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase, output_dir, phase_vmin=None, phase_vmax=None):
    """
    Save a comparison plot of reconstructed and ground truth images.
    
    Args:
        reconstructed_amplitude (np.ndarray): The reconstructed amplitude array
        reconstructed_phase (np.ndarray): The reconstructed phase array
        epie_amplitude (np.ndarray): The ground truth amplitude array or None
        epie_phase (np.ndarray): The ground truth phase array or None
        output_dir (str or Path): Directory to save the output images
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Squeeze any extra dimensions
        reconstructed_amplitude = np.squeeze(reconstructed_amplitude)
        reconstructed_phase = np.squeeze(reconstructed_phase)
        epie_amplitude = np.squeeze(epie_amplitude)
        epie_phase = np.squeeze(epie_phase)
        
        # Create the comparison figure with a smaller size
        fig, axs = plt.subplots(2, 2, figsize=(4, 4))
        
        # PtychoPINN phase
        im_pinn_phase = axs[0, 0].imshow(reconstructed_phase, cmap='gray', vmin=phase_vmin, vmax=phase_vmax)
        axs[0, 0].set_title('PtychoPINN Phase')
        fig.colorbar(im_pinn_phase, ax=axs[0, 0], fraction=0.046, pad=0.04)
        
        # ePIE phase
        im_epie_phase = axs[0, 1].imshow(epie_phase, cmap='gray', vmin=phase_vmin, vmax=phase_vmax)
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
        
        # Save the figure
        comparison_path = os.path.join(output_dir, "comparison_plot.png")
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {comparison_path}")
        
    except Exception as e:
        print(f"Error saving comparison plot: {str(e)}")

def save_reconstruction_images(reconstructed_amplitude, reconstructed_phase, output_dir, phase_vmin=None, phase_vmax=None):
    """
    Save the reconstructed amplitude and phase as separate PNG files.
    
    Args:
        reconstructed_amplitude (np.ndarray): The reconstructed amplitude array
        reconstructed_phase (np.ndarray): The reconstructed phase array
        output_dir (str or Path): Directory to save the output images
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Squeeze any extra dimensions
        reconstructed_amplitude = np.squeeze(reconstructed_amplitude)
        reconstructed_phase = np.squeeze(reconstructed_phase)
        
        print(f"Amplitude array shape: {reconstructed_amplitude.shape}")
        print(f"Phase array shape: {reconstructed_phase.shape}")
        
        # Save amplitude image
        amplitude_path = os.path.join(output_dir, "reconstructed_amplitude.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(reconstructed_amplitude, cmap='gray')
        plt.colorbar()
        plt.savefig(amplitude_path)
        plt.close()
        
        # Save phase image
        phase_path = os.path.join(output_dir, "reconstructed_phase.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(reconstructed_phase, cmap='viridis', vmin=phase_vmin, vmax=phase_vmax)
        plt.colorbar()
        plt.savefig(phase_path)
        plt.close()
        
        print(f"Reconstructed amplitude saved to: {amplitude_path}")
        print(f"Reconstructed phase saved to: {phase_path}")
        
    except Exception as e:
        print(f"Error saving reconstruction images: {str(e)}")

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

def main(model_prefix: str, test_data_file: str, output_path: str, visualize_probe: bool, K: int, nsamples: int, comparison_plot: bool = False) -> None:
    """
    Main function to orchestrate the inference process.

    Args:
        model_prefix (str): Path prefix for the saved model and its configuration.
        test_data_file (str): Path to the .npz file containing test data.
        output_path (str): Path prefix for saving output files and images.
        visualize_probe (bool): Flag to generate and save probe visualization.
        K (int): Number of nearest neighbors for grouped data generation.
        nsamples (int): Number of samples for grouped data generation.
        comparison_plot (bool): Flag to generate comparison plot.

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

        # Save separate reconstruction images
        print("Saving reconstruction images...")
        save_reconstruction_images(reconstructed_amplitude, reconstructed_phase, output_path)
        
        # Generate comparison plot if requested and ground truth is available
        if comparison_plot and epie_amplitude is not None and epie_phase is not None:
            print("Generating comparison plot...")
            save_comparison_plot(reconstructed_amplitude, reconstructed_phase, 
                                epie_amplitude, epie_phase, output_path)
        elif comparison_plot:
            print("Skipping comparison plot generation - ground truth not available")

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

def main():
    """Main entry point for the ptychography inference script."""
    try:
        args = parse_arguments()
        config = setup_inference_configuration(args, args.config)
        
        # Set up centralized logging
        setup_logging(config.output_dir)
        
        print("Starting ptychography inference script...")
        
        # Update global params with new-style config
        update_legacy_dict(params.cfg, config)

        # Load model
        print("Loading model...")
        model, _ = load_model(config.model_path)

        # Load test data
        print("Loading test data...")
        test_data = load_data(args.test_data)

        # Determine number of samples for inference
        gridsize = params.cfg.get('gridsize', 1)
        total_patterns = len(test_data.xcoords)
        
        if args.n_images is not None:
            # User specified number of images/groups
            if gridsize == 1:
                nsamples = min(args.n_images, total_patterns)
                print(f"Inference config: gridsize={gridsize}, using {nsamples} individual patterns (user specified {args.n_images})")
            else:
                max_groups = total_patterns // (gridsize ** 2)
                nsamples = min(args.n_images, max_groups)
                if nsamples == 0:
                    nsamples = 1  # Minimum of 1 group
                print(f"Inference config: gridsize={gridsize}, using {nsamples} groups (≈{nsamples * gridsize**2} total patterns, user specified {args.n_images})")
        else:
            # Default behavior: use full dataset
            if gridsize == 1:
                nsamples = total_patterns
                print(f"Inference config: gridsize={gridsize}, using all {nsamples} individual patterns")
            else:
                nsamples = total_patterns // (gridsize ** 2)
                if nsamples == 0:
                    nsamples = 1  # Minimum of 1 group
                print(f"Inference config: gridsize={gridsize}, using {nsamples} groups (≈{nsamples * gridsize**2} total patterns)")

        # Perform inference
        print("Performing inference...")
        reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase = perform_inference(
            model, test_data, params.cfg, K=4, nsamples=nsamples)

        # Save separate reconstruction images
        print("Saving reconstruction images...")
        save_reconstruction_images(reconstructed_amplitude, reconstructed_phase, config.output_dir, 
                                  phase_vmin=args.phase_vmin, phase_vmax=args.phase_vmax)
        
        # Generate comparison plot if requested and ground truth is available
        if args.comparison_plot and epie_amplitude is not None and epie_phase is not None:
            print("Generating comparison plot...")
            save_comparison_plot(reconstructed_amplitude, reconstructed_phase, 
                                epie_amplitude, epie_phase, config.output_dir,
                                phase_vmin=args.phase_vmin, phase_vmax=args.phase_vmax)
        elif args.comparison_plot:
            print("Skipping comparison plot generation - ground truth not available")

        print("Inference process completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Script execution failed: {str(e)}")
        sys.exit(1)
    finally:
        print("Cleaning up resources...")
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
