#!/usr/bin/env python3
"""
Standalone Tike Reconstruction Script

This script performs ptychographic reconstruction using the Tike iterative algorithm
and saves the results in a standardized NPZ format with rich metadata for use in
model comparison studies.

Usage:
    python run_tike_reconstruction.py <input_npz> <output_dir> [options]

The script produces:
- tike_reconstruction.npz: Standardized output with reconstructed object/probe and metadata
- reconstruction_visualization.png: 2x2 plot showing amplitude/phase of results
"""

import argparse
import logging
import time
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tike.ptycho
import tike.precision
import tike

# Add ptycho to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ptycho.log_config import setup_logging
from ptycho.cli_args import add_logging_arguments, get_logging_config


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone Tike ptychographic reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tike_reconstruction.py input.npz ./tike_output
    python run_tike_reconstruction.py input.npz ./tike_output --iterations 500 --quiet
        """
    )
    
    # Required positional arguments
    parser.add_argument(
        'input_npz',
        help='Input NPZ file containing diffraction, probeGuess, xcoords, ycoords'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for reconstruction results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of reconstruction iterations (default: 1000)'
    )
    parser.add_argument(
        '--num-gpu',
        type=int,
        default=1,
        help='Number of GPUs to use (default: 1)'
    )
    
    # Add standard logging arguments
    add_logging_arguments(parser)
    
    return parser.parse_args()


def load_tike_data(npz_path):
    """
    Load necessary arrays from input NPZ file for Tike reconstruction.
    
    Args:
        npz_path: Path to input NPZ file
        
    Returns:
        dict: Dictionary containing diffraction, probe, xcoords, ycoords arrays
        
    Raises:
        FileNotFoundError: If NPZ file doesn't exist
        KeyError: If required keys are missing from NPZ file
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Input NPZ file not found: {npz_path}")
    
    logger.info(f"Loading data from {npz_path}")
    
    with np.load(npz_path) as data:
        # Handle potential key name variations
        diffraction_keys = ['diffraction', 'diff3d']
        diffraction = None
        for key in diffraction_keys:
            if key in data:
                diffraction = data[key]
                logger.debug(f"Found diffraction data under key: {key}")
                break
        
        if diffraction is None:
            raise KeyError(f"No diffraction data found. Expected one of: {diffraction_keys}")
        
        # Required arrays
        required_keys = ['probeGuess', 'xcoords', 'ycoords']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")
        
        result = {
            'diffraction': diffraction,
            'probeGuess': data['probeGuess'],
            'xcoords': data['xcoords'],
            'ycoords': data['ycoords']
        }
        
        logger.info(f"Loaded data shapes:")
        for key, arr in result.items():
            logger.info(f"  {key}: {arr.shape} ({arr.dtype})")
    
    return result


def configure_tike_parameters(data_dict, iterations, num_gpu):
    """
    Configure Tike reconstruction parameters based on loaded data.
    
    Args:
        data_dict: Dictionary from load_tike_data()
        iterations: Number of iterations to perform
        num_gpu: Number of GPUs to use
        
    Returns:
        tuple: (data, tike.ptycho.PtychoParameters) configured for reconstruction
    """
    logger = logging.getLogger(__name__)
    logger.info("Configuring Tike parameters...")
    
    # Extract and convert data types
    diffraction = data_dict['diffraction'].astype(tike.precision.floating)
    probe = data_dict['probeGuess'].astype(tike.precision.cfloating)
    
    # Prepare scan coordinates (stack y, x)
    scan = np.stack([
        data_dict['ycoords'].astype(tike.precision.floating),
        data_dict['xcoords'].astype(tike.precision.floating)
    ], axis=1)
    
    # Add batch and other dimensions to probe as expected by Tike
    probe = probe[np.newaxis, np.newaxis, np.newaxis, :, :]
    
    # Create padded object using Tike's automatic padding
    psi_2d, scan = tike.ptycho.object.get_padded_object(
        scan=scan,
        probe=probe,
    )
    psi = psi_2d[np.newaxis, :, :]
    
    logger.info(f"Created padded object with shape: {psi.shape}")
    logger.info(f"Updated scan positions shape: {scan.shape}")
    
    # Configure algorithm options
    algorithm_options = tike.ptycho.RpieOptions(
        num_iter=iterations,
        num_batch=10,  # Default batch size
    )
    
    # Configure object options with adaptive moment
    object_options = tike.ptycho.ObjectOptions(
        use_adaptive_moment=True
    )
    
    # Configure probe options with robust settings
    probe_options = tike.ptycho.ProbeOptions(
        use_adaptive_moment=True,
        probe_support=0.05,
        force_centered_intensity=True,
    )
    
    # Position options - keep positions fixed for stability
    position_options = None
    
    # Configure exitwave options with Poisson noise model
    exitwave_options = tike.ptycho.ExitWaveOptions(
        measured_pixels=np.ones_like(diffraction[0], dtype=bool),
        noise_model="poisson",
    )
    
    # Assemble parameters
    parameters = tike.ptycho.PtychoParameters(
        psi=psi,
        probe=probe,
        scan=scan,
        algorithm_options=algorithm_options,
        object_options=object_options,
        probe_options=probe_options,
        position_options=position_options,
        exitwave_options=exitwave_options,
    )
    
    logger.info("Tike parameters configured successfully")
    
    return diffraction, parameters


def save_tike_results(result, output_dir, metadata_dict):
    """
    Save Tike reconstruction results in standardized NPZ format.
    
    Args:
        result: Tike reconstruction result object
        output_dir: Output directory path
        metadata_dict: Dictionary containing reconstruction metadata
    """
    logger = logging.getLogger(__name__)
    
    # Extract reconstructed arrays (remove batch dimensions)
    reconstructed_object = result.psi[0]  # Remove batch dimension
    reconstructed_probe = result.probe[0, 0, 0, :, :]  # Remove all extra dimensions
    
    # Prepare output file path
    output_file = os.path.join(output_dir, 'tike_reconstruction.npz')
    
    # Save in standardized format
    np.savez_compressed(
        output_file,
        reconstructed_object=reconstructed_object,
        reconstructed_probe=reconstructed_probe,
        metadata=np.array([metadata_dict])  # Single-element object array
    )
    
    logger.info(f"Saved reconstruction results to {output_file}")
    
    # Log array information for verification
    logger.debug(f"Saved arrays:")
    logger.debug(f"  reconstructed_object: {reconstructed_object.shape} ({reconstructed_object.dtype})")
    logger.debug(f"  reconstructed_probe: {reconstructed_probe.shape} ({reconstructed_probe.dtype})")
    
    return output_file


def save_visualization(result, output_dir):
    """
    Generate and save visualization of reconstruction results.
    
    Args:
        result: Tike reconstruction result object
        output_dir: Output directory path
        
    Returns:
        str: Path to saved visualization file
    """
    logger = logging.getLogger(__name__)
    
    # Extract arrays for visualization
    reconstructed_object = result.psi[0]
    reconstructed_probe = result.probe[0, 0, 0, :, :]
    
    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Object amplitude
    ax = axes[0, 0]
    im = ax.imshow(np.abs(reconstructed_object), cmap='gray')
    ax.set_title('Reconstructed Object Amplitude')
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    # Object phase
    ax = axes[0, 1]
    im = ax.imshow(np.angle(reconstructed_object), cmap='twilight')
    ax.set_title('Reconstructed Object Phase')
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    # Probe amplitude
    ax = axes[1, 0]
    im = ax.imshow(np.abs(reconstructed_probe), cmap='gray')
    ax.set_title('Reconstructed Probe Amplitude')
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    # Probe phase
    ax = axes[1, 1]
    im = ax.imshow(np.angle(reconstructed_probe), cmap='twilight')
    ax.set_title('Reconstructed Probe Phase')
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    # Save visualization
    vis_file = os.path.join(output_dir, 'reconstruction_visualization.png')
    plt.savefig(vis_file, dpi=150, bbox_inches='tight')
    plt.close()  # Close to prevent blocking
    
    logger.info(f"Saved visualization to {vis_file}")
    
    return vis_file


def main():
    """Main entry point for the Tike reconstruction script."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up centralized logging
    logging_config = get_logging_config(args) if hasattr(args, 'quiet') else {}
    setup_logging(output_dir, **logging_config)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Tike reconstruction...")
    logger.info(f"Input: {args.input_npz}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"GPUs: {args.num_gpu}")
    
    try:
        # Load data
        data_dict = load_tike_data(args.input_npz)
        
        # Configure parameters
        diffraction, parameters = configure_tike_parameters(
            data_dict, args.iterations, args.num_gpu
        )
        
        # Run reconstruction with timing
        logger.info("Starting Tike reconstruction...")
        start_time = time.time()
        
        result = tike.ptycho.reconstruct(
            data=diffraction,
            parameters=parameters,
            num_gpu=args.num_gpu,
        )
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        logger.info(f"Reconstruction completed in {computation_time:.2f} seconds")
        
        # Prepare metadata
        metadata = {
            'algorithm': 'tike',
            'tike_version': tike.__version__,
            'iterations': args.iterations,
            'computation_time_seconds': computation_time,
            'parameters': {
                'num_gpu': args.num_gpu,
                'batch_size': 10,
                'noise_model': 'poisson',
                'use_adaptive_moment': True,
                'force_centered_intensity': True,
            },
            'input_file': str(Path(args.input_npz).resolve()),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save results
        npz_file = save_tike_results(result, output_dir, metadata)
        vis_file = save_visualization(result, output_dir)
        
        logger.info("Reconstruction completed successfully!")
        logger.info(f"Output files:")
        logger.info(f"  NPZ data: {npz_file}")
        logger.info(f"  Visualization: {vis_file}")
        
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()