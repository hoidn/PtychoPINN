#!/usr/bin/env python3
"""
Reconstruct TIKE dataset using pty-chi library.
This script handles necessary data format conversions and runs reconstruction.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add pty-chi to path (relative to project root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pty-chi', 'src'))

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size, get_default_complex_dtype

def load_and_convert_tike_data(npz_path, n_images=None):
    """
    Load TIKE dataset and convert to pty-chi format.
    
    Args:
        npz_path: Path to TIKE NPZ file
        n_images: Optional number of images to use (for testing)
    
    Returns:
        Dictionary with converted data
    """
    print(f"Loading TIKE dataset from: {npz_path}")
    
    with np.load(npz_path) as data:
        # Get diffraction data (amplitude)
        if 'diff3d' in data:
            amplitude_data = data['diff3d']
        elif 'diffraction' in data:
            amplitude_data = data['diffraction']
        else:
            raise KeyError("No diffraction data found")
        
        # Get coordinates
        xcoords = data['xcoords']
        ycoords = data['ycoords']
        
        # Get probe and object
        probe_complex = data['probeGuess']
        object_guess = data['objectGuess']
        
        # Optional: limit number of images for testing
        if n_images is not None:
            n_images = min(n_images, amplitude_data.shape[0])
            amplitude_data = amplitude_data[:n_images]
            xcoords = xcoords[:n_images]
            ycoords = ycoords[:n_images]
            print(f"Using first {n_images} images for testing")
    
    # Convert amplitude to intensity
    print("\nConverting amplitude to intensity...")
    intensity_data = amplitude_data ** 2
    
    print(f"  Original amplitude: min={amplitude_data.min():.6f}, max={amplitude_data.max():.6f}, mean={amplitude_data.mean():.6f}")
    print(f"  Converted intensity: min={intensity_data.min():.8f}, max={intensity_data.max():.8f}, mean={intensity_data.mean():.8f}")
    
    # Stack coordinates in [y, x] format for pty-chi
    positions_px = np.stack([ycoords, xcoords], axis=1)
    
    # Zero-center the positions as required by pty-chi
    positions_px_centered = positions_px.copy()
    positions_px_centered[:, 0] -= positions_px[:, 0].mean()  # Center Y
    positions_px_centered[:, 1] -= positions_px[:, 1].mean()  # Center X
    
    print(f"\nPositions shape: {positions_px.shape} (n_images, 2)")
    print(f"  Original X range: [{xcoords.min():.2f}, {xcoords.max():.2f}]")
    print(f"  Original Y range: [{ycoords.min():.2f}, {ycoords.max():.2f}]")
    print(f"  Centered X range: [{positions_px_centered[:, 1].min():.2f}, {positions_px_centered[:, 1].max():.2f}]")
    print(f"  Centered Y range: [{positions_px_centered[:, 0].min():.2f}, {positions_px_centered[:, 0].max():.2f}]")
    
    return {
        'intensity': intensity_data.astype(np.float32),
        'positions_px': positions_px_centered.astype(np.float32),  # Use centered positions
        'probe_guess': probe_complex,
        'object_guess': object_guess,
        'n_images': intensity_data.shape[0]
    }

def configure_reconstruction(data_dict, algorithm='DM', num_epochs=20):
    """
    Configure pty-chi reconstruction options.
    
    Args:
        data_dict: Dictionary from load_and_convert_tike_data
        algorithm: Algorithm to use ('DM', 'LSQML', 'PIE', etc.)
        num_epochs: Number of reconstruction epochs
    
    Returns:
        Configured options object
    """
    print(f"\nConfiguring {algorithm} reconstruction...")
    
    # Create algorithm-specific options
    if algorithm == 'DM':
        options = api.DMOptions()
        options.reconstructor_options.exit_wave_update_relaxation = 1.0
    elif algorithm == 'LSQML':
        options = api.LSQMLOptions()
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
        options.reconstructor_options.batch_size = 96
    elif algorithm == 'PIE':
        options = api.PIEOptions()
        options.object_options.alpha = 0.1
        options.probe_options.alpha = 0.1
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Set diffraction data
    options.data_options.data = torch.from_numpy(data_dict['intensity'])
    
    # Set wavelength and detector parameters (these are estimates)
    options.data_options.wavelength_m = 1e-9  # 1 nm (X-ray)
    options.data_options.detector_pixel_size_m = 1e-6  # 1 micron
    options.data_options.fft_shift = True
    
    # Configure object
    probe_shape = data_dict['probe_guess'].shape
    object_size = get_suggested_object_size(
        data_dict['positions_px'], 
        probe_shape, 
        extra=50
    )
    
    print(f"  Object size: {object_size}")
    
    # Initialize object from guess if available
    if data_dict['object_guess'] is not None:
        # Use the existing object guess
        obj_guess = data_dict['object_guess']
        # Ensure it's the right size
        if obj_guess.shape != tuple(object_size):
            # Pad or crop to match expected size
            obj_tensor = torch.zeros(object_size, dtype=get_default_complex_dtype())
            min_h = min(obj_guess.shape[0], object_size[0])
            min_w = min(obj_guess.shape[1], object_size[1])
            obj_tensor[:min_h, :min_w] = torch.from_numpy(obj_guess[:min_h, :min_w])
        else:
            obj_tensor = torch.from_numpy(obj_guess)
        
        options.object_options.initial_guess = obj_tensor.unsqueeze(0)  # Add batch dimension
    else:
        options.object_options.initial_guess = torch.ones(
            [1, *object_size], 
            dtype=get_default_complex_dtype()
        )
    
    options.object_options.pixel_size_m = 1e-8  # 10 nm
    options.object_options.optimizable = True
    
    # Configure probe
    probe_tensor = torch.from_numpy(data_dict['probe_guess'])
    
    # Ensure probe has 4 dimensions: (n_opr_modes, n_modes, h, w)
    if probe_tensor.ndim == 2:
        # Add both n_opr_modes and n_modes dimensions
        probe_tensor = probe_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, h, w)
    elif probe_tensor.ndim == 3:
        # Add n_opr_modes dimension
        probe_tensor = probe_tensor.unsqueeze(0)  # Shape: (1, n_modes, h, w)
    
    options.probe_options.initial_guess = probe_tensor
    options.probe_options.optimizable = True
    
    # Set probe power constraint based on data
    options.probe_options.power_constraint.enabled = True
    total_intensity = np.sum(data_dict['intensity'], axis=(-2, -1))
    options.probe_options.power_constraint.probe_power = np.median(total_intensity)
    
    print(f"  Probe power constraint: {options.probe_options.power_constraint.probe_power:.6f}")
    
    # Configure positions
    options.probe_position_options.position_x_px = data_dict['positions_px'][:, 1]  # X is column 1
    options.probe_position_options.position_y_px = data_dict['positions_px'][:, 0]  # Y is column 0
    options.probe_position_options.optimizable = False  # Can enable for position correction
    
    # Reconstruction parameters
    options.reconstructor_options.num_epochs = num_epochs

    # Set chunk_length only for algorithms that support it (DM, PIE)
    if algorithm in ['DM', 'PIE']:
        options.reconstructor_options.chunk_length = min(100, data_dict['n_images'])

    # Device selection
    if torch.cuda.is_available():
        options.reconstructor_options.default_device = api.Devices.GPU
        print("  Using GPU acceleration")
    else:
        options.reconstructor_options.default_device = api.Devices.CPU
        print("  Using CPU (slower)")
    
    return options

def run_reconstruction(options):
    """
    Run the pty-chi reconstruction.
    
    Args:
        options: Configured options object
    
    Returns:
        PtychographyTask object with results
    """
    print("\n=== Starting Reconstruction ===")
    
    # Create and run task
    task = PtychographyTask(options)
    
    print(f"Running {options.reconstructor_options.num_epochs} epochs...")
    task.run()
    
    print("Reconstruction complete!")
    
    return task

def save_results(task, output_dir="ptychi_reconstruction"):
    """
    Save reconstruction results.
    
    Args:
        task: Completed PtychographyTask
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n=== Saving Results to {output_dir} ===")
    
    # Get reconstructed data
    reconstructed_object = task.get_data_to_cpu("object", as_numpy=True)[0]  # Remove batch dimension
    reconstructed_probe = task.get_data_to_cpu("probe", as_numpy=True)
    
    # Extract 2D probe from potentially 4D array (n_opr_modes, n_modes, h, w)
    if reconstructed_probe.ndim == 4:
        reconstructed_probe = reconstructed_probe[0, 0]  # Take first OPR mode and first mode
    elif reconstructed_probe.ndim == 3:
        reconstructed_probe = reconstructed_probe[0]  # Take first mode
    
    # Save as NPZ
    npz_path = output_path / "ptychi_reconstruction.npz"
    np.savez_compressed(
        npz_path,
        reconstructed_object=reconstructed_object,
        reconstructed_probe=reconstructed_probe,
        algorithm='pty-chi',
        timestamp=datetime.now().isoformat()
    )
    print(f"Saved NPZ: {npz_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Object amplitude
    im = axes[0, 0].imshow(np.abs(reconstructed_object), cmap='gray')
    axes[0, 0].set_title('Reconstructed Object Amplitude')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Object phase
    im = axes[0, 1].imshow(np.angle(reconstructed_object), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Reconstructed Object Phase')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Probe amplitude
    im = axes[1, 0].imshow(np.abs(reconstructed_probe), cmap='gray')
    axes[1, 0].set_title('Reconstructed Probe Amplitude')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Probe phase
    im = axes[1, 1].imshow(np.angle(reconstructed_probe), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Reconstructed Probe Phase')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_path / "reconstruction_visualization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {fig_path}")
    
    # Print statistics
    print("\nReconstruction Statistics:")
    print(f"  Object shape: {reconstructed_object.shape}")
    print(f"  Object amplitude range: [{np.abs(reconstructed_object).min():.6f}, {np.abs(reconstructed_object).max():.6f}]")
    print(f"  Probe shape: {reconstructed_probe.shape}")
    print(f"  Probe amplitude range: [{np.abs(reconstructed_probe).min():.6f}, {np.abs(reconstructed_probe).max():.6f}]")
    
    return npz_path

def main(argv=None):
    """Main entry point with CLI argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruct TIKE dataset using pty-chi library"
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("tike_outputs/fly001_reconstructed_final_downsampled/fly001_reconstructed_final_downsampled_data.npz"),
        help="Path to TIKE NPZ file (default: tike_outputs/fly001_reconstructed_final_downsampled/fly001_reconstructed_final_downsampled_data.npz)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ptychi_tike_reconstruction_converged"),
        help="Directory to save reconstruction results (default: ptychi_tike_reconstruction_converged)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="DM",
        choices=["DM", "LSQML", "PIE"],
        help="Reconstruction algorithm to use (default: DM)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of reconstruction epochs (default: 200)"
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=2000,
        help="Number of images to use (default: 2000, use None for all)"
    )

    args = parser.parse_args(argv)

    # Extract arguments
    tike_dataset = args.input_npz
    output_dir = args.output_dir
    algorithm = args.algorithm
    num_epochs = args.num_epochs
    n_images = args.n_images

    print("=== PTY-CHI Reconstruction of TIKE Dataset ===")
    print(f"Dataset: {tike_dataset}")
    print(f"Algorithm: {algorithm}")
    print(f"Epochs: {num_epochs}")
    print(f"Images: {n_images if n_images else 'all'}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load and convert data
        data_dict = load_and_convert_tike_data(tike_dataset, n_images=n_images)

        # Configure reconstruction
        options = configure_reconstruction(data_dict, algorithm=algorithm, num_epochs=num_epochs)

        # Run reconstruction
        task = run_reconstruction(options)

        # Save results
        output_path = save_results(task, output_dir=output_dir)

        print(f"\n✓ Reconstruction complete! Results saved to {output_dir}/")

    except Exception as e:
        print(f"\n✗ Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())