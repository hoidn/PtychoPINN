#!/usr/bin/env python3
"""
Extended Tike reconstruction script that saves updated NPZ files.

This demonstrates how to use the update_tool module to create new NPZ files
with reconstructed objects while preserving all other data.
"""

import numpy as np
import tike.ptycho
import tike.precision
import matplotlib.pyplot as plt
import logging
import sys
import os

# Add the tools directory to Python path to import update_tool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
from update_tool import update_object_guess

logging.basicConfig(level=logging.INFO)


def main():
    """Run reconstruction and save results to both image and updated NPZ."""
    # Input/output paths
    npz_file = 'fly/fly001.npz'
    output_npz = 'fly/fly001_reconstructed.npz'
    output_image = 'final_reconstruction_with_npz.png'
    
    print(f"Loading data from {npz_file}")
    with np.load(npz_file) as f:
        data = f['diff3d']
        probe = f['probeGuess']
        scan = np.stack([f['ycoords'], f['xcoords']], axis=1)

    print("Preparing parameters for Tike...")
    data = data.astype(tike.precision.floating)
    probe = probe.astype(tike.precision.cfloating)[np.newaxis, np.newaxis, np.newaxis, :, :]
    scan = scan.astype(tike.precision.floating)

    psi_2d, scan = tike.ptycho.object.get_padded_object(
        scan=scan,
        probe=probe,
    )
    psi = psi_2d[np.newaxis, :, :]
    print(f"Created a new, padded object with shape: {psi.shape}")

    print("Configuring reconstruction...")
    algorithm_options = tike.ptycho.RpieOptions(
        num_iter=1000,
        num_batch=10,
    )
    object_options = tike.ptycho.ObjectOptions(use_adaptive_moment=True)
    
    probe_options = tike.ptycho.ProbeOptions(
        use_adaptive_moment=True,
        probe_support=0.05,
        force_centered_intensity=True,
    )
    
    position_options = None

    exitwave_options = tike.ptycho.ExitWaveOptions(
        measured_pixels=np.ones_like(data[0], dtype=bool),
        noise_model="poisson",
    )

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

    print("Starting reconstruction...")
    result = tike.ptycho.reconstruct(
        data=data,
        parameters=parameters,
        num_gpu=1,
    )
    print("Reconstruction finished.")

    # Extract results
    reconstructed_psi = result.psi[0]  # This is 2D after indexing
    reconstructed_probe = result.probe[0, 0, 0, :, :]
    
    # Save visualization
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    ax = axes[0, 0]
    im = ax.imshow(np.abs(reconstructed_psi), cmap='gray')
    ax.set_title('Final Object Amplitude')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[0, 1]
    im = ax.imshow(np.angle(reconstructed_psi), cmap='twilight')
    ax.set_title('Final Object Phase')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1, 0]
    im = ax.imshow(np.abs(reconstructed_probe), cmap='gray')
    ax.set_title('Final Probe Amplitude')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1, 1]
    im = ax.imshow(np.angle(reconstructed_probe), cmap='twilight')
    ax.set_title('Final Probe Phase')
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Saved visualization to {output_image}")
    plt.close()
    
    # Save updated NPZ file with reconstructed object
    print(f"\nUpdating NPZ file with reconstruction...")
    try:
        # Note: result.psi has shape (1, H, W), update_object_guess will handle it
        update_object_guess(npz_file, result.psi, output_npz)
        print(f"Successfully created {output_npz} with reconstructed object!")
        
        # Optionally save the probe as well to a separate file
        probe_output = 'fly/fly001_probe.npy'
        np.save(probe_output, reconstructed_probe)
        print(f"Also saved reconstructed probe to {probe_output}")
        
    except Exception as e:
        print(f"Error updating NPZ: {e}")
        # Save reconstruction as fallback
        fallback_output = 'fly/fly001_object_only.npy'
        np.save(fallback_output, reconstructed_psi)
        print(f"Saved reconstruction to {fallback_output} as fallback")


if __name__ == '__main__':
    main()