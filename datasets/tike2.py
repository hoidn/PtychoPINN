import numpy as np
import tike.ptycho
import tike.precision
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

def main():
    """Run a full reconstruction with the correct coordinate convention and position correction."""
    npz_file = 'fly/fly001.npz'
    print(f"Loading data from {npz_file}")
    with np.load(npz_file) as f:
        data = f['diff3d']
        psi = f['objectGuess']
        probe = f['probeGuess']
        # Use the winning convention: (y, x)
        scan = np.stack([f['ycoords'], f['xcoords']], axis=1)
    
    print("Preparing parameters for Tike...")
    data = data.astype(tike.precision.floating)
    psi = psi.astype(tike.precision.cfloating)[np.newaxis, :, :]
    probe = probe.astype(tike.precision.cfloating)[np.newaxis, np.newaxis, np.newaxis, :, :]
    
    scan = scan.astype(tike.precision.floating)
    
    # Normalize scan positions to fit within the object array
    scan -= np.min(scan, axis=0)
    scan += 1.0
    
    print("Configuring reconstruction with probe updates and POSITION CORRECTION...")
    algorithm_options = tike.ptycho.RpieOptions(
        num_iter=1000, 
        num_batch=10,
    )
    
    object_options = tike.ptycho.ObjectOptions(use_adaptive_moment=True)
    
    # Enable probe updates for a better reconstruction
    probe_options = tike.ptycho.ProbeOptions(
        use_adaptive_moment=True,
        probe_support=0.05,
    )
    
    # --- ENABLE POSITION CORRECTION ---
    # Configure position correction with adaptive moment optimization
    position_options = tike.ptycho.PositionOptions(
        use_adaptive_moment=True,
        update_magnitude_limit=0.5,  # Limit position update magnitude
        update_start=10,  # Start position updates after 10 iterations
    )
    # -----------------------------------
    
    exitwave_options = tike.ptycho.ExitWaveOptions(
        measured_pixels=np.ones_like(data[0], dtype=bool)
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
    
    print("Starting reconstruction with position correction...")
    result = tike.ptycho.reconstruct(
        data=data,
        parameters=parameters,
        num_gpu=1,
    )
    
    print("Reconstruction finished.")
    
    # --- Visualization ---
    reconstructed_psi = result.psi[0]
    reconstructed_probe = result.probe[0, 0, 0, :, :]
    corrected_scan = result.scan
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Object amplitude
    ax = axes[0, 0]
    im = ax.imshow(np.abs(reconstructed_psi), cmap='gray')
    ax.set_title('Final Object Amplitude')
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    # Object phase
    ax = axes[0, 1]
    im = ax.imshow(np.angle(reconstructed_psi), cmap='twilight')
    ax.set_title('Final Object Phase')
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    # Position correction comparison
    ax = axes[0, 2]
    ax.scatter(scan[:, 1], scan[:, 0], c='red', s=20, alpha=0.7, label='Original')
    ax.scatter(corrected_scan[:, 1], corrected_scan[:, 0], c='blue', s=20, alpha=0.7, label='Corrected')
    ax.set_title('Scan Position Correction')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Probe amplitude
    ax = axes[1, 0]
    im = ax.imshow(np.abs(reconstructed_probe), cmap='gray')
    ax.set_title('Final Probe Amplitude')
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    # Probe phase
    ax = axes[1, 1]
    im = ax.imshow(np.angle(reconstructed_probe), cmap='twilight')
    ax.set_title('Final Probe Phase')
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    # Position correction magnitude
    ax = axes[1, 2]
    position_shifts = np.sqrt(np.sum((corrected_scan - scan)**2, axis=1))
    ax.hist(position_shifts, bins=30, alpha=0.7, edgecolor='black')
    ax.set_title('Position Correction Magnitude')
    ax.set_xlabel('Shift Distance (pixels)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_reconstruction_with_position_correction.png', dpi=150)
    print("Saved final result to final_reconstruction_with_position_correction.png")
    
    # Print position correction statistics
    print(f"\nPosition correction statistics:")
    print(f"Mean shift: {np.mean(position_shifts):.3f} pixels")
    print(f"Max shift: {np.max(position_shifts):.3f} pixels")
    print(f"Std deviation: {np.std(position_shifts):.3f} pixels")
    
    plt.show()

if __name__ == '__main__':
    main()
