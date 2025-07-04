import numpy as np
import tike.ptycho
import tike.precision
import matplotlib.pyplot as plt
import logging
import sys
import os

# Add tools directory to path for update_tool import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
from update_tool import update_object_guess

logging.basicConfig(level=logging.INFO)

def main():
    """Run a full reconstruction with automatic padding, probe centering, and Poisson noise model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Tike ptychographic reconstruction')
    parser.add_argument('input_npz', nargs='?', default='fly/fly001.npz',
                        help='Input NPZ file path (default: fly/fly001.npz)')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory (default: tike_outputs/<dataset_name>)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of iterations (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size (default: 10)')
    
    args = parser.parse_args()
    npz_file = args.input_npz
    
    # Set up output directory
    if args.output_dir is None:
        dataset_name = os.path.splitext(os.path.basename(npz_file))[0]
        output_dir = os.path.join('tike_outputs', dataset_name)
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
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
        num_iter=args.iterations,
        num_batch=args.batch_size,
    )
    object_options = tike.ptycho.ObjectOptions(use_adaptive_moment=True)
    
    probe_options = tike.ptycho.ProbeOptions(
        use_adaptive_moment=True,
        probe_support=0.05,
        force_centered_intensity=True,
    )
    
    # Keep positions fixed for now
    position_options = None

    # --- SOLUTION: USE A MORE STABLE POISSON NOISE MODEL ---
    exitwave_options = tike.ptycho.ExitWaveOptions(
        measured_pixels=np.ones_like(data[0], dtype=bool),
        noise_model="poisson", # Change from default 'gaussian' to 'poisson'
    )
    # --------------------------------------------------------

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

    print("Starting final reconstruction...")
    result = tike.ptycho.reconstruct(
        data=data,
        parameters=parameters,
        num_gpu=1,
    )
    print("Reconstruction finished.")

    # --- Visualization ---
    reconstructed_psi = result.psi[0]
    reconstructed_probe = result.probe[0, 0, 0, :, :]
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
    vis_file = os.path.join(output_dir, 'reconstruction_visualization.png')
    plt.savefig(vis_file)
    print(f"Saved visualization to {vis_file}")
    plt.close()  # Close instead of show to prevent blocking
    
    # Save reconstruction data in appropriate formats
    print("\nSaving reconstruction data...")
    
    # 1. Save updated NPZ with reconstructed object
    dataset_name = os.path.splitext(os.path.basename(npz_file))[0]
    output_npz = os.path.join(output_dir, f"{dataset_name}_reconstructed.npz")
    try:
        update_object_guess(npz_file, result.psi, output_npz)
        print(f"✓ Created {output_npz} with reconstructed object")
    except Exception as e:
        print(f"✗ Error creating updated NPZ: {e}")
    
    # 2. Save individual numpy arrays for flexibility
    # Save object (2D format)
    object_file = os.path.join(output_dir, "object.npy")
    np.save(object_file, reconstructed_psi)
    print(f"✓ Saved object reconstruction to {object_file}")
    
    # Save probe
    probe_file = os.path.join(output_dir, "probe.npy")
    np.save(probe_file, reconstructed_probe)
    print(f"✓ Saved probe reconstruction to {probe_file}")
    
    # 3. Save as a dictionary for easy loading
    recon_dict_file = os.path.join(output_dir, "reconstruction_full.npz")
    np.savez_compressed(
        recon_dict_file,
        object=reconstructed_psi,
        probe=reconstructed_probe,
        scan_positions=result.scan,
        algorithm='tike-rpie',
        iterations=args.iterations,
        noise_model='poisson'
    )
    print(f"✓ Saved complete reconstruction data to {recon_dict_file}")
    
    # 4. Save metadata
    metadata_file = os.path.join(output_dir, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Tike Reconstruction Metadata\n")
        f.write(f"============================\n")
        f.write(f"Input file: {npz_file}\n")
        f.write(f"Algorithm: RPIE\n")
        f.write(f"Iterations: {args.iterations}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Noise model: Poisson\n")
        f.write(f"Object shape: {reconstructed_psi.shape}\n")
        f.write(f"Probe shape: {reconstructed_probe.shape}\n")
        f.write(f"Number of scan positions: {result.scan.shape[0]}\n")
    print(f"✓ Saved metadata to {metadata_file}")
    
    print("\nReconstruction complete! Output directory: {}")
    print(f"  {output_dir}/")
    print(f"  ├── reconstruction_visualization.png")
    print(f"  ├── {dataset_name}_reconstructed.npz  (updated full dataset)")
    print(f"  ├── object.npy                        (object only)")
    print(f"  ├── probe.npy                         (probe only)")
    print(f"  ├── reconstruction_full.npz           (all results)")
    print(f"  └── metadata.txt                      (parameters used)")

if __name__ == '__main__':
    main()
