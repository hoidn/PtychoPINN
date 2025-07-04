#!/usr/bin/env python
# scripts/simulation/simulate_and_save.py

"""
Generates a simulated ptychography dataset and saves it to an NPZ file.

This script can be used in two ways:

1. As a command-line tool:
   ------------------------
   Executes the simulation by providing command-line arguments.

   Example:
       python scripts/simulation/simulate_and_save.py \\
           --input-file /path/to/probe_and_object.npz \\
           --output-file /path/to/simulated_data.npz \\
           --nimages 1000

2. As an importable Python module:
   -------------------------------
   The core logic is available via the `simulate_and_save` function.

   Example:
       from scripts.simulation.simulate_and_save import simulate_and_save

       simulate_and_save(
           input_file_path='probe_and_object.npz',
           output_file_path='simulated_data.npz',
           nimages=500,
           seed=42
       )
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add the project root to the Python path to allow imports from the `ptycho` library
# This is necessary if running the script from the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho.nongrid_simulation import generate_simulated_data, load_probe_object
import matplotlib.pyplot as plt
import numpy as np

def simulate_and_save(
    input_file_path: str | Path,
    output_file_path: str | Path,
    nimages: int,
    buffer: Optional[float] = None,
    seed: Optional[int] = None,
    gridsize: int = 1,
    visualize: bool = False,
) -> None:
    """
    Loads an object/probe, runs a ptychography simulation, and saves the result.

    Args:
        input_file_path: Path to the input .npz file containing 'objectGuess' and 'probeGuess'.
        output_file_path: Path to save the output simulated data as an .npz file.
        nimages: Number of diffraction patterns (scan positions) to simulate.
        buffer: Border size to avoid when generating random coordinates.
                If None, defaults to 35% of the object's smaller dimension.
        seed: Random seed for reproducibility.
        gridsize: Size of the solution region grid. Use 1 for PINN-style simulation,
                 2+ for traditional grouped ptychography.
        visualize: Whether to generate PNG visualizations of the simulation results.

    Raises:
        FileNotFoundError: If the input file does not exist.
        Exception: For other errors during simulation or saving.
    """
    # Set gridsize parameter before any simulation operations
    from ptycho import params
    original_gridsize = params.cfg['gridsize']
    print(f"Setting gridsize to {gridsize} (was {original_gridsize})")
    if gridsize != original_gridsize:
        params.cfg['gridsize'] = gridsize
    
    # 1. Load the initial object and probe
    print(f"Loading object and probe from: {input_file_path}")
    object_guess, probe_guess = load_probe_object(str(input_file_path))
    print(f"  - Object shape: {object_guess.shape}")
    print(f"  - Probe shape: {probe_guess.shape}")

    # Set default buffer if not provided
    if buffer is None:
        # Use a smaller buffer to allow more object coverage
        # Buffer should be roughly probe_size/2 to avoid edge artifacts
        buffer = max(probe_guess.shape) // 2

    # 2. Generate the simulated data in memory
    print(f"Simulating {nimages} diffraction patterns...")
    print(f"Buffer: {buffer}, Object shape: {object_guess.shape}")
    # generate_simulated_data returns a tuple (RawData, patches). We only need the RawData object.
    raw_data_instance, _ = generate_simulated_data(
        objectGuess=object_guess,
        probeGuess=probe_guess,
        nimages=nimages,
        buffer=buffer,
        random_seed=seed,
        return_patches=True
    )
    print("Simulation complete.")
    
    # Debug: Check scan position variation
    x_range = raw_data_instance.xcoords.max() - raw_data_instance.xcoords.min()
    y_range = raw_data_instance.ycoords.max() - raw_data_instance.ycoords.min()
    print(f"Scan position ranges: X={x_range:.2f}, Y={y_range:.2f}")
    
    # Debug: Check if diffraction patterns are actually different
    diff_var = np.var([raw_data_instance.diff3d[i].mean() for i in range(min(10, nimages))])
    print(f"Diffraction pattern mean variance (first 10): {diff_var:.6f}")

    # 3. Save the RawData object to the specified output file
    # Ensure the output directory exists
    output_dir = Path(output_file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving simulated data to: {output_file_path}")
    raw_data_instance.to_file(str(output_file_path))
    print("File saved successfully.")

    # 4. Generate visualizations if requested
    if visualize:
        print("Generating visualization plots...")
        visualize_simulation_results(
            object_guess=object_guess,
            probe_guess=probe_guess,
            raw_data_instance=raw_data_instance,
            output_file_path=output_file_path
        )


def visualize_simulation_results(
    object_guess: np.ndarray,
    probe_guess: np.ndarray,
    raw_data_instance,
    output_file_path: str | Path
) -> None:
    """
    Create comprehensive visualizations of simulation results.
    
    Args:
        object_guess: Original object used for simulation
        probe_guess: Original probe used for simulation
        raw_data_instance: RawData object containing simulation results
        output_file_path: Base path for output files (will be modified for PNG)
    """
    # Create output directory for visualizations
    base_path = Path(output_file_path)
    output_dir = base_path.parent
    base_name = base_path.stem
    
    # Create a comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Input Object (amplitude and phase)
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(np.abs(object_guess), cmap='viridis')
    ax1.set_title('Input Object Amplitude')
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(np.angle(object_guess), cmap='twilight')
    ax2.set_title('Input Object Phase')
    ax2.set_xlabel('Pixels')
    ax2.set_ylabel('Pixels')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 2. Input Probe (amplitude and phase)
    ax3 = plt.subplot(3, 4, 3)
    im3 = ax3.imshow(np.abs(probe_guess), cmap='viridis')
    ax3.set_title('Input Probe Amplitude')
    ax3.set_xlabel('Pixels')
    ax3.set_ylabel('Pixels')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = plt.subplot(3, 4, 4)
    im4 = ax4.imshow(np.angle(probe_guess), cmap='twilight')
    ax4.set_title('Input Probe Phase')
    ax4.set_xlabel('Pixels')
    ax4.set_ylabel('Pixels')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # 3. Scan positions
    ax5 = plt.subplot(3, 4, 5)
    ax5.scatter(raw_data_instance.xcoords, raw_data_instance.ycoords, 
                alpha=0.6, s=20, c='blue')
    ax5.set_title(f'Scan Positions (n={len(raw_data_instance.xcoords)})')
    ax5.set_xlabel('X Coordinate')
    ax5.set_ylabel('Y Coordinate')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 4. Sample diffraction patterns (first 3)
    for i in range(min(3, raw_data_instance.diff3d.shape[0])):
        ax = plt.subplot(3, 4, 6 + i)
        # Use log scale for better visibility
        diff_pattern = raw_data_instance.diff3d[i]
        im = ax.imshow(np.log1p(diff_pattern), cmap='jet')
        ax.set_title(f'Diffraction Pattern {i+1}\\n(log scale)')
        ax.set_xlabel('Detector Pixels')
        ax.set_ylabel('Detector Pixels')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 5. Data statistics
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    # Calculate statistics
    diff_stats = {
        'Shape': f"{raw_data_instance.diff3d.shape}",
        'Min': f"{raw_data_instance.diff3d.min():.2e}",
        'Max': f"{raw_data_instance.diff3d.max():.2e}",
        'Mean': f"{raw_data_instance.diff3d.mean():.2e}",
        'Std': f"{raw_data_instance.diff3d.std():.2e}"
    }
    
    coord_stats = {
        'X range': f"[{raw_data_instance.xcoords.min():.1f}, {raw_data_instance.xcoords.max():.1f}]",
        'Y range': f"[{raw_data_instance.ycoords.min():.1f}, {raw_data_instance.ycoords.max():.1f}]",
        'N positions': f"{len(raw_data_instance.xcoords)}"
    }
    
    stats_text = "Diffraction Statistics:\\n"
    for key, value in diff_stats.items():
        stats_text += f"  {key}: {value}\\n"
    stats_text += "\\nCoordinate Statistics:\\n"
    for key, value in coord_stats.items():
        stats_text += f"  {key}: {value}\\n"
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 6. Diffraction pattern histogram
    ax10 = plt.subplot(3, 4, 10)
    diff_flat = raw_data_instance.diff3d.flatten()
    ax10.hist(diff_flat[diff_flat > 0], bins=50, alpha=0.7, edgecolor='black')
    ax10.set_xlabel('Intensity')
    ax10.set_ylabel('Count')
    ax10.set_title('Diffraction Intensity Distribution')
    ax10.set_yscale('log')
    
    # 7. Coordinate distribution
    ax11 = plt.subplot(3, 4, 11)
    ax11.hist2d(raw_data_instance.xcoords, raw_data_instance.ycoords, 
                bins=30, cmap='Blues', alpha=0.8)
    ax11.set_xlabel('X Coordinate')
    ax11.set_ylabel('Y Coordinate')
    ax11.set_title('Coordinate Density')
    
    # 8. Sample of illuminated patches (if available)
    if hasattr(raw_data_instance, 'Y') and raw_data_instance.Y is not None:
        ax12 = plt.subplot(3, 4, 12)
        # Show amplitude of first illuminated patch
        patch_amp = np.abs(raw_data_instance.Y[0])
        im12 = ax12.imshow(patch_amp, cmap='viridis')
        ax12.set_title('Sample Illuminated Patch\\n(Amplitude)')
        ax12.set_xlabel('Pixels')
        ax12.set_ylabel('Pixels')
        plt.colorbar(im12, ax=ax12, fraction=0.046)
    else:
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        ax12.text(0.5, 0.5, 'Illuminated patches\\nnot available', 
                  transform=ax12.transAxes, ha='center', va='center',
                  fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the visualization
    viz_path = output_dir / f"{base_name}_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved visualization to {viz_path}")
    
    # Create a second figure for diffraction pattern montage
    n_samples = min(16, raw_data_instance.diff3d.shape[0])
    if n_samples > 0:
        fig2, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n_samples):
            ax = axes[i]
            diff_pattern = raw_data_instance.diff3d[i]
            im = ax.imshow(np.log1p(diff_pattern), cmap='jet')
            ax.set_title(f'Pattern {i+1}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(n_samples, 16):
            axes[i].axis('off')
        
        plt.suptitle(f'Diffraction Pattern Montage (log scale)\\nFirst {n_samples} patterns', fontsize=14)
        plt.tight_layout()
        
        montage_path = output_dir / f"{base_name}_diffraction_montage.png"
        plt.savefig(montage_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        print(f"✓ Saved diffraction montage to {montage_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the simulation script."""
    parser = argparse.ArgumentParser(
        description="Generate and save a simulated ptychography dataset."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input .npz file containing 'objectGuess' and 'probeGuess'."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the output simulated data as an .npz file."
    )
    parser.add_argument(
        "--nimages",
        type=int,
        default=2000,
        help="Number of diffraction patterns (scan positions) to simulate. Default: 2000."
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=None,
        help="Border size for random coordinates. Defaults to 35%% of the object's smaller dimension."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default: None."
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=1,
        help="Size of the solution region grid. Use 1 for PINN-style simulation (default), 2+ for traditional grouped ptychography."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate PNG visualizations of the simulation results for visual inspection."
    )
    return parser.parse_args()


def main():
    """Main function to handle command-line execution."""
    args = parse_arguments()
    try:
        simulate_and_save(
            input_file_path=args.input_file,
            output_file_path=args.output_file,
            nimages=args.nimages,
            buffer=args.buffer,
            seed=args.seed,
            gridsize=args.gridsize,
            visualize=args.visualize
        )
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


# This block ensures that main() is only called when the script is executed directly
if __name__ == "__main__":
    main()
