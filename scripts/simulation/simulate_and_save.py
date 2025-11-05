#!/usr/bin/env python
# scripts/simulation/simulate_and_save.py

"""
Generates a simulated ptychography dataset and saves it to an NPZ file.
Optionally, it can also generate a rich PNG visualization of the simulation.

Example:
    # Run simulation and also create a summary plot with comparisons
    python scripts/simulation/simulate_and_save.py \\
        --input-file /path/to/prepared_data.npz \\
        --output-file /path/to/simulation_output.npz \\
        --visualize
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ptycho components
from ptycho.nongrid_simulation import generate_simulated_data
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p
from ptycho.metadata import MetadataManager
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

def load_data_for_sim(file_path: str, load_all: bool = False) -> tuple:
    """Loads object and probe, and optionally all other data from an NPZ file.

    Uses MetadataManager to handle metadata-bearing NPZ files safely,
    avoiding allow_pickle=False errors.

    Args:
        file_path: Path to NPZ file
        load_all: If True, return all data arrays (excluding metadata)

    Returns:
        Tuple of (objectGuess, probeGuess, all_data) where all_data is None
        if load_all=False, otherwise dict of all arrays (excluding _metadata)

    References:
        - DATA-001 (metadata preservation requirement)
        - MetadataManager.load_with_metadata() for safe metadata handling
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Use MetadataManager to safely load NPZ with metadata
    data_dict, metadata = MetadataManager.load_with_metadata(file_path)

    if 'objectGuess' not in data_dict or 'probeGuess' not in data_dict:
        raise ValueError("The .npz file must contain 'objectGuess' and 'probeGuess'")

    objectGuess = data_dict['objectGuess']
    probeGuess = data_dict['probeGuess']

    if load_all:
        # Return all data arrays (metadata already filtered by MetadataManager)
        return objectGuess, probeGuess, data_dict
    else:
        return objectGuess, probeGuess, None

def simulate_and_save(
    config: TrainingConfig,
    input_file_path: str | Path,
    output_file_path: str | Path,
    original_data_for_vis: Optional[Dict[str, Any]],
    buffer: Optional[float] = None,
    seed: Optional[int] = None,
    visualize: bool = False,
) -> None:
    """
    Loads an object/probe, runs a ptychography simulation, saves the result,
    and optionally generates a visualization.
    """
    update_legacy_dict(p.cfg, config)
    print("--- Configuration Updated for Simulation ---")
    p.print_params()
    print("------------------------------------------\n")
    
    object_guess, probe_guess, _ = load_data_for_sim(str(input_file_path), load_all=False)
    print(f"Loading object and probe from: {input_file_path}")
    print(f"  - Object shape: {object_guess.shape}")
    print(f"  - Probe shape: {probe_guess.shape}")

    if buffer is None:
        buffer = max(probe_guess.shape) // 2

    if seed is not None:
        print(f"Setting random seed to: {seed}")
        np.random.seed(seed)

    print(f"Simulating {config.n_images} diffraction patterns...")
    raw_data_instance, ground_truth_patches = generate_simulated_data(
        config=config,
        objectGuess=object_guess,
        probeGuess=probe_guess,
        buffer=buffer,
        return_patches=True,
    )
    print("Simulation complete.")
    
    output_dir = Path(output_file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- KEY CHANGE: Add objectGuess to the output ---
    # The raw_data_instance from the simulation doesn't contain the ground truth
    # object it was created from. We explicitly add it here before saving.
    raw_data_instance.objectGuess = object_guess
    print("Added source 'objectGuess' to the output dataset for ground truth.")
    # -------------------------------------------------
    
    print(f"Saving simulated data to: {output_file_path}")
    
    # Create comprehensive data dictionary including ground truth patches
    data_dict = {
        'xcoords': raw_data_instance.xcoords,
        'ycoords': raw_data_instance.ycoords,
        'xcoords_start': raw_data_instance.xcoords_start,
        'ycoords_start': raw_data_instance.ycoords_start,
        'diff3d': raw_data_instance.diff3d,
        'probeGuess': raw_data_instance.probeGuess,
        'objectGuess': raw_data_instance.objectGuess,
        'scan_index': raw_data_instance.scan_index,
        'ground_truth_patches': ground_truth_patches
    }
    
    # Create metadata with simulation parameters
    metadata = MetadataManager.create_metadata(
        config=config,
        script_name="simulate_and_save.py",
        input_file=str(input_file_path),
        buffer=buffer,
        seed=seed,
        simulation_type="coordinate_based"
    )
    
    # Save with metadata
    MetadataManager.save_with_metadata(
        str(output_file_path),
        data_dict,
        metadata
    )
    print(f"File saved successfully with metadata (nphotons={config.nphotons}).")

    if visualize:
        print("Generating visualization plot...")
        visualize_simulation_results(
            object_guess=object_guess,
            probe_guess=probe_guess,
            raw_data_instance=raw_data_instance,
            ground_truth_patches=ground_truth_patches,
            original_data_dict=original_data_for_vis,
            output_file_path=output_file_path
        )

def crop_center(img, cropx, cropy):
    """Helper function to crop the center of an image."""
    y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty+cropy, startx:startx+cropx]

def visualize_simulation_results(
    object_guess: np.ndarray,
    probe_guess: np.ndarray,
    raw_data_instance,
    ground_truth_patches,
    original_data_dict: Optional[Dict[str, Any]],
    output_file_path: str | Path
) -> None:
    """
    Creates and saves a comprehensive visualization of simulation results.
    """
    base_path = Path(output_file_path)
    viz_path = base_path.with_name(f"{base_path.stem}_visualization.png")
    
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    plt.suptitle("Ptychography Simulation Summary", fontsize=20)

    # Row 1: Inputs and Scan Positions
    ax = axes[0, 0]
    im = ax.imshow(np.abs(object_guess), cmap='gray')
    ax.set_title(f"Input Object Amp (Shape: {object_guess.shape})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.imshow(np.abs(probe_guess), cmap='gray')
    ax.set_title(f"Input Probe Amp (Shape: {probe_guess.shape})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 2]
    im = ax.imshow(np.abs(object_guess), cmap='gray', alpha=0.7)
    ax.scatter(raw_data_instance.xcoords, raw_data_instance.ycoords, s=5, c='red', alpha=0.5)
    ax.set_title(f"Scan Positions on Object (n={len(raw_data_instance.xcoords)})")
    ax.set_aspect('equal')
    ax.set_xlim(0, object_guess.shape[1])
    ax.set_ylim(object_guess.shape[0], 0)

    axes[0, 3].axis('off')

    # Row 2: Diffraction Comparison
    has_orig_diff = original_data_dict is not None and all(k in original_data_dict for k in ['diffraction', 'xcoords', 'ycoords'])
    if has_orig_diff:
        orig_points = np.stack([original_data_dict['xcoords'], original_data_dict['ycoords']], axis=1)
        tree = cKDTree(orig_points)
        sim_point = np.array([raw_data_instance.xcoords[0], raw_data_instance.ycoords[0]])
        
        zoom_factor = object_guess.shape[0] / original_data_dict['diffraction'].shape[1]
        sim_point_scaled = sim_point / zoom_factor if zoom_factor > 1 else sim_point
        dist, idx = tree.query(sim_point_scaled)
        
        ax = axes[1, 0]
        sim_diff = raw_data_instance.diff3d[0]
        im = ax.imshow(np.log1p(sim_diff), cmap='jet')
        ax.set_title("Simulated Diffraction [0]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 1]
        orig_diff = original_data_dict['diffraction'][idx]
        im = ax.imshow(np.log1p(orig_diff), cmap='jet')
        ax.set_title(f"Nearest Original Diffraction [{idx}]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 2]
        sim_diff_cropped = crop_center(sim_diff, orig_diff.shape[1], orig_diff.shape[0])
        diff_map = np.abs(np.log1p(sim_diff_cropped) - np.log1p(orig_diff))
        im = ax.imshow(diff_map, cmap='magma')
        ax.set_title(f"Difference (Dist: {dist:.1f} px)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[1, 3].axis('off')
    else:
        for i in range(4):
            ax = axes[1, i]
            if i < len(raw_data_instance.diff3d):
                im = ax.imshow(np.log1p(raw_data_instance.diff3d[i]), cmap='jet')
                ax.set_title(f"Simulated Diffraction {i}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')

    # Row 3: Sample Ground Truth Patches
    for i in range(4):
        ax = axes[2, i]
        if ground_truth_patches is not None and i < len(ground_truth_patches):
            im = ax.imshow(np.abs(ground_truth_patches[i]), cmap='gray')
            ax.set_title(f"Ground Truth Patch {i} (Amp)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(viz_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved visualization to {viz_path}")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the simulation script."""
    parser = argparse.ArgumentParser(
        description="Generate and save a simulated ptychography dataset."
    )
    parser.add_argument(
        "--input-file", type=str, required=True,
        help="Path to the input .npz file containing 'objectGuess' and 'probeGuess'."
    )
    parser.add_argument(
        "--output-file", type=str, required=True,
        help="Path to save the output simulated data as an .npz file."
    )
    parser.add_argument("--n-images", type=int, default=500, help="Number of scan positions to simulate.")
    parser.add_argument("--n-photons", type=float, default=1e9, help="Total photon count for normalization.")
    parser.add_argument("--gridsize", type=int, default=1, help="Grid size for simulation (usually 1 for PINN-style).")
    parser.add_argument("--buffer", type=float, default=None, help="Border size for random coordinates.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--visualize", action="store_true",
        help="If set, generate a PNG visualization of the simulation inputs and outputs."
    )
    return parser.parse_args()

def main():
    """Main function to handle command-line execution."""
    args = parse_arguments()
    
    # Load data once at the beginning
    object_guess, probe_guess, original_data_dict = load_data_for_sim(args.input_file, load_all=True)
    
    model_config = ModelConfig(
        N=probe_guess.shape[0],
        gridsize=args.gridsize
    )
    
    training_config = TrainingConfig(
        model=model_config,
        n_images=args.n_images,
        nphotons=args.n_photons,
        train_data_file=Path("dummy.npz"), 
        test_data_file=Path("dummy.npz")
    )
    
    try:
        simulate_and_save(
            config=training_config,
            input_file_path=args.input_file,
            output_file_path=args.output_file,
            original_data_for_vis=original_data_dict,
            buffer=args.buffer,
            seed=args.seed,
            visualize=args.visualize
        )
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        raise e
        sys.exit(1)

if __name__ == "__main__":
    main()
