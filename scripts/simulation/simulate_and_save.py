#!/usr/bin/env python
# scripts/simulation/simulate_and_save.py

"""
Generates a simulated ptychography dataset and saves it to an NPZ file.

This script uses explicit orchestration of modular functions instead of the
monolithic RawData.from_simulation method, fixing gridsize > 1 crashes and
improving maintainability.

Refactored: 2025-08-02 - Replaced monolithic from_simulation with modular workflow
to fix gridsize > 1 ValueError and improve architectural consistency.

Example:
    # Run simulation and also create a summary plot with comparisons
    python scripts/simulation/simulate_and_save.py \\
        --input-file /path/to/prepared_data.npz \\
        --output-file /path/to/simulation_output.npz \\
        --visualize
        
    # Run with gridsize > 1
    python scripts/simulation/simulate_and_save.py \\
        --input-file /path/to/prepared_data.npz \\
        --output-file /path/to/simulation_output.npz \\
        --gridsize 2
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
# Note: Delaying some imports until after configuration is set up
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p
from ptycho.workflows.simulation_utils import load_probe_from_source, validate_probe_object_compatibility
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import logging
import tensorflow as tf

# Set up logger
logger = logging.getLogger(__name__)

def load_data_for_sim(file_path: str, load_all: bool = False) -> tuple:
    """Loads object and probe, and optionally all other data from an NPZ file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with np.load(file_path) as data:
        if 'objectGuess' not in data or 'probeGuess' not in data:
            raise ValueError("The .npz file must contain 'objectGuess' and 'probeGuess'")
        
        objectGuess = data['objectGuess']
        probeGuess = data['probeGuess']
        
        if load_all:
            all_data = {key: data[key] for key in data.files}
            return objectGuess, probeGuess, all_data
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
    probe_file: Optional[str] = None,
    debug: bool = False,
) -> None:
    """
    Loads an object/probe, runs a ptychography simulation, saves the result,
    and optionally generates a visualization.
    
    This refactored version uses explicit orchestration of modular functions
    instead of the monolithic RawData.from_simulation method.
    """
    # Set up debug logging if requested
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(lineno)d - %(levelname)s - %(message)s')
        logger.setLevel(logging.DEBUG)
    
    # Section 1: Input Loading & Validation
    update_legacy_dict(p.cfg, config)
    logger.debug("--- Configuration Updated for Simulation ---")
    if debug:
        p.print_params()
    
    # 1.A: Load NPZ input
    object_guess, probe_guess, _ = load_data_for_sim(str(input_file_path), load_all=False)
    print(f"Loading object and probe from: {input_file_path}")
    print(f"  - Object shape: {object_guess.shape}, dtype: {object_guess.dtype}")
    print(f"  - Probe shape: {probe_guess.shape}, dtype: {probe_guess.dtype}")
    
    # Validate complex dtype
    if not np.iscomplexobj(object_guess):
        raise ValueError(f"objectGuess must be complex, got {object_guess.dtype}")
    if not np.iscomplexobj(probe_guess):
        raise ValueError(f"probeGuess must be complex, got {probe_guess.dtype}")
    
    # 1.B: Probe override logic
    if probe_file is not None:
        try:
            print(f"\nOverriding probe with external file: {probe_file}")
            external_probe = load_probe_from_source(probe_file)
            validate_probe_object_compatibility(external_probe, object_guess)
            probe_guess = external_probe
            print(f"  - External probe shape: {probe_guess.shape}")
            logger.info(f"Successfully loaded external probe from {probe_file}")
        except (ValueError, FileNotFoundError, KeyError) as e:
            raise ValueError(f"Failed to load probe from {probe_file}: {str(e)}")

    if buffer is None:
        buffer = max(probe_guess.shape) // 2

    if seed is not None:
        print(f"Setting random seed to: {seed}")
        np.random.seed(seed)

    # Section 2: Coordinate Generation & Grouping
    # 2.A: Import and configure parameters
    p.set('N', probe_guess.shape[0])
    p.set('gridsize', config.model.gridsize)
    logger.debug(f"Set N={probe_guess.shape[0]}, gridsize={config.model.gridsize}")
    
    # Now safe to import modules that depend on params
    from ptycho import raw_data
    from ptycho import tf_helper as hh
    from ptycho.diffsim import illuminate_and_diffract
    
    # Generate scan coordinates
    height, width = object_guess.shape
    buffer = min(buffer, min(height, width) / 2 - 1)
    xcoords = np.random.uniform(buffer, width - buffer, config.n_images)
    ycoords = np.random.uniform(buffer, height - buffer, config.n_images)
    scan_index = np.zeros(config.n_images, dtype=int)
    
    logger.debug(f"Generated {config.n_images} scan positions within bounds")
    logger.debug(f"X range: [{xcoords.min():.2f}, {xcoords.max():.2f}]")
    logger.debug(f"Y range: [{ycoords.min():.2f}, {ycoords.max():.2f}]")
    
    # 2.B: Generate grouped coordinates
    print(f"Simulating {config.n_images} diffraction patterns with gridsize={config.model.gridsize}...")
    
    # For gridsize=1, we don't need grouping
    if config.model.gridsize == 1:
        # Simple case: each coordinate is its own group
        scan_offsets = np.stack([ycoords, xcoords], axis=1)  # Shape: (n_images, 2)
        group_neighbors = np.arange(config.n_images).reshape(-1, 1)  # Shape: (n_images, 1)
        n_groups = config.n_images
        logger.debug(f"GridSize=1: {n_groups} groups, each with 1 pattern")
    else:
        # Use group_coords for gridsize > 1
        # First calculate relative coordinates
        global_offsets, local_offsets, nn_indices = raw_data.calculate_relative_coords(xcoords, ycoords)
        # Check if these are already numpy arrays or tensors
        scan_offsets = global_offsets if isinstance(global_offsets, np.ndarray) else global_offsets.numpy()
        group_neighbors = nn_indices if isinstance(nn_indices, np.ndarray) else nn_indices.numpy()
        n_groups = scan_offsets.shape[0]
        logger.debug(f"GridSize={config.model.gridsize}: {n_groups} groups, each with {config.model.gridsize**2} patterns")
        logger.debug(f"scan_offsets shape: {scan_offsets.shape}, group_neighbors shape: {group_neighbors.shape}")
    
    # Section 3: Patch Extraction
    # 3.A: Extract object patches (Y)
    if config.model.gridsize == 1:
        # For gridsize=1, we can directly extract patches without the complex grouping
        N = config.model.N
        # Pad the object once
        gt_padded = hh.pad(object_guess[None, ..., None], N // 2)
        
        # Create array to hold patches
        Y_patches_list = []
        
        # Extract patches one by one
        for i in range(n_groups):
            offset = tf.constant([[scan_offsets[i, 1], scan_offsets[i, 0]]], dtype=tf.float32)  # Note: x,y order for translate
            translated = hh.translate(gt_padded, -offset)
            patch = translated[0, :N, :N, 0]  # Extract center patch
            Y_patches_list.append(patch)
        
        # Stack into tensor with shape (B, N, N, 1) for gridsize=1
        Y_patches = tf.stack(Y_patches_list, axis=0)
        Y_patches = tf.expand_dims(Y_patches, axis=-1)  # Add channel dimension
        logger.debug(f"Extracted {len(Y_patches_list)} patches for gridsize=1")
    else:
        # For gridsize>1, use the already calculated offsets
        Y_patches = raw_data.get_image_patches(
            object_guess,
            global_offsets,
            local_offsets,
            N=config.model.N,
            gridsize=config.model.gridsize
        )
    
    Y_patches_np = Y_patches.numpy()
    logger.debug(f"Extracted patches shape: {Y_patches_np.shape}, dtype: {Y_patches_np.dtype}")
    
    # 3.B: Validate patch content
    assert np.any(Y_patches_np != 0), "All patches are zero!"
    assert np.any(np.imag(Y_patches_np) != 0), "Patches have no imaginary component!"
    logger.debug(f"Patches valid: min abs={np.abs(Y_patches_np).min():.3f}, max abs={np.abs(Y_patches_np).max():.3f}")
    
    # Section 4: Format Conversion & Physics Simulation
    # 4.A: Convert Channel to Flat Format
    Y_flat = hh._channel_to_flat(Y_patches)
    logger.debug(f"Converted to flat format: {Y_patches.shape} -> {Y_flat.shape}")
    
    # Split into amplitude and phase for illuminate_and_diffract
    Y_I_flat = tf.math.abs(Y_flat)
    Y_phi_flat = tf.math.angle(Y_flat)
    
    # 4.B: Prepare probe for simulation
    # Expand probe dimensions to match expected format
    probe_tensor = tf.constant(probe_guess[:, :, np.newaxis], dtype=tf.complex64)
    logger.debug(f"Probe tensor shape: {probe_tensor.shape}")
    
    # 4.C: Run physics simulation
    X_flat, _, _, _ = illuminate_and_diffract(Y_I_flat, Y_phi_flat, probe_tensor)
    logger.debug(f"Diffraction simulation complete: output shape {X_flat.shape}")
    
    # Verify output is real amplitude
    assert tf.reduce_all(tf.math.imag(X_flat) == 0), "Diffraction should be real amplitude"
    
    # 4.D: Convert Flat to Channel Format
    X_channel = hh._flat_to_channel(X_flat, N=config.model.N, gridsize=config.model.gridsize)
    logger.debug(f"Converted back to channel format: {X_flat.shape} -> {X_channel.shape}")
    
    # Section 5: Output Assembly & Saving
    # 5.A: Reshape arrays for NPZ format
    N = config.model.N
    if config.model.gridsize == 1:
        # For gridsize=1, squeeze the channel dimension
        diffraction = np.squeeze(X_channel.numpy(), axis=-1)  # Shape: (n_images, N, N)
        Y_final = np.squeeze(Y_patches_np, axis=-1)  # Shape: (n_images, N, N)
    else:
        # For gridsize>1, reshape to 3D by flattening groups
        diffraction = X_channel.numpy().reshape(-1, N, N)  # Shape: (n_groups * gridsize², N, N)
        Y_final = Y_patches_np.reshape(-1, N, N)  # Shape: (n_groups * gridsize², N, N)
    
    logger.debug(f"Final diffraction shape: {diffraction.shape}, dtype: {diffraction.dtype}")
    logger.debug(f"Final Y shape: {Y_final.shape}, dtype: {Y_final.dtype}")
    
    # 5.B: Prepare coordinate arrays
    if config.model.gridsize == 1:
        # Simple case: use original coordinates
        xcoords_final = xcoords
        ycoords_final = ycoords
    else:
        # For gridsize>1, need to expand coordinates for each neighbor
        xcoords_final = []
        ycoords_final = []
        for group_idx in range(n_groups):
            for neighbor_idx in group_neighbors[group_idx]:
                xcoords_final.append(xcoords[neighbor_idx])
                ycoords_final.append(ycoords[neighbor_idx])
        xcoords_final = np.array(xcoords_final)
        ycoords_final = np.array(ycoords_final)
    
    logger.debug(f"Final coordinates length: {len(xcoords_final)}")
    assert len(xcoords_final) == diffraction.shape[0], f"Coordinate mismatch: {len(xcoords_final)} != {diffraction.shape[0]}"
    
    # 5.C: Assemble output dictionary
    output_dict = {
        'diffraction': diffraction.astype(np.float32),  # Amplitude as per data contract
        'Y': Y_final,  # Ground truth patches
        'objectGuess': object_guess,
        'probeGuess': probe_guess,
        'xcoords': xcoords_final.astype(np.float64),
        'ycoords': ycoords_final.astype(np.float64),
        'scan_index': np.repeat(scan_index[:n_groups], config.model.gridsize**2) if config.model.gridsize > 1 else scan_index
    }
    
    # Add legacy keys for backward compatibility
    output_dict['diff3d'] = diffraction.astype(np.float32)
    output_dict['xcoords_start'] = xcoords_final.astype(np.float64)
    output_dict['ycoords_start'] = ycoords_final.astype(np.float64)
    
    print(f"Output summary:")
    for key, val in output_dict.items():
        if isinstance(val, np.ndarray):
            print(f"  - {key}: shape {val.shape}, dtype {val.dtype}")
    
    # 5.D: Save NPZ file
    output_dir = Path(output_file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_file_path, **output_dict)
    print(f"✓ Saved simulated data to: {output_file_path}")

    if visualize:
        print("Generating visualization plot...")
        # Create a minimal RawData-like object for visualization compatibility
        class VisualizationData:
            def __init__(self, data_dict):
                self.xcoords = data_dict['xcoords']
                self.ycoords = data_dict['ycoords']
                self.diff3d = data_dict['diffraction']
        
        vis_data = VisualizationData(output_dict)
        visualize_simulation_results(
            object_guess=object_guess,
            probe_guess=probe_guess,
            raw_data_instance=vis_data,
            ground_truth_patches=Y_final,
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
    parser.add_argument(
        "--probe-file", type=str, default=None,
        help="Path to external probe file (.npy or .npz) to override the probe from input file"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging to trace tensor shapes and data flow"
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
            visualize=args.visualize,
            probe_file=args.probe_file,
            debug=args.debug
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
