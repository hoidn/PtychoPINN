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
from dataclasses import replace
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ptycho components
from ptycho.config.config import (
    DetectorSimulationConfig,
    ModelConfig,
    ProbeSimulationConfig,
    ScanSimulationConfig,
    SimulationConfig,
    SyntheticObjectConfig,
    TrainingConfig,
    load_simulation_config,
    simulation_config_to_dict,
    update_legacy_dict,
    validate_simulation_config,
)
from ptycho import params as p
from ptycho.config.legacy_state import scoped_legacy_params
from ptycho.image.cropping import center_crop_spatial
from ptycho.metadata import MetadataManager
from ptycho.simulation.probe_transform import (
    apply_probe_mask,
    apply_probe_transform_pipeline,
    apply_probe_transform_pipeline_with_metadata,
    normalize_probe_transform_pipeline,
    parse_probe_transform_pipeline,
)
from ptycho.simulation.identity import (
    build_simulation_probe_lineage,
    reject_mismatched_output_identity,
)
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

@scoped_legacy_params
def simulate_and_save(
    config: TrainingConfig,
    simulation: SimulationConfig,
    input_file_path: str | Path,
    output_file_path: str | Path,
    original_data_for_vis: Optional[Dict[str, Any]],
    visualize: bool = False,
) -> None:
    """
    Loads an object/probe, runs a ptychography simulation, saves the result,
    and optionally generates a visualization.
    """
    update_legacy_dict(p.cfg, simulation)
    update_legacy_dict(p.cfg, config)
    print("--- Configuration Updated for Simulation ---")
    p.print_params()
    print("------------------------------------------\n")
    
    object_guess, probe_guess, _ = load_data_for_sim(str(input_file_path), load_all=False)
    probe_guess, probe_lineage = prepare_probe_for_simulation_with_lineage(
        probe_guess, simulation
    )
    print(f"Loading object and probe from: {input_file_path}")
    print(f"  - Object shape: {object_guess.shape}")
    print(f"  - Probe shape: {probe_guess.shape}")

    buffer = simulation.scan.buffer
    if simulation.seed is not None:
        print(f"Setting random seed to: {simulation.seed}")
        np.random.seed(simulation.seed)

    reject_mismatched_output_identity(
        output_file_path,
        expected_simulation_digest=str(
            probe_lineage["simulation_config_sha256"]
        ),
        expected_recipe_digest=str(probe_lineage["dataset_recipe_sha256"]),
    )
    print(f"Simulating {config.n_images} diffraction patterns...")
    # The simulation bridge must be populated before importing legacy simulation.
    from ptycho.nongrid_simulation import generate_simulated_data

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
        seed=simulation.seed,
        simulation=simulation_config_to_dict(simulation),
        **probe_lineage,
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
    return center_crop_spatial(np.asarray(img), cropy, cropx)

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

def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the simulation script."""
    parser = argparse.ArgumentParser(
        description="Generate and save a simulated ptychography dataset."
    )
    parser.add_argument("--simulation-config", type=Path)
    parser.add_argument(
        "--input-file", type=str, required=True,
        help="Path to the input .npz file containing 'objectGuess' and 'probeGuess'."
    )
    parser.add_argument(
        "--output-file", type=str, required=True,
        help="Path to save the output simulated data as an .npz file."
    )
    parser.add_argument("--n-images", type=int, help="Number of scan positions to simulate.")
    parser.add_argument("--n-photons", type=float, help="Total photon count for normalization.")
    parser.add_argument("--gridsize", type=int, help="Grid size for simulation (usually 1 for PINN-style).")
    parser.add_argument("--buffer", type=int, help="Border size for random coordinates.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument(
        "--visualize", action="store_true",
        help="If set, generate a PNG visualization of the simulation inputs and outputs."
    )
    return parser.parse_args(argv)


def resolve_simulation_config(
    args: argparse.Namespace,
    *,
    object_shape: tuple[int, ...],
    probe_shape: tuple[int, ...],
) -> SimulationConfig:
    """Apply CLI > file > historical-no-file defaults deterministically."""
    if len(object_shape) < 2 or len(probe_shape) < 2:
        raise ValueError("object and probe inputs must have two spatial dimensions")
    object_size = (int(object_shape[0]), int(object_shape[1]))
    probe_size = (int(probe_shape[0]), int(probe_shape[1]))
    if probe_size[0] != probe_size[1]:
        raise ValueError(f"input probe must be square, got {probe_size}")

    if args.simulation_config is None:
        N = probe_size[0]
        simulation = SimulationConfig(
            N=N,
            probe=ProbeSimulationConfig(
                source="custom",
                source_path=Path(args.input_file),
                transform_pipeline=f"pad_preserve:{N}",
            ),
            object=SyntheticObjectConfig(
                kind="lines",
                image_size=object_size,
                diffractions_per_object=(
                    args.n_images if args.n_images is not None else 500
                ),
            ),
            scan=ScanSimulationConfig(
                kind="nongrid",
                grid_size=(
                    args.gridsize if args.gridsize is not None else 1,
                )
                * 2,
                buffer=(
                    args.buffer if args.buffer is not None else max(probe_size) // 2
                ),
            ),
            detector=DetectorSimulationConfig(
                photons_per_pattern=(
                    args.n_photons if args.n_photons is not None else 1e9
                )
            ),
            seed=args.seed if args.seed is not None else 42,
        )
    else:
        simulation = load_simulation_config(args.simulation_config)
        if simulation.object.image_size != object_size:
            raise ValueError(
                "simulation.object.image_size="
                f"{simulation.object.image_size} conflicts with input object shape {object_size}"
            )
        if args.n_images is not None:
            simulation = replace(
                simulation,
                object=replace(
                    simulation.object,
                    diffractions_per_object=args.n_images,
                ),
            )
        if args.n_photons is not None:
            simulation = replace(
                simulation,
                detector=replace(
                    simulation.detector,
                    photons_per_pattern=args.n_photons,
                ),
            )
        if args.gridsize is not None or args.buffer is not None:
            grid = args.gridsize if args.gridsize is not None else simulation.scan.grid_size[0]
            simulation = replace(
                simulation,
                scan=replace(
                    simulation.scan,
                    grid_size=(grid, grid),
                    buffer=(
                        args.buffer
                        if args.buffer is not None
                        else simulation.scan.buffer
                    ),
                ),
            )
        if args.seed is not None:
            simulation = replace(simulation, seed=args.seed)
    validate_simulation_config(simulation)
    if simulation.scan.kind != "nongrid":
        raise ValueError(
            "simulation.scan.kind must be 'nongrid' for simulate_and_save; "
            "use the grid-lines entrypoint for grid recipes"
        )
    if simulation.probe.source == "custom" and simulation.probe.source_path is None:
        raise ValueError(
            "simulation.probe.source_path is required for a file-backed custom probe"
        )
    return simulation


def prepare_probe_for_simulation(
    probe: np.ndarray,
    simulation: SimulationConfig,
) -> np.ndarray:
    """Apply the resolved generated-data probe recipe exactly once."""

    prepared, _ = prepare_probe_for_simulation_with_lineage(probe, simulation)
    return prepared


def prepare_probe_for_simulation_with_lineage(
    probe: np.ndarray,
    simulation: SimulationConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Prepare one probe and return its complete generated-data identity."""

    if simulation.probe.source == "custom":
        if simulation.probe.source_path is None:
            source = np.asarray(probe, dtype=np.complex64).squeeze()
        else:
            source_data, _ = MetadataManager.load_with_metadata(
                str(simulation.probe.source_path)
            )
            if "probeGuess" not in source_data:
                raise KeyError(
                    f"probeGuess missing from {simulation.probe.source_path}"
                )
            source = np.asarray(
                source_data["probeGuess"], dtype=np.complex64
            ).squeeze()
    else:
        from ptycho.probe import get_default_probe

        source = np.asarray(
            get_default_probe(N=simulation.N, fmt="np"),
            dtype=np.complex64,
        ).squeeze()
    normalized, steps = normalize_probe_transform_pipeline(
        target_N=simulation.N,
        probe_shape=source.shape,
        probe_scale_mode="pipeline",
        probe_smoothing_sigma=0.0,
        probe_transform_pipeline=simulation.probe.transform_pipeline,
    )
    transform_result = apply_probe_transform_pipeline_with_metadata(source, steps)
    prepared = apply_probe_mask(
        transform_result.probe, simulation.probe.mask_diameter
    )
    lineage = build_simulation_probe_lineage(
        simulation,
        raw_probe=source,
        normalized_pipeline=normalized,
        transformed_probe=prepared,
        transform_metadata=transform_result.metadata,
    )
    return prepared, lineage

def main():
    """Main function to handle command-line execution."""
    args = parse_arguments()
    
    # Load data once at the beginning
    object_guess, probe_guess, original_data_dict = load_data_for_sim(args.input_file, load_all=True)
    simulation = resolve_simulation_config(
        args,
        object_shape=object_guess.shape,
        probe_shape=probe_guess.shape,
    )
    
    model_config = ModelConfig(
        N=simulation.N,
        gridsize=simulation.scan.grid_size[0]
    )
    
    training_config = TrainingConfig(
        model=model_config,
        n_images=simulation.object.diffractions_per_object,
        nphotons=simulation.detector.photons_per_pattern,
        train_data_file=Path("dummy.npz"), 
        test_data_file=Path("dummy.npz")
    )
    
    try:
        simulate_and_save(
            config=training_config,
            simulation=simulation,
            input_file_path=args.input_file,
            output_file_path=args.output_file,
            original_data_for_vis=original_data_dict,
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
