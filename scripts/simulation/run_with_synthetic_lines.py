#!/usr/bin/env python
# scripts/simulation/run_with_synthetic_lines.py

"""
A runner script that first generates a synthetic object and probe,
then executes the main simulation workflow (`simulate_and_save.py`) using
the generated data as input.

This script automates the following steps:
1. Creates a synthetic object of type 'lines'.
2. Creates a default probe.
3. Saves the object and probe to a temporary NPZ file.
4. Calls `scripts/simulation/simulate_and_save.py` with the path to this
   NPZ file, forwarding all other specified arguments.

Example Usage:
    # Basic run, saving the report to 'lines_report/'
    python scripts/simulation/run_with_synthetic_lines.py \\
        --output-dir lines_report

    # Run with more images and a different probe size
    python scripts/simulation/run_with_synthetic_lines.py \\
        --output-dir lines_report_large \\
        --probe-size 128 \\
        --n-images 4000
"""

import argparse
from dataclasses import replace
import json
import os
import sys
import subprocess
import numpy as np
from pathlib import Path

# Add the project root to the Python path to allow imports from the `ptycho` library
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho import params as p
from ptycho.config import (
    ProbeSimulationConfig,
    ScanSimulationConfig,
    SimulationConfig,
    SyntheticObjectConfig,
    load_simulation_config,
    simulation_config_to_dict,
    validate_simulation_config,
)
from ptycho.config.config import update_legacy_dict
from ptycho.simulation.probe_transform import (
    parse_probe_transform_pipeline,
    serialize_probe_transform_pipeline,
)


def parse_arguments(argv=None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Runs the simulation workflow with a synthetic lines object.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--simulation-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-size", type=int)
    return parser.parse_known_args(argv)


def resolve_synthetic_simulation(args: argparse.Namespace) -> SimulationConfig:
    if args.simulation_config is None:
        N = args.probe_size if args.probe_size is not None else 64
        simulation = SimulationConfig(
            N=N,
            probe=ProbeSimulationConfig(
                source="ideal",
                transform_pipeline=f"pad_preserve:{N}",
            ),
            object=SyntheticObjectConfig(
                kind="lines",
                image_size=(int(N * 3.5),) * 2,
            ),
            scan=ScanSimulationConfig(
                kind="nongrid",
                buffer=N // 2,
            ),
        )
    else:
        simulation = load_simulation_config(args.simulation_config)
        if args.probe_size is not None and args.probe_size != simulation.N:
            steps = [
                ({**step, "target_N": args.probe_size} if "target_N" in step else step)
                for step in parse_probe_transform_pipeline(
                    simulation.probe.transform_pipeline
                )
            ]
            simulation = replace(
                simulation,
                N=args.probe_size,
                probe=replace(
                    simulation.probe,
                    transform_pipeline=serialize_probe_transform_pipeline(steps),
                ),
            )
    if simulation.object.kind != "lines":
        raise ValueError(
            "run_with_synthetic_lines requires simulation.object.kind='lines'"
        )
    if simulation.scan.kind != "nongrid":
        raise ValueError(
            "simulation.scan.kind must be 'nongrid' for run_with_synthetic_lines"
        )
    if simulation.probe.source == "custom" and simulation.probe.source_path is None:
        raise ValueError(
            "simulation.probe.source_path is required for a file-backed custom probe"
        )
    validate_simulation_config(simulation)
    return simulation

def generate_and_save_synthetic_input(
    output_dir: Path,
    simulation: SimulationConfig,
) -> Path:
    """
    Generates a synthetic 'lines' object and a default probe, saving them to an NPZ file.

    Args:
        output_dir: The directory where the temporary input file will be saved.
        simulation: The resolved lines-object and probe construction recipe.

    Returns:
        The path to the generated NPZ file.
    """
    print("--- Step 1: Generating Synthetic Input Data ---")

    update_legacy_dict(p.cfg, simulation)
    from ptycho.diffsim import sim_object_image
    from ptycho.probe import get_default_probe

    if simulation.seed is not None:
        np.random.seed(simulation.seed)
    if simulation.probe.source == "ideal":
        print(f"Creating a default probe for target size {simulation.N}...")
        source_probe = get_default_probe(N=simulation.N, fmt='np').astype(np.complex64)
    else:
        if simulation.probe.source_path is None:
            raise ValueError("custom probe construction requires simulation.probe.source_path")
        with np.load(simulation.probe.source_path, allow_pickle=False) as archive:
            source_probe = np.asarray(archive["probeGuess"], dtype=np.complex64)
    full_object_size = simulation.object.image_size[0]
    
    print(f"Creating a synthetic 'lines' object of size {full_object_size}x{full_object_size}...")
    p.set('data_source', 'lines')
    p.set('size', full_object_size)
    
    synthetic_object = sim_object_image(size=full_object_size)
    synthetic_object = synthetic_object.squeeze().astype(np.complex64)
    
    print(f"Generated source probe: {source_probe.shape}, object: {synthetic_object.shape}")

    synthetic_input_path = output_dir / "synthetic_input.npz"

    print(f"Saving synthetic data to: {synthetic_input_path}")
    np.savez(
        synthetic_input_path,
        objectGuess=synthetic_object,
        probeGuess=source_probe
    )
    print("--- Synthetic Input Data Generation Complete ---\n")
    return synthetic_input_path


def run_simulation_workflow(
    synthetic_input_path: Path,
    output_dir: Path,
    extra_args: list,
    simulation_config_path: Path,
) -> None:
    """
    Executes the simulate_and_save.py script as a subprocess.

    Args:
        synthetic_input_path: Path to the generated NPZ input file.
        output_dir: The final output directory for the simulation.
        extra_args: A list of additional arguments to forward to simulate_and_save.py.
    """
    print("--- Step 2: Running Simulation ---")
    
    # Path to the target script is now the corrected one
    simulate_script_path = Path(__file__).parent / "simulate_and_save.py"
    
    # Output path for the final simulated data
    output_file_path = output_dir / "simulated_data.npz"

    # Construct the command
    command = [
        sys.executable,
        str(simulate_script_path),
        "--input-file", str(synthetic_input_path),
        "--output-file", str(output_file_path),
        "--simulation-config", str(simulation_config_path),
    ]
    
    # Add any extra arguments forwarded from the command line.
    command.extend(extra_args)

    print(f"Executing command:\n{' '.join(command)}\n")

    try:
        subprocess.run(command, check=True)
        print("\n--- Simulation Complete ---")
        print(f"Simulated data saved to: {output_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError: The simulation script failed with exit code {e.returncode}.", file=sys.stderr)
        sys.exit(e.returncode)


def main():
    """Main function to orchestrate the entire process."""
    args, extra_args = parse_arguments()
    simulation = resolve_synthetic_simulation(args)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Generate and save the synthetic object and probe
        synthetic_input_path = generate_and_save_synthetic_input(
            output_dir=output_dir,
            simulation=simulation,
        )
        resolved_config_path = output_dir / "resolved_simulation_config.json"
        resolved_config_path.write_text(
            json.dumps(
                {"simulation": simulation_config_to_dict(simulation)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        # Step 2: Run the main simulation workflow with the generated file
        run_simulation_workflow(
            synthetic_input_path=synthetic_input_path,
            output_dir=output_dir,
            extra_args=extra_args,
            simulation_config_path=resolved_config_path,
        )
        
        print(f"\nSuccess! The final simulated data is available in: {output_dir}/simulated_data.npz")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
