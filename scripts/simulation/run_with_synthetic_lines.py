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
import os
import sys
import subprocess
import numpy as np
from pathlib import Path

# Add the project root to the Python path to allow imports from the `ptycho` library
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary components from the ptycho library
from ptycho import params as p
from ptycho.diffsim import sim_object_image
from ptycho.probe import get_default_probe

def generate_and_save_synthetic_input(
    output_dir: Path,
    probe_size: int,
) -> Path:
    """
    Generates a synthetic 'lines' object and a default probe, saving them to an NPZ file.

    Args:
        output_dir: The directory where the temporary input file will be saved.
        probe_size: The size for the probe (object will be ~3.5x larger).

    Returns:
        The path to the generated NPZ file.
    """
    print("--- Step 1: Generating Synthetic Input Data ---")

    print(f"Creating a default probe of size {probe_size}x{probe_size}...")
    p.set('N', probe_size)
    p.set('default_probe_scale', 0.7)
    default_probe = get_default_probe(N=probe_size, fmt='np').astype(np.complex64)
    
    object_scale_factor = 3.5
    full_object_size = int(probe_size * object_scale_factor)
    
    print(f"Creating a synthetic 'lines' object of size {full_object_size}x{full_object_size}...")
    p.set('data_source', 'lines')
    p.set('size', full_object_size)
    
    synthetic_object = sim_object_image(size=full_object_size)
    synthetic_object = synthetic_object.squeeze().astype(np.complex64)
    
    print(f"Generated probe: {default_probe.shape}, object: {synthetic_object.shape}")

    synthetic_input_path = output_dir / "synthetic_input.npz"

    print(f"Saving synthetic data to: {synthetic_input_path}")
    np.savez(
        synthetic_input_path,
        objectGuess=synthetic_object,
        probeGuess=default_probe
    )
    print("--- Synthetic Input Data Generation Complete ---\n")
    return synthetic_input_path


def run_simulation_workflow(
    synthetic_input_path: Path,
    output_dir: Path,
    extra_args: list
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
    # Use parse_known_args to separate this script's args from the rest
    parser = argparse.ArgumentParser(
        description="Runs the full ptychography simulation workflow using a synthetically generated 'lines' object.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the final report and intermediate synthetic data."
    )
    parser.add_argument(
        "--probe-size",
        type=int,
        default=64,
        help="The size (N) of the synthetic square probe to generate. The object will be scaled accordingly."
    )
    
    # This will parse the known arguments and leave the rest in `extra_args`
    args, extra_args = parser.parse_known_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Generate and save the synthetic object and probe
        synthetic_input_path = generate_and_save_synthetic_input(
            output_dir=output_dir,
            probe_size=args.probe_size
        )

        # Step 2: Run the main simulation workflow with the generated file
        run_simulation_workflow(
            synthetic_input_path=synthetic_input_path,
            output_dir=output_dir,
            extra_args=extra_args # Pass the collected extra arguments
        )
        
        print(f"\nSuccess! The final simulated data is available in: {output_dir}/simulated_data.npz")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
