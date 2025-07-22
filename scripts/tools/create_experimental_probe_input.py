#!/usr/bin/env python3
"""
Experimental Probe Input Generator

This tool creates simulation input files by combining experimental probes from the fly64 dataset
with synthetic 'lines' objects. The output is compatible with the simulate_and_save.py pipeline.

Key Features:
- Extracts experimental probe from fly64 dataset
- Generates synthetic 'lines' object using the same process as run_with_synthetic_lines.py
- Creates NPZ file compatible with simulate_and_save.py data contracts
- Provides command-line interface for easy integration

Usage:
    python scripts/tools/create_experimental_probe_input.py \
        --fly64-file datasets/fly64/fly001_64_train_converted.npz \
        --output-file simulation_input_experimental_probe.npz

Example Integration:
    # 1. Create input file with experimental probe
    python scripts/tools/create_experimental_probe_input.py \
        --fly64-file datasets/fly64/fly001_64_train_converted.npz \
        --output-file experimental_input.npz
    
    # 2. Run simulation with experimental probe
    python scripts/simulation/simulate_and_save.py \
        --input-file experimental_input.npz \
        --output-file experimental_sim_data.npz \
        --n-images 2000

Created for: Probe Generalization Study (Phase 2)
Author: Claude Code Assistant
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Import necessary components from the ptycho library
from ptycho import params as p
from ptycho.diffsim import sim_object_image


def load_experimental_probe(fly64_path: str) -> np.ndarray:
    """
    Load the experimental probe from the fly64 dataset.
    
    Args:
        fly64_path: Path to the fly64 NPZ file
        
    Returns:
        The experimental probe as a complex64 array
        
    Raises:
        FileNotFoundError: If the fly64 file doesn't exist
        KeyError: If the probeGuess key is missing
        ValueError: If the probe data is malformed
    """
    try:
        print(f"Loading experimental probe from: {fly64_path}")
        data = np.load(fly64_path)
        
        if 'probeGuess' not in data:
            raise KeyError("probeGuess key not found in fly64 dataset")
        
        probe = data['probeGuess']
        
        # Validate probe structure
        if probe.dtype != np.complex64:
            print(f"Warning: Converting probe dtype from {probe.dtype} to complex64")
            probe = probe.astype(np.complex64)
        
        if len(probe.shape) != 2:
            raise ValueError(f"Expected 2D probe, got shape {probe.shape}")
        
        if probe.shape[0] != probe.shape[1]:
            raise ValueError(f"Expected square probe, got shape {probe.shape}")
        
        # Check for NaN/inf values
        if not np.isfinite(probe).all():
            raise ValueError("Probe contains NaN or infinite values")
        
        print(f"Successfully loaded probe: shape={probe.shape}, dtype={probe.dtype}")
        return probe
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Fly64 dataset not found: {fly64_path}")
    except Exception as e:
        raise ValueError(f"Error loading experimental probe: {e}")


def generate_synthetic_lines_object(probe_size: int) -> np.ndarray:
    """
    Generate a synthetic 'lines' object compatible with the experimental probe.
    
    Uses the same logic as run_with_synthetic_lines.py to ensure consistency.
    
    Args:
        probe_size: Size of the probe (object will be ~3.5x larger)
        
    Returns:
        The synthetic object as a complex64 array
    """
    print(f"Generating synthetic 'lines' object for probe size {probe_size}...")
    
    # Calculate object size using same scale factor as run_with_synthetic_lines.py
    object_scale_factor = 3.5
    full_object_size = int(probe_size * object_scale_factor)
    
    # Set parameters for synthetic object generation
    p.set('data_source', 'lines')
    p.set('size', full_object_size)
    
    # Generate the synthetic object
    synthetic_object = sim_object_image(size=full_object_size)
    synthetic_object = synthetic_object.squeeze().astype(np.complex64)
    
    print(f"Generated synthetic object: shape={synthetic_object.shape}, dtype={synthetic_object.dtype}")
    return synthetic_object


def save_input_file(object_guess: np.ndarray, probe_guess: np.ndarray, output_path: str) -> None:
    """
    Save the object and probe to an NPZ file compatible with simulate_and_save.py.
    
    Args:
        object_guess: The synthetic object array
        probe_guess: The experimental probe array
        output_path: Path where the NPZ file will be saved
    """
    print(f"Saving input file to: {output_path}")
    
    # Ensure arrays are complex64 for compatibility
    object_guess = object_guess.astype(np.complex64)
    probe_guess = probe_guess.astype(np.complex64)
    
    # Save using the expected key names for simulate_and_save.py
    np.savez(
        output_path,
        objectGuess=object_guess,
        probeGuess=probe_guess
    )
    
    print(f"Successfully saved input file with:")
    print(f"  objectGuess: {object_guess.shape} {object_guess.dtype}")
    print(f"  probeGuess: {probe_guess.shape} {probe_guess.dtype}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Create simulation input file with experimental probe and synthetic object",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with fly64 dataset
  %(prog)s --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file experimental_input.npz
  
  # Custom probe size
  %(prog)s --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file custom_input.npz --probe-size 128

Integration with simulation:
  # After creating input file, use with simulate_and_save.py
  python scripts/simulation/simulate_and_save.py --input-file experimental_input.npz --output-file sim_data.npz --n-images 2000
        """)
    
    parser.add_argument(
        '--fly64-file',
        required=True,
        help='Path to fly64 NPZ dataset containing experimental probe'
    )
    
    parser.add_argument(
        '--output-file',
        required=True,
        help='Path for output NPZ file (will be created/overwritten)'
    )
    
    parser.add_argument(
        '--probe-size',
        type=int,
        default=64,
        help='Expected probe size in pixels (default: 64)'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input file exists
        if not Path(args.fly64_file).exists():
            print(f"Error: Input file does not exist: {args.fly64_file}")
            return 1
        
        # Validate output directory is writable
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load experimental probe
        experimental_probe = load_experimental_probe(args.fly64_file)
        
        # Validate probe size matches expectation
        if experimental_probe.shape[0] != args.probe_size:
            print(f"Warning: Probe size {experimental_probe.shape[0]} doesn't match expected {args.probe_size}")
            probe_size = experimental_probe.shape[0]
        else:
            probe_size = args.probe_size
        
        # Generate synthetic object
        synthetic_object = generate_synthetic_lines_object(probe_size)
        
        # Save combined input file
        save_input_file(synthetic_object, experimental_probe, args.output_file)
        
        print(f"\n✅ Success! Created experimental probe input file: {args.output_file}")
        print(f"Ready for simulation with: python scripts/simulation/simulate_and_save.py --input-file {args.output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())