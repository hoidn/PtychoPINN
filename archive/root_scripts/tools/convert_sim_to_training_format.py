#!/usr/bin/env python3
"""
Convert simulated data format to training format expected by comparison script.
The simulation outputs use 'diff3d' but training expects 'diffraction'.
"""
import numpy as np
import argparse

def convert_format(input_file, output_file):
    """Convert simulation format to training format."""
    print(f"Converting {input_file} to {output_file}...")
    
    data = np.load(input_file)
    
    # Convert keys from simulation format to training format
    converted_data = {}
    
    # Map keys
    key_mapping = {
        'diff3d': 'diffraction',
        'xcoords': 'xcoords', 
        'ycoords': 'ycoords',
        'probeGuess': 'probeGuess',
        'objectGuess': 'objectGuess'
    }
    
    for sim_key, train_key in key_mapping.items():
        if sim_key in data:
            converted_data[train_key] = data[sim_key]
            print(f"  Mapped {sim_key} -> {train_key}: {data[sim_key].shape}")
    
    # Copy any additional keys that might be needed
    for key in data.keys():
        if key not in key_mapping and key not in converted_data:
            converted_data[key] = data[key]
            print(f"  Copied {key}: {data[key].shape}")
    
    print(f"Saving converted data to {output_file}")
    np.savez_compressed(output_file, **converted_data)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert simulation format to training format")
    parser.add_argument("input_file", help="Input NPZ file (simulation format)")
    parser.add_argument("output_file", help="Output NPZ file (training format)")
    
    args = parser.parse_args()
    convert_format(args.input_file, args.output_file)