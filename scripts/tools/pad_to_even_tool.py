#!/usr/bin/env python3
"""
Ptychography Data Padding Tool

This tool checks the dimensions of 'objectGuess' and 'probeGuess' in an NPZ file.
If any dimension is odd, it pads the array with a single row/column of zeros
to make it even. This is often required for FFT-based algorithms.
"""

import argparse
import os
import sys
import numpy as np

def pad_to_even(arr: np.ndarray) -> np.ndarray:
    """Pads a 2D array with zeros to make its dimensions even."""
    h, w = arr.shape
    
    # Determine how much padding is needed for height and width
    pad_h = 1 if h % 2 != 0 else 0
    pad_w = 1 if w % 2 != 0 else 0
    
    if pad_h == 0 and pad_w == 0:
        # No padding needed
        return arr
        
    print(f"Padding array from ({h}, {w}) to ({h + pad_h}, {w + pad_w})...")
    
    # Create padding specification: ((before_y, after_y), (before_x, after_x))
    # We add all padding to the 'after' side.
    pad_width = ((0, pad_h), (0, pad_w))
    
    return np.pad(arr, pad_width, mode='constant', constant_values=0)


def process_npz_for_even_dims(input_path: str, output_path: str):
    """Loads, pads, and saves a ptychography NPZ file."""
    print(f"--- Processing for even dimensions: {input_path} ---")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    with np.load(input_path) as data:
        data_dict = {key: data[key] for key in data.files}

    # Process 'objectGuess' and 'probeGuess'
    for key in ['objectGuess', 'probeGuess']:
        if key in data_dict:
            original_arr = data_dict[key]
            padded_arr = pad_to_even(original_arr)
            
            if padded_arr is not original_arr:
                data_dict[key] = padded_arr
            else:
                print(f"'{key}' already has even dimensions. No changes made.")

    # Save the new file
    print(f"Saving processed data to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, **data_dict)
    print("--- Padding Complete ---")


def main():
    parser = argparse.ArgumentParser(description="Pad arrays in an NPZ file to have even dimensions.")
    parser.add_argument("input_npz", help="Path to the source .npz file.")
    parser.add_argument("output_npz", help="Path to save the padded .npz file.")
    args = parser.parse_args()

    try:
        process_npz_for_even_dims(args.input_npz, args.output_npz)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
