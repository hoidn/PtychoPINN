#!/usr/bin/env python3
"""
Ptychography Data Downsampling Tool

This tool downsamples a ptychography dataset by:
1. Cropping the center of each diffraction pattern.
2. Binning (averaging) the real-space object and probe arrays.
3. Scaling the coordinate arrays to match the new pixel size.
"""

import argparse
import os
import sys
import numpy as np
from skimage.measure import block_reduce

def crop_center(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Crops the center of a 2D array."""
    h, w = img.shape
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    return img[start_y : start_y + new_h, start_x : start_x + new_w]

def bin_complex_array(arr: np.ndarray, bin_factor: int) -> np.ndarray:
    """Bins a complex array by averaging 2x2 blocks on its real and imaginary parts."""
    if not np.iscomplexobj(arr):
        raise ValueError("Input array must be complex.")
    
    block_size = (bin_factor, bin_factor)
    
    # Bin real and imaginary parts separately
    binned_real = block_reduce(arr.real, block_size=block_size, func=np.mean)
    binned_imag = block_reduce(arr.imag, block_size=block_size, func=np.mean)
    
    return (binned_real + 1j * binned_imag).astype(arr.dtype)

def downsample_npz(
    input_path: str,
    output_path: str,
    crop_factor: int,
    bin_factor: int,
):
    """Loads, downsamples, and saves a ptychography NPZ file."""
    print(f"--- Downsampling Data from: {input_path} ---")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    with np.load(input_path) as data:
        # Create a mutable dictionary of all arrays
        data_dict = {key: data[key] for key in data.files}

    # --- 1. Crop Diffraction Data ---
    if 'diff3d' in data_dict:
        original_patterns = data_dict['diff3d']
        h, w = original_patterns.shape[1:]
        new_h, new_w = h // crop_factor, w // crop_factor
        
        print(f"Cropping 'diff3d' from ({h}, {w}) to ({new_h}, {new_w})...")
        cropped_patterns = np.array([
            crop_center(p, new_h, new_w) for p in original_patterns
        ])
        data_dict['diff3d'] = cropped_patterns

    # --- 2. Bin Real-Space Arrays ---
    for key in ['objectGuess', 'probeGuess']:
        if key in data_dict:
            original_arr = data_dict[key]
            print(f"Binning '{key}' from {original_arr.shape} by a factor of {bin_factor}...")
            binned_arr = bin_complex_array(original_arr, bin_factor)
            data_dict[key] = binned_arr
            print(f"  New shape: {binned_arr.shape}")

    # --- 3. Scale Coordinate Arrays ---
    # This is critical for physical consistency. If pixel size doubles, coordinate values must halve.
    coord_scale_factor = 1.0 / bin_factor
    print(f"Scaling coordinate arrays by a factor of {coord_scale_factor}...")
    for key in ['xcoords', 'ycoords', 'xcoords_start', 'ycoords_start']:
        if key in data_dict:
            data_dict[key] *= coord_scale_factor

    # --- 4. Save the new file ---
    print(f"Saving downsampled data to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, **data_dict)
    print("--- Downsampling Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Downsample a ptychography NPZ dataset.")
    parser.add_argument("input_npz", help="Path to the source .npz file.")
    parser.add_argument("output_npz", help="Path to save the downsampled .npz file.")
    parser.add_argument("--crop-factor", type=int, required=True, help="Factor to crop diffraction patterns (e.g., 2 for 1/4 area).")
    parser.add_argument("--bin-factor", type=int, required=True, help="Factor to bin real-space arrays (e.g., 2 for 2x2 blocks).")
    args = parser.parse_args()

    try:
        downsample_npz(args.input_npz, args.output_npz, args.crop_factor, args.bin_factor)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
