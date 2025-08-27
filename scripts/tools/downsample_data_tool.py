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

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho.metadata import MetadataManager

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
        
    # Load with metadata support
    data_dict, metadata = MetadataManager.load_with_metadata(input_path)

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

    # --- 4. Process Ground Truth Patches (if present) ---
    if 'ground_truth_patches' in data_dict:
        original_patches = data_dict['ground_truth_patches']
        print(f"Processing 'ground_truth_patches' with shape {original_patches.shape}...")
        
        # HARDENED: Only accept complex arrays, fail loud if real-valued
        if not np.iscomplexobj(original_patches):
            raise ValueError(
                "FATAL ERROR: Ground truth patches are real-valued, but they must be complex. "
                "This indicates a bug in the simulation pipeline (likely in get_image_patches). "
                "Real-valued patches lose critical phase information and cannot be used for "
                "supervised training. Please regenerate the simulation data after fixing "
                "the root cause in ptycho/raw_data.py."
            )
        
        # Correct case: complex patches
        print("  Ground truth patches are complex-valued (correct).")
        if original_patches.ndim == 4 and original_patches.shape[-1] == 1:
            # Shape: (N, H, W, 1) - remove last dimension, bin, keep as 3D for compatibility
            binned_patches = np.array([
                bin_complex_array(patch[..., 0], bin_factor) for patch in original_patches
            ])
        else:
            # Shape: (N, H, W) - bin directly
            binned_patches = np.array([
                bin_complex_array(patch, bin_factor) for patch in original_patches
            ])
        
        # Rename to 'Y' for data loader compatibility
        data_dict['Y'] = binned_patches
        del data_dict['ground_truth_patches']
        print(f"  Binned and renamed to 'Y' with shape: {binned_patches.shape}")

    # --- 5. Save the new file with metadata ---
    print(f"Saving downsampled data to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Add transformation record if metadata exists
    if metadata:
        transform_params = {
            'crop_factor': crop_factor,
            'bin_factor': bin_factor
        }
        metadata = MetadataManager.add_transformation_record(
            metadata, 
            "downsample_data_tool.py", 
            "downsample", 
            transform_params
        )
    
    # Save with metadata
    MetadataManager.save_with_metadata(output_path, data_dict, metadata)
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
