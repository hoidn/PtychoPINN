#!/usr/bin/env python3
"""
NPZ Update Tool

This tool updates NPZ files with new object reconstructions and performs data conversions.
Can be used both as a standalone script or imported as a module.

Usage:
    As script: python update_tool.py original.npz reconstruction.npy output.npz
    As module: from update_tool import update_object_guess
"""

import numpy as np
import argparse
from typing import Union
import os


def update_object_guess(
    original_npz_path: str,
    new_object_guess: Union[np.ndarray, str],
    output_npz_path: str
) -> None:
    """
    Update the objectGuess in an NPZ file with a new reconstruction.
    
    This function:
    1. Loads all data from the original NPZ file
    2. Updates the 'objectGuess' field with the new reconstruction
    3. Converts 'diff3d' to float32 for memory efficiency
    4. Saves to a new NPZ file (does not modify the original)
    
    Args:
        original_npz_path: Path to the original NPZ file
        new_object_guess: Either a numpy array or path to .npy file containing
                         the new object reconstruction
        output_npz_path: Path where the updated NPZ will be saved
    
    Raises:
        FileNotFoundError: If the original NPZ file doesn't exist
        KeyError: If required keys are missing from the NPZ file
        ValueError: If array shapes are incompatible
    """
    # Step 1: Load the original data
    if not os.path.exists(original_npz_path):
        raise FileNotFoundError(f"Original NPZ file not found: {original_npz_path}")
    
    print(f"Loading original data from {original_npz_path}...")
    with np.load(original_npz_path) as original_data:
        # Copy all arrays to a standard dictionary (NpzFile is read-only)
        data_dict = {key: original_data[key].copy() for key in original_data.files}
        print(f"  Found keys: {list(data_dict.keys())}")
    
    # Step 2: Load new object guess if it's a file path
    if isinstance(new_object_guess, str):
        print(f"Loading reconstruction from {new_object_guess}...")
        new_object_guess = np.load(new_object_guess)
    
    # Step 3: Handle potential shape differences
    # Tike often outputs 3D arrays (1, H, W) but NPZ expects 2D (H, W)
    if new_object_guess.ndim == 3 and new_object_guess.shape[0] == 1:
        print(f"  Squeezing 3D array {new_object_guess.shape} to 2D...")
        new_object_guess = new_object_guess.squeeze(0)
    
    print(f"  New object shape: {new_object_guess.shape}")
    if 'objectGuess' in data_dict:
        print(f"  Original object shape: {data_dict['objectGuess'].shape}")
    
    # Step 4: Update the data
    print("Updating 'objectGuess' and converting 'diff3d' to float32...")
    data_dict['objectGuess'] = new_object_guess
    
    if 'diff3d' in data_dict:
        original_dtype = data_dict['diff3d'].dtype
        data_dict['diff3d'] = data_dict['diff3d'].astype(np.float32)
        print(f"  Converted diff3d from {original_dtype} to float32")
    
    # Step 5: Save the new file
    print(f"Saving new NPZ file to {output_npz_path}...")
    np.savez_compressed(output_npz_path, **data_dict)
    
    # Verify the save
    file_size_mb = os.path.getsize(output_npz_path) / (1024 * 1024)
    print(f"Done! Created {output_npz_path} ({file_size_mb:.1f} MB)")


# ===================================================================
# Command-Line Interface
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update the 'objectGuess' in an NPZ file with a new reconstruction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update NPZ with reconstruction from .npy file
  python update_tool.py data/original.npz results/recon.npy data/updated.npz
  
  # Can also be used in a Python script:
  from update_tool import update_object_guess
  update_object_guess('original.npz', reconstructed_array, 'updated.npz')
        """
    )
    
    parser.add_argument(
        "original_npz",
        help="Path to the original .npz file"
    )
    parser.add_argument(
        "reconstruction_npy",
        help="Path to the .npy file containing the new object reconstruction"
    )
    parser.add_argument(
        "output_npz",
        help="Path where the new, updated .npz file will be saved"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Call the core function with the command-line arguments
        update_object_guess(
            args.original_npz,
            args.reconstruction_npy,
            args.output_npz
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)