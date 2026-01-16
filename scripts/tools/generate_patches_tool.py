#!/usr/bin/env python3
"""
Ptychography Patch Generation Tool

This tool generates Y patches from objectGuess using the RawData class's
generate_grouped_data() method. It loads an NPZ file, instantiates a RawData
object, generates the Y patches, and saves a new NPZ file with all original
data plus the newly generated Y array.

This is essential for supervised learning workflows that require pre-computed
ground truth patches.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from ptycho.raw_data import RawData

def generate_patches(
    input_path: Path,
    output_path: Path,
    patch_size: int = 64,
    k_neighbors: int = 7,
    nsamples: int = 1
):
    """Generate Y patches from objectGuess and save to new NPZ file.
    
    Args:
        input_path: Path to the source .npz file containing objectGuess
        output_path: Path to save the new .npz file with Y patches
        patch_size: Size of the patches (N parameter for generate_grouped_data)
        k_neighbors: Number of nearest neighbors (K parameter)
        nsamples: Number of samples per coordinate group
        
    Raises:
        FileNotFoundError: If the input file does not exist
        KeyError: If required keys are missing from the NPZ file
        ValueError: If the data format is invalid
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"--- Generating Patches: {input_path} ---")
    print(f"  - Output file: {output_path}")
    print(f"  - Patch size (N): {patch_size}")
    print(f"  - K neighbors: {k_neighbors}")
    print(f"  - Samples: {nsamples}")
    
    # Load the input NPZ file
    with np.load(input_path) as data:
        data_dict = {key: data[key] for key in data.files}
    
    print(f"  - Loaded {len(data_dict)} arrays from input file")
    
    # Validate required keys - support both 'diffraction' and 'diff3d'
    required_keys = ['xcoords', 'ycoords', 'probeGuess', 'objectGuess']
    missing_keys = [key for key in required_keys if key not in data_dict]
    if missing_keys:
        raise KeyError(f"Missing required keys: {missing_keys}")
    
    # Check for diffraction data (support both naming conventions)
    if 'diffraction' in data_dict:
        diffraction = data_dict['diffraction']
    elif 'diff3d' in data_dict:
        diffraction = data_dict['diff3d']
    else:
        raise KeyError("Missing diffraction data: need either 'diffraction' or 'diff3d'")
    
    # Extract data arrays
    xcoords = data_dict['xcoords']
    ycoords = data_dict['ycoords']
    probeGuess = data_dict['probeGuess']
    objectGuess = data_dict['objectGuess']
    
    # Validate data shapes
    n_scans = len(xcoords)
    if len(ycoords) != n_scans:
        raise ValueError(f"xcoords and ycoords must have same length. Got {len(xcoords)} and {len(ycoords)}")
    if diffraction.shape[0] != n_scans:
        raise ValueError(f"diffraction patterns must match coordinate count. Got {diffraction.shape[0]} patterns, {n_scans} coordinates")
    
    print(f"  - Coordinates: {n_scans} scan positions")
    print(f"  - Diffraction shape: {diffraction.shape}")
    print(f"  - ProbeGuess shape: {probeGuess.shape}")
    print(f"  - ObjectGuess shape: {objectGuess.shape}")
    
    # Extract or create scan_index array
    if 'scan_index' in data_dict:
        scan_index = data_dict['scan_index']
    else:
        scan_index = np.arange(n_scans)
    
    # Extract start coordinates if available
    if 'xcoords_start' in data_dict and 'ycoords_start' in data_dict:
        xcoords_start = data_dict['xcoords_start']
        ycoords_start = data_dict['ycoords_start']
        # Use full RawData constructor
        raw_data = RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords_start,
            ycoords_start=ycoords_start,
            diff3d=diffraction,
            probeGuess=probeGuess,
            scan_index=scan_index,
            objectGuess=objectGuess
        )
    else:
        # Use from_coords_without_pc since we don't have separate start coordinates
        raw_data = RawData.from_coords_without_pc(
            xcoords=xcoords,
            ycoords=ycoords,
            diff3d=diffraction,
            probeGuess=probeGuess,
            scan_index=scan_index,
            objectGuess=objectGuess
        )
    
    print("  - Created RawData object successfully")
    
    # Generate grouped data with Y patches
    print(f"  - Generating Y patches...")
    print(f"  - Using gridsize=1 for PtychoPINN compatibility")
    grouped_data = raw_data.generate_grouped_data(
        N=patch_size,
        K=k_neighbors,
        nsamples=nsamples,
        gridsize=1
    )
    
    if 'Y' not in grouped_data:
        raise ValueError("generate_grouped_data() did not produce Y patches")
    
    Y_patches = grouped_data['Y']
    print(f"  - Generated Y patches with shape: {Y_patches.shape}")
    print(f"  - Y patches are correctly generated with 3D shape for PINN compatibility")
    
    # Create output dictionary with all original data plus Y patches
    output_dict = data_dict.copy()
    output_dict['Y'] = Y_patches
    
    # Save additional metadata from grouped_data if available
    for key in ['coords_start_offsets', 'coords_start_relative']:
        if key in grouped_data:
            output_dict[key] = grouped_data[key]
            print(f"  - Added {key} with shape: {grouped_data[key].shape}")
    
    # Save the output file
    print(f"Saving prepared dataset with Y patches to: {output_path}")
    np.savez_compressed(output_path, **output_dict)
    
    # Verify the output
    with np.load(output_path) as verify_data:
        verify_keys = verify_data.files
        print(f"  - Verified output contains {len(verify_keys)} arrays")
        if 'Y' in verify_keys:
            print(f"  - Y array shape: {verify_data['Y'].shape}")
        else:
            raise ValueError("Y array missing from output file")
    
    print("--- Patch Generation Complete ---")

def main():
    parser = argparse.ArgumentParser(
        description="Generate Y patches from objectGuess for supervised learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate patches with default settings
    python generate_patches_tool.py input.npz output.npz
    
    # Generate patches with custom parameters
    python generate_patches_tool.py input.npz output.npz --patch-size 128 --k-neighbors 5
        """
    )
    parser.add_argument("input_npz", type=Path, help="Path to the source .npz file containing objectGuess")
    parser.add_argument("output_npz", type=Path, help="Path to save the new .npz file with Y patches")
    parser.add_argument("--patch-size", type=int, default=64,
                        help="Size of the patches (N parameter, default: 64)")
    parser.add_argument("--k-neighbors", type=int, default=7,
                        help="Number of nearest neighbors (K parameter, default: 7)")
    parser.add_argument("--nsamples", type=int, default=1,
                        help="Number of samples per coordinate group (default: 1)")
    
    args = parser.parse_args()
    
    try:
        generate_patches(
            args.input_npz,
            args.output_npz,
            args.patch_size,
            args.k_neighbors,
            args.nsamples
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()