#!/usr/bin/env python3
"""
Ptychography Data Canonicalization Tool

This tool processes an NPZ file to create a standardized version.

Its primary purpose is to ensure that the output file conforms to the
specifications laid out in the project's official Data Contracts document.
See `docs/data_contracts.md` for details on required shapes and keys.

Key Operations:
1. Renames 'diff3d' to 'diffraction'.
2. Squeezes the last dimension of the 'Y' array to ensure it is 3D.
3. Converts uint16 arrays to float32.
4. Optionally flips X and/or Y coordinates.
"""

import sys
from pathlib import Path
import numpy as np
import argparse

def transpose_rename_convert(
    in_file: str | Path,
    out_file: str | Path,
    flipx: bool = False,
    flipy: bool = False,
) -> None:
    """Loads, processes, and saves a ptychography NPZ file to a canonical format.

    This function standardizes a ptychography dataset to conform to the project's
    Data Contracts specification. It performs key renaming, data type conversion,
    array reshaping, and optional coordinate transformations.

    Args:
        in_file: Path to the source .npz file to be processed.
        out_file: Path where the canonicalized .npz file will be saved.
        flipx: If True, flips the sign of all x-coordinates (default: False).
        flipy: If True, flips the sign of all y-coordinates (default: False).

    Raises:
        FileNotFoundError: If the input file does not exist.
        
    Operations performed:
        - Renames 'diff3d' key to 'diffraction'
        - Squeezes 4D 'Y' arrays to 3D by removing singleton last dimension
        - Converts uint16 arrays to float32 for numerical stability
        - Optionally flips coordinate signs for coordinate system alignment
    """
    in_path = Path(in_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"Loading data from: {in_path}")
    npz = np.load(in_path, allow_pickle=False)
    save_dict: dict[str, np.ndarray] = {}

    print(f"Processing {len(npz.files)} arrays...")
    
    for key in npz.files:
        arr = npz[key].copy()  # Work on a copy
        new_key = key

        # 1. Rename 'diff3d' to 'diffraction'
        if key == "diff3d":
            print(f"  - Renaming 'diff3d' to 'diffraction'")
            new_key = "diffraction"
        
        # 2. Process 'Y' array (ground truth patches)
        elif key == "Y":
            if arr.ndim == 4 and arr.shape[-1] == 1:
                print(f"  - Squeezing 'Y' array from {arr.shape} to {arr.squeeze(axis=-1).shape}")
                arr = arr.squeeze(axis=-1)

        # 3. Convert uint16 to float32
        if arr.dtype == np.uint16:
            print(f"  - Converting '{key}' from uint16 to float32")
            arr = arr.astype(np.float32)

        # 4. Flip coordinates if requested
        if flipx and key.lower() in ("xcoords", "xcoords_start"):
            print(f"  - Flipping x-coordinates in '{key}'")
            arr = -arr
        if flipy and key.lower() in ("ycoords", "ycoords_start"):
            print(f"  - Flipping y-coordinates in '{key}'")
            arr = -arr

        save_dict[new_key] = arr

    np.savez_compressed(out_file, **save_dict)
    print(f"✔ Saved consistently formatted data to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Canonicalize a ptychography NPZ dataset.")
    parser.add_argument("input_npz", help="Path to the source .npz file.")
    parser.add_argument("output_npz", help="Path to save the processed .npz file.")
    parser.add_argument("--flipx", action="store_true", help="Flip the sign of all x-coordinates.")
    parser.add_argument("--flipy", action="store_true", help="Flip the sign of all y-coordinates.")
    args = parser.parse_args()

    try:
        transpose_rename_convert(args.input_npz, args.output_npz, args.flipx, args.flipy)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()