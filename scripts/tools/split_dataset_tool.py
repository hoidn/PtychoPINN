#!/usr/bin/env python3
"""
Ptychography Dataset Splitting Tool

This tool splits a single NPZ dataset into non-overlapping training and testing
sets based on the spatial coordinates of the scan positions.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

def split_dataset(
    input_path: Path,
    output_dir: Path,
    split_fraction: float,
    split_axis: str,
):
    """Loads an NPZ, splits it, and saves two new NPZ files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"--- Splitting Dataset: {input_path} ---")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Split fraction (train): {split_fraction:.2f}")
    print(f"  - Split axis: '{split_axis}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    with np.load(input_path) as data:
        data_dict = {key: data[key] for key in data.files}

    # 1. Determine the split point
    coord_key = f"{split_axis}coords"
    if coord_key not in data_dict:
        raise KeyError(f"Coordinate key '{coord_key}' not found in NPZ file.")
    
    coords = data_dict[coord_key]
    min_coord, max_coord = coords.min(), coords.max()
    split_point = min_coord + (max_coord - min_coord) * split_fraction
    
    print(f"  - Coordinate range on '{split_axis}' axis: [{min_coord:.2f}, {max_coord:.2f}]")
    print(f"  - Calculated split point: {split_point:.2f}")

    # 2. Create boolean masks for train and test sets
    train_mask = coords < split_point
    test_mask = ~train_mask
    n_total = len(coords)
    n_train = np.sum(train_mask)
    n_test = np.sum(test_mask)
    print(f"  - Total positions: {n_total}. Train: {n_train}, Test: {n_test}")

    # 3. Create the two new data dictionaries
    train_dict, test_dict = {}, {}
    
    # Use 'xcoords' length as the reference for per-scan arrays
    n_scans = len(data_dict['xcoords'])

    for key, arr in data_dict.items():
        if arr.shape[0] == n_scans:
            # This is a per-scan array, so we split it
            train_dict[key] = arr[train_mask]
            test_dict[key] = arr[test_mask]
        else:
            # This is a global array (like objectGuess), so copy it to both
            train_dict[key] = arr
            test_dict[key] = arr

    # 4. Save the new files
    train_path = output_dir / f"{input_path.stem}_train.npz"
    test_path = output_dir / f"{input_path.stem}_test.npz"

    print(f"Saving training set ({len(train_dict['xcoords'])} scans) to: {train_path}")
    np.savez_compressed(train_path, **train_dict)
    
    print(f"Saving testing set ({len(test_dict['xcoords'])} scans) to: {test_path}")
    np.savez_compressed(test_path, **test_dict)
    
    print("--- Splitting Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Split a ptychography NPZ dataset into train and test sets.")
    parser.add_argument("input_npz", type=Path, help="Path to the source .npz file.")
    parser.add_argument("output_dir", type=Path, help="Directory to save the train and test .npz files.")
    parser.add_argument("--split-fraction", type=float, default=0.8,
                        help="Fraction of the data to use for the training set (e.g., 0.8 for an 80/20 split).")
    parser.add_argument("--split-axis", choices=['x', 'y'], default='y',
                        help="The spatial axis along which to perform the split ('y' for top/bottom, 'x' for left/right).")
    args = parser.parse_args()

    try:
        split_dataset(args.input_npz, args.output_dir, args.split_fraction, args.split_axis)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()