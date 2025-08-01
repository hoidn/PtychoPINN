#!/usr/bin/env python3
"""
Generate validation subsets for quick testing and validation experiments.

This script creates smaller datasets (1000 images) with train/test splits:
- Sequential subsets: From fly001_64_train_converted.npz (preserves spatial locality for gridsize > 1)
- Random subsets: From fly64_shuffled.npz (full spatial coverage for gridsize = 1)
"""

import numpy as np
import argparse
import os
from pathlib import Path


def create_subset(data_dict, indices, desc="subset"):
    """Create a subset of the data using given indices."""
    subset = {}
    for key, value in data_dict.items():
        if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == len(data_dict['xcoords']):
            # This is a per-image array
            subset[key] = value[indices]
            print(f"  {key}: {value.shape} -> {subset[key].shape}")
        else:
            # This is a scalar or metadata
            subset[key] = value
    
    print(f"Created {desc} with {len(indices)} images")
    return subset


def main():
    parser = argparse.ArgumentParser(description="Generate validation subsets for PtychoPINN")
    parser.add_argument("--output-dir", type=str, default="datasets/fly64",
                      help="Output directory for validation subsets")
    parser.add_argument("--total-images", type=int, default=1000,
                      help="Total images to extract (default: 1000)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                      help="Ratio for train/test split (default: 0.8)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_total = args.total_images
    n_train = int(n_total * args.train_ratio)
    n_test = n_total - n_train
    
    print(f"Generating validation subsets:")
    print(f"  Total images: {n_total}")
    print(f"  Train images: {n_train}")
    print(f"  Test images: {n_test}")
    print()
    
    # Generate sequential subsets (for gridsize > 1)
    print("="*60)
    print("Generating SEQUENTIAL subsets (for gridsize > 1)")
    print("="*60)
    
    sequential_source = output_dir / "fly001_64_train_converted.npz"
    if not sequential_source.exists():
        print(f"ERROR: Source file not found: {sequential_source}")
        return 1
    
    print(f"Loading source: {sequential_source}")
    data = np.load(sequential_source)
    print(f"Source contains {len(data['xcoords'])} total images")
    
    # Take first 1000 images (sequential order preserves spatial locality)
    train_indices = np.arange(n_train)
    test_indices = np.arange(n_train, n_total)
    
    # Create train subset
    print(f"\nCreating sequential train subset...")
    train_subset = create_subset(dict(data), train_indices, "train subset")
    
    # Check spatial coverage
    x_range = (train_subset['xcoords'].min(), train_subset['xcoords'].max())
    y_range = (train_subset['ycoords'].min(), train_subset['ycoords'].max())
    print(f"  Spatial coverage: X={x_range}, Y={y_range}")
    
    # Save train subset
    train_path = output_dir / "fly64_sequential_train_800.npz"
    np.savez_compressed(train_path, **train_subset)
    print(f"  Saved to: {train_path}")
    
    # Create test subset
    print(f"\nCreating sequential test subset...")
    test_subset = create_subset(dict(data), test_indices, "test subset")
    
    # Check spatial coverage
    x_range = (test_subset['xcoords'].min(), test_subset['xcoords'].max())
    y_range = (test_subset['ycoords'].min(), test_subset['ycoords'].max())
    print(f"  Spatial coverage: X={x_range}, Y={y_range}")
    
    # Save test subset
    test_path = output_dir / "fly64_sequential_test_200.npz"
    np.savez_compressed(test_path, **test_subset)
    print(f"  Saved to: {test_path}")
    
    # Generate random subsets (for gridsize = 1)
    print("\n" + "="*60)
    print("Generating RANDOM subsets (for gridsize = 1)")
    print("="*60)
    
    random_source = output_dir / "fly64_shuffled.npz"
    if not random_source.exists():
        print(f"ERROR: Source file not found: {random_source}")
        return 1
    
    print(f"Loading source: {random_source}")
    data = np.load(random_source)
    print(f"Source contains {len(data['xcoords'])} total images")
    
    # Verify it's been shuffled
    if '_shuffle_applied' in data and data['_shuffle_applied']:
        print(f"  ✓ Dataset has been shuffled (seed: {data.get('_shuffle_seed', 'unknown')})")
    else:
        print(f"  WARNING: Dataset may not be shuffled")
    
    # Take first 1000 images (already randomized)
    train_indices = np.arange(n_train)
    test_indices = np.arange(n_train, n_total)
    
    # Create train subset
    print(f"\nCreating random train subset...")
    train_subset = create_subset(dict(data), train_indices, "train subset")
    
    # Check spatial coverage
    x_range = (train_subset['xcoords'].min(), train_subset['xcoords'].max())
    y_range = (train_subset['ycoords'].min(), train_subset['ycoords'].max())
    print(f"  Spatial coverage: X={x_range}, Y={y_range}")
    
    # Save train subset
    train_path = output_dir / "fly64_random_train_800.npz"
    np.savez_compressed(train_path, **train_subset)
    print(f"  Saved to: {train_path}")
    
    # Create test subset
    print(f"\nCreating random test subset...")
    test_subset = create_subset(dict(data), test_indices, "test subset")
    
    # Check spatial coverage
    x_range = (test_subset['xcoords'].min(), test_subset['xcoords'].max())
    y_range = (test_subset['ycoords'].min(), test_subset['ycoords'].max())
    print(f"  Spatial coverage: X={y_range}, Y={y_range}")
    
    # Save test subset
    test_path = output_dir / "fly64_random_test_200.npz"
    np.savez_compressed(test_path, **test_subset)
    print(f"  Saved to: {test_path}")
    
    print("\n" + "="*60)
    print("✓ All validation subsets generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()