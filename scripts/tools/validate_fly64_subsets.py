#!/usr/bin/env python3
"""
Validate the fly64 validation subsets to ensure they have correct properties.
"""

import numpy as np
from pathlib import Path


def validate_dataset(path, expected_size, dataset_type):
    """Validate a single dataset."""
    print(f"\nValidating {path.name} ({dataset_type})...")
    
    if not path.exists():
        print(f"  ❌ File not found!")
        return False
    
    data = np.load(path)
    
    # Check size
    actual_size = len(data['xcoords'])
    if actual_size != expected_size:
        print(f"  ❌ Wrong size: expected {expected_size}, got {actual_size}")
        return False
    else:
        print(f"  ✓ Correct size: {actual_size} images")
    
    # Check required keys
    required_keys = ['xcoords', 'ycoords', 'diffraction', 'probeGuess', 'objectGuess']
    for key in required_keys:
        if key not in data:
            print(f"  ❌ Missing required key: {key}")
            return False
    print(f"  ✓ All required keys present")
    
    # Check data types
    if data['diffraction'].dtype != np.float32:
        print(f"  ❌ Wrong diffraction dtype: {data['diffraction'].dtype} (expected float32)")
        return False
    else:
        print(f"  ✓ Correct diffraction dtype: float32")
    
    # Check spatial coverage
    x_range = data['xcoords'].max() - data['xcoords'].min()
    y_range = data['ycoords'].max() - data['ycoords'].min()
    print(f"  Spatial coverage: X range = {x_range:.1f}, Y range = {y_range:.1f}")
    
    if dataset_type == "sequential":
        # Sequential should have limited spatial coverage
        if y_range > 30:
            print(f"  ⚠️  Large Y range for sequential data: {y_range:.1f}")
        else:
            print(f"  ✓ Limited Y range appropriate for sequential/gridsize>1")
    else:
        # Random should have wide spatial coverage
        if x_range < 150 or y_range < 150:
            print(f"  ⚠️  Limited spatial coverage for random data")
        else:
            print(f"  ✓ Wide spatial coverage appropriate for random/gridsize=1")
    
    return True


def check_train_test_overlap(train_path, test_path):
    """Check if train and test sets have overlapping scan positions."""
    print(f"\nChecking train/test overlap...")
    
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    # Create unique position identifiers
    train_pos = set(zip(train_data['xcoords'], train_data['ycoords']))
    test_pos = set(zip(test_data['xcoords'], test_data['ycoords']))
    
    overlap = train_pos & test_pos
    if overlap:
        print(f"  ❌ Found {len(overlap)} overlapping positions!")
        return False
    else:
        print(f"  ✓ No overlap between train and test sets")
        return True


def main():
    dataset_dir = Path("datasets/fly64")
    
    print("="*60)
    print("Validating FLY64 Validation Subsets")
    print("="*60)
    
    all_valid = True
    
    # Validate sequential datasets
    print("\nSEQUENTIAL DATASETS (for gridsize > 1):")
    print("-"*40)
    
    seq_train = dataset_dir / "fly64_sequential_train_800.npz"
    seq_test = dataset_dir / "fly64_sequential_test_200.npz"
    
    all_valid &= validate_dataset(seq_train, 800, "sequential")
    all_valid &= validate_dataset(seq_test, 200, "sequential")
    all_valid &= check_train_test_overlap(seq_train, seq_test)
    
    # Validate random datasets
    print("\nRANDOM DATASETS (for gridsize = 1):")
    print("-"*40)
    
    rand_train = dataset_dir / "fly64_random_train_800.npz"
    rand_test = dataset_dir / "fly64_random_test_200.npz"
    
    all_valid &= validate_dataset(rand_train, 800, "random")
    all_valid &= validate_dataset(rand_test, 200, "random")
    all_valid &= check_train_test_overlap(rand_train, rand_test)
    
    # Final summary
    print("\n" + "="*60)
    if all_valid:
        print("✅ All validation subsets are correctly formatted!")
        print("\nYou can now use these datasets for quick validation:")
        print("\n# GridSize 2 (sequential):")
        print("ptycho_train --train-data datasets/fly64/fly64_sequential_train_800.npz \\")
        print("             --test-data datasets/fly64/fly64_sequential_test_200.npz \\")
        print("             --gridsize 2 --n-images 200 --epochs 10")
        print("\n# GridSize 1 (random):")
        print("ptycho_train --train-data datasets/fly64/fly64_random_train_800.npz \\")
        print("             --test-data datasets/fly64/fly64_random_test_200.npz \\")
        print("             --gridsize 1 --n-images 800 --epochs 10")
    else:
        print("❌ Some validation subsets have issues!")
    print("="*60)


if __name__ == "__main__":
    main()