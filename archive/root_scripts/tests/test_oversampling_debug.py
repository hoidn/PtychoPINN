#!/usr/bin/env python3
"""
Minimal test script to debug the oversampling issue.

This script reproduces the exact same parameters as the failing case to isolate
whether the issue is in the shell script or the Python data loading pipeline.
"""

import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, '/home/ollie/Documents/PtychoPINN')

from ptycho.workflows.components import load_data
from ptycho.loader import PtychoDataContainer


def main():
    print("=== Oversampling Debug Test ===")
    print()
    
    # Exact parameters from the failing case
    params = {
        'n_subsample': 128,
        'n_groups': 1024,
        'neighbor_count': 7,
        'gridsize': 2
    }
    
    # Data files from the failing case
    train_data_file = "prepare_1e4_photons_5k/dataset/train.npz"
    test_data_file = "prepare_1e4_photons_5k/dataset/test.npz"
    
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    print("Data files:")
    print(f"  train_data: {train_data_file}")
    print(f"  test_data: {test_data_file}")
    print()
    
    # Check if files exist
    if not os.path.exists(train_data_file):
        print(f"ERROR: Train data file does not exist: {train_data_file}")
        return
    if not os.path.exists(test_data_file):
        print(f"ERROR: Test data file does not exist: {test_data_file}")
        return
    
    print("Files exist - proceeding with data loading...")
    print()
    
    try:
        # Step 1: Load the raw data
        print("=== Step 1: Loading raw data ===")
        raw_train = load_data(train_data_file)
        raw_test = load_data(test_data_file)
        
        print(f"Raw train data shape: {raw_train.diff3d.shape}")
        print(f"Raw test data shape: {raw_test.diff3d.shape}")
        print()
        
        # Step 2: Mimic the exact training workflow
        print("=== Step 2: Following exact training workflow ===")
        print("This is where the oversampling issue would occur...")
        
        # Mimic the exact call from train.py line 132-137
        print("Calling load_data with n_images=n_groups for API compatibility...")
        ptycho_data = load_data(
            train_data_file, 
            n_images=params['n_groups'],  # Pass n_groups as n_images to maintain API compatibility
            n_subsample=params['n_subsample'],
            subsample_seed=42
        )
        
        print("load_data completed successfully!")
        print(f"Loaded ptycho_data type: {type(ptycho_data)}")
        print(f"Loaded train data shape: {ptycho_data.diff3d.shape}")
        print()
        
        # Load test data
        test_data = load_data(test_data_file)
        print(f"Loaded test data shape: {test_data.diff3d.shape}")
        print()
        
        # Step 3: Analyze what happened with the data loading
        print("=== Step 3: Data Loading Analysis ===")
        original_train_images = raw_train.diff3d.shape[0]
        loaded_train_images = ptycho_data.diff3d.shape[0]
        
        print(f"Original train images: {original_train_images}")
        print(f"Loaded train images after load_data: {loaded_train_images}")
        print(f"Requested n_groups: {params['n_groups']}")
        print(f"Requested n_subsample: {params['n_subsample']}")
        print()
        
        if loaded_train_images > original_train_images:
            print("✓ OVERSAMPLING DETECTED in load_data!")
            print(f"  Data was expanded from {original_train_images} to {loaded_train_images} images")
            oversample_factor = loaded_train_images / original_train_images
            print(f"  Oversampling factor: {oversample_factor:.2f}x")
        elif loaded_train_images < original_train_images:
            print("✓ SUBSAMPLING DETECTED in load_data!")
            print(f"  Data was reduced from {original_train_images} to {loaded_train_images} images")
            subsample_factor = loaded_train_images / original_train_images
            print(f"  Subsampling factor: {subsample_factor:.2f}x")
        else:
            print("✓ NO SAMPLING CHANGE in load_data")
            print(f"  Data size unchanged: {original_train_images} images")
        print()
        
        # Step 4: Check the logic conditions
        print("=== Step 4: Logic condition analysis ===")
        print("Checking the conditions that might trigger oversampling...")
        print()
        
        # The key condition from the error: n_subsample < n_groups
        print(f"n_subsample ({params['n_subsample']}) < n_groups ({params['n_groups']}): {params['n_subsample'] < params['n_groups']}")
        print(f"Original images ({original_train_images}) < n_groups ({params['n_groups']}): {original_train_images < params['n_groups']}")
        print(f"Loaded images ({loaded_train_images}) < n_groups ({params['n_groups']}): {loaded_train_images < params['n_groups']}")
        print()
        
        if params['n_subsample'] < params['n_groups']:
            print("⚠️  n_subsample < n_groups condition is met!")
            print("   This is the main condition that triggers oversampling issues.")
            if loaded_train_images < params['n_groups']:
                print("   AND the loaded dataset still has fewer images than n_groups.")
                print("   This will likely cause oversampling issues downstream.")
            else:
                print("   But loaded dataset now has enough images for n_groups.")
        else:
            print("ℹ️  n_subsample >= n_groups - no oversampling condition.")
        
        print()
        print("=== Step 5: Understanding the Parameter Conflict ===")
        print(f"The issue is that n_subsample={params['n_subsample']} < n_groups={params['n_groups']}")
        print("This means:")
        print(f"  1. We ask to subsample only {params['n_subsample']} images")
        print(f"  2. But then we ask to create {params['n_groups']} groups")
        print(f"  3. Since {params['n_subsample']} < {params['n_groups']}, we need oversampling!")
        print()
        print("This is a fundamental parameter conflict.")
        print("Solution: Either increase n_subsample or decrease n_groups.")
        
        print()
        print("=== Test completed successfully! ===")
        
    except Exception as e:
        print(f"ERROR during data loading: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()