#!/usr/bin/env python3
"""
Focused test script that directly tests the oversampling logic with exact parameters.

This script bypasses all configuration layers and directly tests if the oversampling
logic works with nsamples=1024, K=7, gridsize=2, using only 128 images.
"""

import numpy as np
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/ollie/Documents/PtychoPINN')

# Set up verbose logging to see everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)

from ptycho.raw_data import RawData
from ptycho import params

def test_oversampling_focused():
    """
    Focused test of oversampling logic with exact parameters.
    """
    print("=" * 80)
    print("FOCUSED OVERSAMPLING TEST")
    print("=" * 80)
    
    # Step 1: Load actual train data
    train_data_path = '/home/ollie/Documents/PtychoPINN/datasets/fly64/fly001_64_train.npz'
    print(f"\n1. Loading train data from: {train_data_path}")
    
    if not os.path.exists(train_data_path):
        print(f"ERROR: Train data file not found at {train_data_path}")
        # Try alternative
        train_data_path = '/home/ollie/Documents/PtychoPINN/datasets/fly64/fly001_64_prepared_final_train.npz'
        print(f"Trying alternative: {train_data_path}")
        
    if not os.path.exists(train_data_path):
        print("ERROR: No train data file found. Exiting.")
        return
    
    data = np.load(train_data_path)
    print(f"   Keys in dataset: {list(data.keys())}")
    
    # Step 2: Subsample to exactly 128 images
    diffraction_key = 'diff3d' if 'diff3d' in data else 'diffraction'
    n_original = len(data[diffraction_key])
    n_subset = 128
    print(f"\n2. Subsampling from {n_original} to {n_subset} images")
    
    # Take first 128 images to be deterministic
    subset_indices = np.arange(n_subset)
    
    subset_data = {
        diffraction_key: data[diffraction_key][subset_indices],
        'xcoords': data['xcoords'][subset_indices],
        'ycoords': data['ycoords'][subset_indices],
        'objectGuess': data['objectGuess'],
        'probeGuess': data['probeGuess']
    }
    
    # Add optional fields if they exist
    if 'Y' in data:
        subset_data['Y'] = data['Y'][subset_indices]
    if 'xcoords_start' in data:
        subset_data['xcoords_start'] = data['xcoords_start'][subset_indices]
    if 'ycoords_start' in data:
        subset_data['ycoords_start'] = data['ycoords_start'][subset_indices]
    if 'scan_index' in data:
        subset_data['scan_index'] = data['scan_index'][subset_indices]
    else:
        # Create a default scan_index if not present
        subset_data['scan_index'] = np.zeros(n_subset, dtype=int)
    
    print(f"   Subset diffraction shape: {subset_data[diffraction_key].shape}")
    print(f"   Subset coordinates shape: {subset_data['xcoords'].shape}")
    
    # Step 3: Create RawData object
    print(f"\n3. Creating RawData object")
    raw_data = RawData(
        xcoords=subset_data['xcoords'],
        ycoords=subset_data['ycoords'],
        xcoords_start=subset_data['xcoords_start'],
        ycoords_start=subset_data['ycoords_start'],
        diff3d=subset_data[diffraction_key],
        probeGuess=subset_data['probeGuess'],
        scan_index=subset_data['scan_index'],
        objectGuess=subset_data['objectGuess'],
        Y=subset_data.get('Y', None)
    )
    print(f"   RawData created successfully")
    print(f"   Number of images in RawData: {len(raw_data.xcoords)}")
    
    # Step 4: Set up parameters for the exact test case
    print(f"\n4. Setting up parameters")
    nsamples = 1024
    K = 7
    gridsize = 2
    N = 64  # Typical for fly64 data
    
    # Set gridsize in global params (needed by the function)
    params.set('gridsize', gridsize)
    params.set('N', N)
    
    print(f"   nsamples: {nsamples}")
    print(f"   K: {K}")
    print(f"   gridsize: {gridsize}")
    print(f"   N: {N}")
    print(f"   C (gridsize²): {gridsize**2}")
    print(f"   Available images: {len(raw_data.xcoords)}")
    
    # Step 5: Check oversampling conditions
    print(f"\n5. Oversampling condition check")
    C = gridsize ** 2
    n_points = len(raw_data.xcoords)
    
    needs_oversampling = (nsamples > n_points) and (C > 1)
    print(f"   nsamples > n_points: {nsamples} > {n_points} = {nsamples > n_points}")
    print(f"   C > 1: {C} > 1 = {C > 1}")
    print(f"   needs_oversampling: {needs_oversampling}")
    
    if needs_oversampling:
        print("   ✓ OVERSAMPLING SHOULD BE TRIGGERED")
    else:
        print("   ✗ OVERSAMPLING WILL NOT BE TRIGGERED")
    
    # Step 6: Call generate_grouped_data and capture all output
    print(f"\n6. Calling generate_grouped_data with verbose logging")
    print("-" * 60)
    
    try:
        result = raw_data.generate_grouped_data(
            N=N,
            K=K,
            nsamples=nsamples,
            seed=42  # For reproducibility
        )
        
        print("-" * 60)
        print("✓ generate_grouped_data completed successfully!")
        
        # Step 7: Analyze results
        print(f"\n7. Analyzing results")
        print(f"   Result keys: {list(result.keys())}")
        
        if 'diffraction' in result:
            diff_shape = result['diffraction'].shape
            print(f"   Diffraction shape: {diff_shape}")
            print(f"   Expected: ({nsamples}, N, N, {C}) = ({nsamples}, {N}, {N}, {C})")
            
            # Check if we got the right number of groups
            if diff_shape[0] == nsamples:
                print("   ✓ CORRECT: Got exactly the requested number of groups!")
            else:
                print(f"   ✗ ERROR: Expected {nsamples} groups, got {diff_shape[0]}")
        
        if 'X_full' in result:
            x_shape = result['X_full'].shape
            print(f"   X_full shape: {x_shape}")
        
        if 'nn_indices' in result:
            nn_shape = result['nn_indices'].shape
            print(f"   nn_indices shape: {nn_shape}")
            print(f"   nn_indices type: {type(result['nn_indices'])}")
            
            # Show some sample indices to verify they're valid
            sample_indices = result['nn_indices'][:5].flatten()
            print(f"   Sample indices: {sample_indices}")
            print(f"   Max index: {np.max(result['nn_indices'])}, should be < {n_points}")
            
            if np.max(result['nn_indices']) < n_points:
                print("   ✓ All indices are valid (within bounds)")
            else:
                print("   ✗ Some indices are out of bounds!")
        
        # Step 8: Summary
        print(f"\n8. TEST SUMMARY")
        print("=" * 40)
        
        success_criteria = [
            ("Oversampling triggered", needs_oversampling),
            ("Function completed", True),
            ("Got requested groups", result['diffraction'].shape[0] == nsamples if 'diffraction' in result else False),
            ("Valid indices", np.max(result['nn_indices']) < n_points if 'nn_indices' in result else False)
        ]
        
        all_passed = True
        for criterion, passed in success_criteria:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"   {criterion}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Oversampling logic works correctly.")
        else:
            print("\n❌ SOME TESTS FAILED. Check the output above.")
        
        return result
        
    except Exception as e:
        print("-" * 60)
        print(f"❌ ERROR: generate_grouped_data failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        print(f"\nFull traceback:")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_oversampling_focused()
    
    print(f"\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    if result is not None:
        print("✓ Test completed with results")
    else:
        print("❌ Test failed with errors")