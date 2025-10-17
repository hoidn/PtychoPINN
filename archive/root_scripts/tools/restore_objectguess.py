#!/usr/bin/env python3
"""
Restore objectGuess to a specific coordinate variant dataset.
"""

import numpy as np
import os

def restore_objectguess_to_variant(variant_name):
    """
    Restore objectGuess to a specific coordinate variant by copying it from the original dataset.
    """
    # Original dataset with objectGuess
    original_path = 'datasets/fly64/fly001_64_train_converted.npz'
    
    # Target variant dataset
    variant_path = f'datasets/fly64_coord_variants/fly001_64_train_converted_{variant_name}.npz'
    
    if not os.path.exists(original_path):
        print(f"ERROR: Original dataset not found: {original_path}")
        return
    
    if not os.path.exists(variant_path):
        print(f"ERROR: Variant dataset not found: {variant_path}")
        return
    
    print(f"Restoring objectGuess to {variant_name} variant...")
    
    # Load original data to get objectGuess
    original_data = np.load(original_path)
    objectGuess = original_data['objectGuess']
    print(f"Original objectGuess shape: {objectGuess.shape}")
    
    # Load variant data 
    variant_data = np.load(variant_path)
    print(f"Variant keys before: {list(variant_data.keys())}")
    
    # Create a new dataset with objectGuess restored
    new_data = dict(variant_data)
    new_data['objectGuess'] = objectGuess
    
    # Save back to the same file
    np.savez(variant_path, **new_data)
    
    # Verify
    restored_data = np.load(variant_path)
    print(f"Variant keys after: {list(restored_data.keys())}")
    print(f"Restored objectGuess shape: {restored_data['objectGuess'].shape}")
    print(f"SUCCESS: objectGuess restored to {variant_name}")

if __name__ == '__main__':
    restore_objectguess_to_variant('flip_xy')