#!/usr/bin/env python3
"""
Modify coordinate variants to remove ground truth objectGuess.
This forces the model to rely on the coordinate system logic.
"""

import numpy as np
import glob
import os

def modify_variants():
    """
    Load each coordinate variant and set objectGuess to None.
    """
    variants_dir = 'datasets/fly64_coord_variants'
    dataset_paths = sorted(glob.glob(os.path.join(variants_dir, '*.npz')))
    
    if not dataset_paths:
        print(f"ERROR: No dataset variants found in {variants_dir}")
        return
    
    print(f"Found {len(dataset_paths)} variants to modify")
    
    for path in dataset_paths:
        filename = os.path.basename(path)
        print(f"Processing: {filename}")
        
        # Load the data
        data = np.load(path)
        
        # Create a mutable copy and remove objectGuess
        variant_data = dict(data)
        
        # Check if objectGuess exists and remove it
        if 'objectGuess' in variant_data:
            print(f"  Removing objectGuess (shape: {variant_data['objectGuess'].shape})")
            del variant_data['objectGuess']
        else:
            print(f"  No objectGuess found in {filename}")
        
        # Also remove any other ground truth fields that might exist
        gt_fields = ['Y', 'Y_obj', 'Y_I', 'Y_phi', 'norm_Y_I']
        for field in gt_fields:
            if field in variant_data:
                print(f"  Removing {field}")
                del variant_data[field]
        
        # Save back to the same file
        np.savez(path, **variant_data)
        print(f"  Saved modified dataset to {path}")
    
    print("All variants modified successfully!")

if __name__ == '__main__':
    modify_variants()