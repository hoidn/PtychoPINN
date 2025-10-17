#!/usr/bin/env python3
"""
Investigate data scaling issues in the photon datasets.
"""

import numpy as np
import os

def analyze_dataset(filepath):
    """Analyze a dataset file and print key statistics."""
    print(f"\n=== Analyzing {os.path.basename(filepath)} ===")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    data = np.load(filepath)
    print(f"Keys in file: {list(data.keys())}")
    
    # Check diffraction data
    if 'diff3d' in data:
        diff_data = data['diff3d']
        print(f"diff3d shape: {diff_data.shape}")
        print(f"diff3d dtype: {diff_data.dtype}")
        print(f"diff3d min: {diff_data.min()}")
        print(f"diff3d max: {diff_data.max()}")
        print(f"diff3d mean: {diff_data.mean()}")
        print(f"diff3d std: {diff_data.std()}")
        
        # Sample values from first diffraction pattern
        print(f"Sample values from diff3d[0, :5, :5]:")
        print(diff_data[0, :5, :5])
        
        # Check for any zero or negative values
        zero_count = np.sum(diff_data == 0)
        negative_count = np.sum(diff_data < 0)
        print(f"Zero values: {zero_count}/{diff_data.size} ({100*zero_count/diff_data.size:.2f}%)")
        print(f"Negative values: {negative_count}/{diff_data.size} ({100*negative_count/diff_data.size:.2f}%)")
        
    # Check other arrays
    for key in data.keys():
        if key != 'diff3d':
            arr = data[key]
            print(f"{key} shape: {arr.shape}, dtype: {arr.dtype}")
            if np.issubdtype(arr.dtype, np.number):
                print(f"  min: {arr.min()}, max: {arr.max()}, mean: {arr.mean()}")

def compare_datasets(file1, file2):
    """Compare statistics between two datasets."""
    print(f"\n=== Comparing {os.path.basename(file1)} vs {os.path.basename(file2)} ===")
    
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    if 'diff3d' in data1 and 'diff3d' in data2:
        diff1 = data1['diff3d']
        diff2 = data2['diff3d']
        
        print(f"Dataset 1 (1e4): mean={diff1.mean():.6e}, max={diff1.max():.6e}")
        print(f"Dataset 2 (1e9): mean={diff2.mean():.6e}, max={diff2.max():.6e}")
        
        ratio_mean = diff2.mean() / diff1.mean()
        ratio_max = diff2.max() / diff1.max()
        print(f"Mean ratio (1e9/1e4): {ratio_mean:.2e}")
        print(f"Max ratio (1e9/1e4): {ratio_max:.2e}")
        
        expected_ratio = 1e9 / 1e4  # Should be 1e5
        print(f"Expected ratio: {expected_ratio:.2e}")
        
        # Compare same pixel locations
        print(f"Pixel [0,0,0] - 1e4: {diff1[0,0,0]:.6e}, 1e9: {diff2[0,0,0]:.6e}")
        print(f"Pixel [0,10,10] - 1e4: {diff1[0,10,10]:.6e}, 1e9: {diff2[0,10,10]:.6e}")

if __name__ == "__main__":
    # Analyze both datasets
    file1 = "photon_1e4_4k_images_CORRECTED.npz"
    file2 = "photon_1e9_4k_images_CORRECTED.npz"
    
    analyze_dataset(file1)
    analyze_dataset(file2)
    compare_datasets(file1, file2)