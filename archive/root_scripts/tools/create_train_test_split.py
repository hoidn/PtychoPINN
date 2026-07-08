#!/usr/bin/env python3
"""
Simple train/test split for probe generalization study.
Split based on first N images for train, remaining for test.
"""
import numpy as np
import argparse

def split_dataset(input_file, train_size, test_size, train_output, test_output):
    """Split dataset into train and test sets by image index."""
    print(f"Loading {input_file}...")
    data = np.load(input_file)
    
    total_images = data['diff3d'].shape[0]
    print(f"Total images: {total_images}")
    print(f"Requested: {train_size} train, {test_size} test")
    
    if train_size + test_size > total_images:
        raise ValueError(f"Requested {train_size + test_size} images, but only {total_images} available")
    
    # Create train set (first train_size images)
    train_data = {}
    test_data = {}
    
    # Copy arrays that are per-image  
    per_image_keys = ['diff3d', 'xcoords', 'ycoords', 'xcoords_start', 'ycoords_start', 'scan_index', 'ground_truth_patches']
    for key in per_image_keys:
        if key in data:
            train_data[key] = data[key][:train_size]
            test_data[key] = data[key][train_size:train_size + test_size]
    
    # Copy arrays that are global (same for both splits)
    global_keys = ['objectGuess', 'probeGuess']
    for key in global_keys:
        if key in data:
            train_data[key] = data[key]
            test_data[key] = data[key]
    
    # Copy any remaining keys
    for key in data.keys():
        if key not in per_image_keys and key not in global_keys:
            if len(data[key].shape) > 0 and data[key].shape[0] == total_images:
                # Appears to be per-image
                train_data[key] = data[key][:train_size]
                test_data[key] = data[key][train_size:train_size + test_size]
            else:
                # Global array
                train_data[key] = data[key]
                test_data[key] = data[key]
    
    print(f"Saving train set to {train_output}: {train_data['diff3d'].shape[0]} images")
    np.savez_compressed(train_output, **train_data)
    
    print(f"Saving test set to {test_output}: {test_data['diff3d'].shape[0]} images")
    np.savez_compressed(test_output, **test_data)
    
    print("Split complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset for probe study")
    parser.add_argument("input_file", help="Input NPZ file")
    parser.add_argument("--train_size", type=int, required=True, help="Number of train images")
    parser.add_argument("--test_size", type=int, required=True, help="Number of test images")
    parser.add_argument("--train_output", required=True, help="Output train file")
    parser.add_argument("--test_output", required=True, help="Output test file")
    
    args = parser.parse_args()
    
    split_dataset(args.input_file, args.train_size, args.test_size, 
                  args.train_output, args.test_output)