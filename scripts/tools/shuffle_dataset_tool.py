#!/usr/bin/env python3
"""
Shuffle Dataset Tool

This tool randomizes the order of per-scan arrays in NPZ files while preserving
global arrays and maintaining data relationships between shuffled arrays.

NOTE: As of the unified sampling update, manual shuffling is no longer required
for gridsize=1 training. The data pipeline now handles random sampling internally
for all gridsize values. This tool remains useful for:
- Creating canonical, reproducible benchmark datasets
- Data curation and organization tasks
- Ensuring backwards compatibility with existing workflows

Usage:
    python shuffle_dataset_tool.py --input-file input.npz --output-file output.npz
    python shuffle_dataset_tool.py --input-file input.npz --output-file output.npz --seed 42 --dry-run

Author: PtychoPINN Project
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np


def setup_logging():
    """Configure logging with informative format."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Shuffle per-scan arrays in NPZ datasets while preserving global arrays',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-file data.npz --output-file shuffled_data.npz
  %(prog)s --input-file data.npz --output-file shuffled_data.npz --seed 42
  %(prog)s --input-file data.npz --output-file shuffled_data.npz --dry-run
        """
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        help='Path to input NPZ file'
    )
    
    parser.add_argument(
        '--output-file', 
        type=str,
        help='Path to output NPZ file'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible shuffling (optional)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be shuffled without writing output file'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run validation tests on synthetic data'
    )
    
    return parser.parse_args()


def validate_input_file(input_file):
    """Validate that input file exists and is readable."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    if not os.access(input_file, os.R_OK):
        raise PermissionError(f"Input file is not readable: {input_file}")
    
    try:
        # Test that file can be loaded as NPZ
        with np.load(input_file, allow_pickle=False) as data:
            if len(data.files) == 0:
                raise ValueError(f"Input NPZ file is empty: {input_file}")
    except Exception as e:
        raise ValueError(f"Input file is not a valid NPZ file: {input_file}. Error: {e}")


def validate_output_path(output_file):
    """Validate that output directory is writable."""
    output_dir = os.path.dirname(os.path.abspath(output_file))
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {output_dir}")


def load_and_analyze_npz(input_file):
    """
    Load NPZ file and analyze array types.
    
    Returns:
        tuple: (data_dict, per_scan_arrays, global_arrays, n_images)
    """
    logging.info("Loading and analyzing NPZ file...")
    
    try:
        # Load NPZ data
        data = np.load(input_file, allow_pickle=False)
        data_dict = {key: data[key] for key in data.files}
    except Exception as e:
        raise ValueError(f"Failed to load NPZ file: {e}")
    
    if len(data_dict) == 0:
        raise ValueError("NPZ file contains no arrays")
    
    # Handle edge case: all arrays are scalar or 0-dimensional
    valid_arrays = {key: arr for key, arr in data_dict.items() if hasattr(arr, 'shape') and len(arr.shape) > 0}
    
    if not valid_arrays:
        raise ValueError("NPZ file contains only scalar arrays - no arrays suitable for shuffling")
    
    # Analyze array shapes to determine n_images
    array_shapes = {key: arr.shape for key, arr in valid_arrays.items()}
    
    # Find the most common first dimension size (likely n_images)
    first_dims = [shape[0] for shape in array_shapes.values()]
    
    if not first_dims:
        raise ValueError("No arrays with valid dimensions found in NPZ file")
    
    # Use the most common first dimension as n_images
    from collections import Counter
    first_dim_counts = Counter(first_dims)
    n_images = first_dim_counts.most_common(1)[0][0]
    
    # Handle edge case: very small datasets
    if n_images <= 1:
        raise ValueError(f"Dataset too small for shuffling: only {n_images} data points found")
    
    logging.info(f"Detected n_images = {n_images}")
    
    # Classify arrays as per-scan or global
    per_scan_arrays = []
    global_arrays = []
    
    for key, arr in data_dict.items():
        # Skip scalar arrays
        if not hasattr(arr, 'shape') or len(arr.shape) == 0:
            global_arrays.append(key)
            logging.info(f"Scalar array: {key} (treated as global)")
            continue
            
        if arr.shape[0] == n_images:
            per_scan_arrays.append(key)
            logging.info(f"Per-scan array: {key} {arr.shape} {arr.dtype}")
        else:
            global_arrays.append(key)
            logging.info(f"Global array: {key} {arr.shape} {arr.dtype}")
    
    # Validate that we have at least one per-scan array
    if not per_scan_arrays:
        raise ValueError("No per-scan arrays found (no arrays with first dimension matching n_images)")
    
    # Validate that all per-scan arrays have consistent first dimension
    for key in per_scan_arrays:
        if data_dict[key].shape[0] != n_images:
            raise ValueError(f"Inconsistent per-scan array size: {key} has {data_dict[key].shape[0]} elements, expected {n_images}")
    
    # Additional validation: check for extremely large datasets
    if n_images > 1000000:  # 1 million limit
        logging.warning(f"Very large dataset detected ({n_images} images). This may consume significant memory.")
    
    logging.info(f"Found {len(per_scan_arrays)} per-scan arrays and {len(global_arrays)} global arrays")
    
    return data_dict, per_scan_arrays, global_arrays, n_images


def shuffle_arrays(data_dict, per_scan_arrays, global_arrays, n_images, seed=None):
    """
    Shuffle per-scan arrays while preserving global arrays.
    
    Args:
        data_dict: Dictionary of all arrays
        per_scan_arrays: List of per-scan array keys
        global_arrays: List of global array keys  
        n_images: Number of images/scans
        seed: Random seed for reproducible shuffling
        
    Returns:
        tuple: (shuffled_data_dict, permutation_indices)
    """
    logging.info("Generating shuffling permutation...")
    
    # Validate inputs
    if n_images <= 0:
        raise ValueError(f"Invalid n_images: {n_images}")
    
    if not per_scan_arrays:
        raise ValueError("No per-scan arrays to shuffle")
    
    # Verify all per-scan arrays exist and have correct size
    for key in per_scan_arrays:
        if key not in data_dict:
            raise ValueError(f"Per-scan array '{key}' not found in data")
        
        arr = data_dict[key]
        if not hasattr(arr, 'shape') or len(arr.shape) == 0:
            raise ValueError(f"Per-scan array '{key}' has no dimensions")
            
        if arr.shape[0] != n_images:
            raise ValueError(f"Per-scan array '{key}' has size {arr.shape[0]}, expected {n_images}")
    
    # Set random seed if provided
    if seed is not None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Seed must be a non-negative integer, got: {seed}")
        np.random.seed(seed)
        logging.info(f"Using random seed: {seed}")
    
    # Generate random permutation
    try:
        permutation_indices = np.random.permutation(n_images)
    except Exception as e:
        raise RuntimeError(f"Failed to generate permutation: {e}")
    
    logging.info(f"Generated permutation for {n_images} indices")
    logging.info(f"First 10 original indices: {list(range(min(10, n_images)))}")
    logging.info(f"First 10 shuffled indices: {list(permutation_indices[:10])}")
    
    # Show sample mapping for verification
    sample_size = min(5, n_images)
    logging.info("Sample index mapping:")
    for i in range(sample_size):
        original_idx = i
        new_idx = permutation_indices[i]
        logging.info(f"  Position {i}: original index {original_idx} -> shuffled index {new_idx}")
    
    # Create shuffled data dictionary
    shuffled_data_dict = {}
    
    # Shuffle per-scan arrays using the same permutation
    for key in per_scan_arrays:
        try:
            original_array = data_dict[key]
            shuffled_array = original_array[permutation_indices]
            shuffled_data_dict[key] = shuffled_array
            logging.info(f"Shuffled per-scan array: {key}")
        except Exception as e:
            raise RuntimeError(f"Failed to shuffle array '{key}': {e}")
    
    # Copy global arrays unchanged
    for key in global_arrays:
        try:
            shuffled_data_dict[key] = data_dict[key]
            logging.info(f"Preserved global array: {key}")
        except Exception as e:
            raise RuntimeError(f"Failed to copy global array '{key}': {e}")
    
    # Add shuffle metadata
    shuffled_data_dict['_shuffle_applied'] = np.array([True])
    if seed is not None:
        shuffled_data_dict['_shuffle_seed'] = np.array([seed])
    
    logging.info("Array shuffling completed")
    
    return shuffled_data_dict, permutation_indices


def verify_shuffling_results(original_data, shuffled_data, per_scan_arrays, global_arrays, permutation_indices):
    """
    Verify and report on shuffling results for dry-run mode.
    
    Args:
        original_data: Original data dictionary
        shuffled_data: Shuffled data dictionary
        per_scan_arrays: List of per-scan array keys
        global_arrays: List of global array keys
        permutation_indices: The permutation used for shuffling
    """
    logging.info("Verification Results:")
    
    # Verify per-scan arrays are shuffled correctly
    for key in per_scan_arrays:
        orig_arr = original_data[key]
        shuf_arr = shuffled_data[key]
        
        # Check a few sample indices to confirm correct mapping
        sample_indices = [0, min(1, len(permutation_indices)-1), min(5, len(permutation_indices)-1)]
        
        mapping_correct = True
        for pos in sample_indices:
            if pos < len(permutation_indices):
                expected_orig_idx = permutation_indices[pos]
                if not np.array_equal(orig_arr[expected_orig_idx], shuf_arr[pos]):
                    mapping_correct = False
                    break
        
        if mapping_correct:
            logging.info(f"  ✓ {key}: Correctly shuffled")
        else:
            logging.error(f"  ✗ {key}: Shuffling error detected!")
    
    # Verify global arrays unchanged
    for key in global_arrays:
        if np.array_equal(original_data[key], shuffled_data[key]):
            logging.info(f"  ✓ {key}: Correctly preserved (unchanged)")
        else:
            logging.error(f"  ✗ {key}: Incorrectly modified!")
    
    # Check that relationships are preserved
    if 'xcoords' in per_scan_arrays and 'diffraction' in per_scan_arrays:
        logging.info("Relationship verification (xcoords vs diffraction[i,0,0]):")
        sample_positions = [0, min(2, len(permutation_indices)-1)]
        
        for pos in sample_positions:
            if pos < len(permutation_indices):
                x_val = shuffled_data['xcoords'][pos]
                # Assuming our test data has diffraction[i,0,0] = xcoords[i]
                diff_val = shuffled_data['diffraction'][pos, 0, 0] if shuffled_data['diffraction'].ndim >= 3 else shuffled_data['diffraction'][pos]
                
                if abs(x_val - diff_val) < 1e-6:  # Allow for floating point precision
                    logging.info(f"  ✓ Position {pos}: xcoords={x_val:.1f}, diffraction[{pos},0,0]={diff_val:.1f} (relationship preserved)")
                else:
                    logging.info(f"  ℹ Position {pos}: xcoords={x_val:.1f}, diffraction[{pos},0,0]={diff_val:.1f} (different values, may be expected)")


def save_shuffled_data(shuffled_data_dict, output_file):
    """
    Save shuffled data to NPZ file.
    
    Args:
        shuffled_data_dict: Dictionary of shuffled arrays
        output_file: Path to output NPZ file
    """
    logging.info(f"Saving shuffled data to: {output_file}")
    
    try:
        # Save to NPZ file
        np.savez(output_file, **shuffled_data_dict)
        
        # Verify the file was written correctly
        try:
            with np.load(output_file, allow_pickle=False) as verification_data:
                saved_keys = set(verification_data.files)
                expected_keys = set(shuffled_data_dict.keys())
                
                if saved_keys != expected_keys:
                    raise ValueError(f"Verification failed: saved keys {saved_keys} != expected keys {expected_keys}")
                
                logging.info(f"Successfully saved {len(saved_keys)} arrays to output file")
                
        except Exception as e:
            raise ValueError(f"Output file verification failed: {e}")
            
    except Exception as e:
        raise IOError(f"Failed to save output file: {e}")


def create_synthetic_test_data(n_images=10, image_size=64):
    """
    Create synthetic NPZ data for testing purposes.
    
    Args:
        n_images: Number of per-scan images to create
        image_size: Size of diffraction patterns (image_size x image_size)
        
    Returns:
        dict: Synthetic data dictionary ready for NPZ saving
    """
    # Create per-scan arrays with predictable patterns
    diffraction = np.zeros((n_images, image_size, image_size), dtype=np.float32)
    xcoords = np.arange(n_images, dtype=np.float32)
    ycoords = np.arange(n_images, dtype=np.float32) + 100  # Offset for distinction
    
    # Make diffraction patterns identifiable by their index
    for i in range(n_images):
        diffraction[i, 0, 0] = i  # Put index value in corner
        diffraction[i, 1, 1] = i * 2  # Put 2*index in another corner
    
    # Create global arrays (same size regardless of n_images)
    object_size = image_size * 3
    objectGuess = np.random.rand(object_size, object_size).astype(np.complex64)
    probeGuess = np.random.rand(image_size, image_size).astype(np.complex64)
    
    synthetic_data = {
        'diffraction': diffraction,
        'xcoords': xcoords,
        'ycoords': ycoords,
        'objectGuess': objectGuess,
        'probeGuess': probeGuess
    }
    
    return synthetic_data


def run_validation_tests():
    """
    Run validation tests on the shuffling tool.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    logging.info("Running validation tests...")
    
    # Test 1: Basic shuffling test
    logging.info("Test 1: Basic shuffling functionality")
    try:
        test_data = create_synthetic_test_data(n_images=10)
        
        # Run shuffling logic directly
        per_scan_arrays = ['diffraction', 'xcoords', 'ycoords']
        global_arrays = ['objectGuess', 'probeGuess']
        n_images = 10
        
        shuffled_data, permutation = shuffle_arrays(
            test_data, per_scan_arrays, global_arrays, n_images, seed=42
        )
        
        # Verify per-scan arrays are shuffled
        if np.array_equal(test_data['xcoords'], shuffled_data['xcoords']):
            logging.error("Test 1 FAILED: xcoords not shuffled")
            return False
            
        # Verify global arrays are unchanged
        if not np.array_equal(test_data['objectGuess'], shuffled_data['objectGuess']):
            logging.error("Test 1 FAILED: objectGuess was modified")
            return False
            
        # Verify all indices appear exactly once
        sorted_permutation = np.sort(permutation)
        expected_indices = np.arange(n_images)
        if not np.array_equal(sorted_permutation, expected_indices):
            logging.error("Test 1 FAILED: Not all indices appear exactly once")
            return False
            
        logging.info("Test 1 PASSED: Basic shuffling works correctly")
        
    except Exception as e:
        logging.error(f"Test 1 FAILED with exception: {e}")
        return False
    
    # Test 2: Relationship preservation test
    logging.info("Test 2: Data relationship preservation")
    try:
        test_data = create_synthetic_test_data(n_images=10)
        
        # Verify relationships are preserved after shuffling
        for i in range(10):
            original_x = test_data['xcoords'][i]
            original_diff_val = test_data['diffraction'][i, 0, 0]
            
            shuffled_x = shuffled_data['xcoords'][i]
            shuffled_diff_val = shuffled_data['diffraction'][i, 0, 0]
            
            # The relationship should be preserved: diff[i,0,0] should equal xcoords[i]
            if shuffled_diff_val != shuffled_x:
                logging.error(f"Test 2 FAILED: Relationship broken at index {i}")
                return False
                
        logging.info("Test 2 PASSED: Data relationships preserved")
        
    except Exception as e:
        logging.error(f"Test 2 FAILED with exception: {e}")
        return False
    
    # Test 3: Reproducibility test
    logging.info("Test 3: Reproducibility with same seed")
    try:
        test_data = create_synthetic_test_data(n_images=10)
        per_scan_arrays = ['diffraction', 'xcoords', 'ycoords']
        global_arrays = ['objectGuess', 'probeGuess']
        n_images = 10
        
        # Run shuffling twice with same seed
        shuffled_data_1, perm_1 = shuffle_arrays(
            test_data, per_scan_arrays, global_arrays, n_images, seed=123
        )
        shuffled_data_2, perm_2 = shuffle_arrays(
            test_data, per_scan_arrays, global_arrays, n_images, seed=123
        )
        
        # Results should be identical
        if not np.array_equal(perm_1, perm_2):
            logging.error("Test 3 FAILED: Different permutations with same seed")
            return False
            
        if not np.array_equal(shuffled_data_1['xcoords'], shuffled_data_2['xcoords']):
            logging.error("Test 3 FAILED: Different shuffled results with same seed")
            return False
            
        # Run with different seed - should get different result
        shuffled_data_3, perm_3 = shuffle_arrays(
            test_data, per_scan_arrays, global_arrays, n_images, seed=456
        )
        
        if np.array_equal(perm_1, perm_3):
            logging.error("Test 3 FAILED: Same permutation with different seeds")
            return False
            
        logging.info("Test 3 PASSED: Reproducibility works correctly")
        
    except Exception as e:
        logging.error(f"Test 3 FAILED with exception: {e}")
        return False
    
    logging.info("All validation tests PASSED!")
    return True


def main():
    """Main entry point."""
    setup_logging()
    
    try:
        args = parse_arguments()
        
        # Handle test mode
        if args.test:
            logging.info("Running validation tests...")
            success = run_validation_tests()
            if success:
                logging.info("All tests passed!")
                sys.exit(0)
            else:
                logging.error("Some tests failed!")
                sys.exit(1)
        
        # Normal operation mode - validate required arguments
        if not args.input_file or not args.output_file:
            logging.error("--input-file and --output-file are required (unless using --test)")
            sys.exit(1)
            
        # Validate input and output paths
        validate_input_file(args.input_file)
        if not args.dry_run:
            validate_output_path(args.output_file)
        
        logging.info(f"Input file: {args.input_file}")
        logging.info(f"Output file: {args.output_file}")
        if args.seed is not None:
            logging.info(f"Random seed: {args.seed}")
        if args.dry_run:
            logging.info("DRY RUN MODE - No output file will be written")
        
        # Load and analyze NPZ file
        data_dict, per_scan_arrays, global_arrays, n_images = load_and_analyze_npz(args.input_file)
        
        # Perform shuffling
        shuffled_data_dict, permutation_indices = shuffle_arrays(
            data_dict, per_scan_arrays, global_arrays, n_images, args.seed
        )
        
        # In dry-run mode, run verification
        if args.dry_run:
            verify_shuffling_results(data_dict, shuffled_data_dict, per_scan_arrays, global_arrays, permutation_indices)
            logging.info("DRY RUN - No output file written")
        else:
            # Save output in normal mode
            save_shuffled_data(shuffled_data_dict, args.output_file)
            logging.info(f"Shuffled dataset saved to: {args.output_file}")
        
        logging.info("Shuffle dataset tool completed successfully")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()