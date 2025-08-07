#!/usr/bin/env python3
"""Quick debug test for patch extraction equivalence."""

import numpy as np
import tensorflow as tf
import logging
from ptycho.raw_data import _get_image_patches_iterative, _get_image_patches_batched

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_debug():
    """Debug the numerical differences."""
    # Create very small, simple test case
    object_size = 32
    patch_size = 8
    batch_size = 2
    c = 1  # Simple case: gridsize=1
    
    # Create simple synthetic object
    x = np.linspace(-1, 1, object_size)
    y = np.linspace(-1, 1, object_size)
    X, Y = np.meshgrid(x, y)
    
    # Simple real-valued object for debugging
    ground_truth = np.exp(-(X**2 + Y**2) / 0.5).astype(np.complex64)
    
    # Pad the object
    pad_size = patch_size
    gt_padded = tf.constant(
        np.pad(ground_truth, ((pad_size, pad_size), (pad_size, pad_size)), 
               mode='constant', constant_values=0.0),
        dtype=tf.complex64
    )
    gt_padded = tf.expand_dims(tf.expand_dims(gt_padded, axis=0), axis=-1)
    
    # Create simple offsets
    offsets = np.array([
        [0.0, 0.0],  # Center
        [2.0, 3.0]   # Offset position
    ])
    
    # Reshape to expected format: (B*c, 1, 2, 1)
    offsets_f = offsets.reshape(batch_size * c, 1, 2, 1).astype(np.float32)
    offsets_f = tf.constant(offsets_f)
    
    print(f"gt_padded shape: {gt_padded.shape}")
    print(f"offsets_f shape: {offsets_f.shape}")
    print(f"batch_size: {batch_size}, c: {c}, patch_size: {patch_size}")
    
    # Run iterative implementation
    result_iterative = _get_image_patches_iterative(
        gt_padded, offsets_f, patch_size, batch_size, c
    )
    
    # Run batched implementation with small mini-batch size for debugging
    result_batched = _get_image_patches_batched(
        gt_padded, offsets_f, patch_size, batch_size, c, mini_batch_size=1
    )
    
    print(f"result_iterative shape: {result_iterative.shape}")
    print(f"result_batched shape: {result_batched.shape}")
    
    # Convert to numpy
    result_iterative_np = result_iterative.numpy()
    result_batched_np = result_batched.numpy()
    
    # Check difference
    diff = np.abs(result_iterative_np - result_batched_np)
    max_diff = np.max(diff)
    
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {np.mean(diff)}")
    
    # Show first patch comparison
    print("\nFirst patch comparison:")
    print("Iterative first few values:", result_iterative_np[0, 0, :3, 0])
    print("Batched first few values:", result_batched_np[0, 0, :3, 0])
    print("Difference first few values:", diff[0, 0, :3, 0])
    
    # Check if they're close enough
    is_close = np.allclose(result_iterative_np, result_batched_np, atol=1e-5)
    print(f"\nAre results close (atol=1e-5)? {is_close}")
    
    if not is_close:
        print("Results are not equivalent. Need to investigate further.")
        # Show some statistics
        print(f"Iterative result stats: min={np.min(result_iterative_np)}, max={np.max(result_iterative_np)}, std={np.std(result_iterative_np)}")
        print(f"Batched result stats: min={np.min(result_batched_np)}, max={np.max(result_batched_np)}, std={np.std(result_batched_np)}")
    else:
        print("✅ Results are equivalent!")

if __name__ == '__main__':
    test_debug()