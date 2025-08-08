#!/usr/bin/env python3
"""Test that mini-batched implementation matches the original batched implementation."""

import numpy as np
import tensorflow as tf
import logging

# Import the original code to get the ORIGINAL batched version
# We'll temporarily implement it here for testing
def _get_image_patches_batched_original(gt_padded, offsets_f, N, B, c):
    """Original batched implementation (memory-hungry but working)."""
    import ptycho.tf_helper as hh
    
    # Create a batched version of the padded image by repeating it B*c times
    gt_padded_batch = tf.repeat(gt_padded, B * c, axis=0)
    
    # Extract offsets
    negated_offsets = -offsets_f[:, 0, :, 0]  # Shape: (B*c, 2)
    
    # Perform a single batched translation
    translated_patches = hh.translate(gt_padded_batch, negated_offsets)
    
    # Slice to get only the central N×N region of each patch
    patches_flat = translated_patches[:, :N, :N, :]  # Shape: (B*c, N, N, 1)
    
    # Reshape from flat format to channel format
    patches_squeezed = tf.squeeze(patches_flat, axis=-1)
    patches_grouped = tf.reshape(patches_squeezed, (B, c, N, N))
    patches_channel = tf.transpose(patches_grouped, [0, 2, 3, 1])
    
    return patches_channel

# Import the mini-batched version
from ptycho.raw_data import _get_image_patches_batched

def test_minibatch_equivalence():
    """Test mini-batched against original batched implementation."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Small test case
    object_size = 32
    patch_size = 8
    batch_size = 4
    c = 2  # gridsize=sqrt(2) for testing
    
    # Create synthetic object
    x = np.linspace(-1, 1, object_size)
    y = np.linspace(-1, 1, object_size)
    X, Y = np.meshgrid(x, y)
    ground_truth = np.exp(-(X**2 + Y**2) / 0.5).astype(np.complex64)
    
    # Pad the object
    pad_size = patch_size
    gt_padded = tf.constant(
        np.pad(ground_truth, ((pad_size, pad_size), (pad_size, pad_size)), 
               mode='constant', constant_values=0.0),
        dtype=tf.complex64
    )
    gt_padded = tf.expand_dims(tf.expand_dims(gt_padded, axis=0), axis=-1)
    
    # Create random offsets
    max_offset = object_size - patch_size
    x_offsets = np.random.uniform(-max_offset/2, max_offset/2, batch_size * c)
    y_offsets = np.random.uniform(-max_offset/2, max_offset/2, batch_size * c)
    offsets = np.stack([y_offsets, x_offsets], axis=1)  # Note: [y, x] order
    offsets_f = offsets.reshape(batch_size * c, 1, 2, 1).astype(np.float32)
    offsets_f = tf.constant(offsets_f)
    
    logger.info(f"Testing with B={batch_size}, c={c}, N={patch_size}")
    
    # Run original batched implementation
    result_original = _get_image_patches_batched_original(
        gt_padded, offsets_f, patch_size, batch_size, c
    )
    
    # Run mini-batched implementation with small batch size
    result_minibatch = _get_image_patches_batched(
        gt_padded, offsets_f, patch_size, batch_size, c, mini_batch_size=2
    )
    
    # Compare results
    result_original_np = result_original.numpy()
    result_minibatch_np = result_minibatch.numpy()
    
    max_diff = np.max(np.abs(result_original_np - result_minibatch_np))
    mean_diff = np.mean(np.abs(result_original_np - result_minibatch_np))
    
    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")
    
    # Check if they're equivalent (should be exactly the same)
    is_close = np.allclose(result_original_np, result_minibatch_np, atol=1e-6)
    
    if is_close:
        logger.info("✅ Mini-batched and original batched implementations are equivalent!")
    else:
        logger.error("❌ Implementations differ! Debugging needed.")
        # Show where differences occur
        diff_mask = np.abs(result_original_np - result_minibatch_np) > 1e-6
        logger.info(f"Differences found at {np.sum(diff_mask)} positions")
    
    return is_close

if __name__ == '__main__':
    success = test_minibatch_equivalence()
    exit(0 if success else 1)