#!/usr/bin/env python3
"""Debug translation differences between original and XLA implementations."""

import numpy as np
import tensorflow as tf
import os

# Disable XLA for comparison
os.environ['USE_XLA_TRANSLATE'] = '0'

from ptycho.tf_helper import translate_core
from ptycho.projective_warp_xla import translate_xla, projective_warp_xla, tfa_params_to_3x3

def create_test_image(size=32):
    """Create a simple test image with a clear pattern."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Create a simple checkerboard pattern
    pattern = ((xx * 8).astype(int) + (yy * 8).astype(int)) % 2
    return pattern.astype(np.float32)

def test_translation_matrix_equivalence():
    """Test if the translation matrix construction is equivalent."""
    print("=== Testing Translation Matrix Construction ===")
    
    # Test translation
    dx, dy = 5.0, -3.0
    translations = tf.constant([[dx, dy]], dtype=tf.float32)
    
    # Original implementation matrix (TFA format)
    dx_neg = -translations[:, 0]  # -5.0
    dy_neg = -translations[:, 1]  # 3.0
    
    # TFA 8-parameter format: [a0, a1, a2, a3, a4, a5, a6, a7]
    # This represents the transformation matrix:
    # [[a0, a1, a2],
    #  [a3, a4, a5],
    #  [a6, a7, 1]]
    tfa_params = tf.stack([
        tf.ones([1], dtype=tf.float32),    # a0 = 1
        tf.zeros([1], dtype=tf.float32),   # a1 = 0
        dx_neg,                            # a2 = -dx
        tf.zeros([1], dtype=tf.float32),   # a3 = 0
        tf.ones([1], dtype=tf.float32),    # a4 = 1
        dy_neg,                            # a5 = -dy
        tf.zeros([1], dtype=tf.float32),   # a6 = 0
        tf.zeros([1], dtype=tf.float32)    # a7 = 0
    ], axis=1)
    
    print(f"Original TFA params: {tfa_params.numpy()}")
    
    # XLA implementation matrix
    dx_xla = -translations[:, 0]
    dy_xla = -translations[:, 1]
    
    # Build homography matrix
    ones = tf.ones([1], dtype=tf.float32)
    zeros = tf.zeros([1], dtype=tf.float32)
    row0 = tf.stack([ones, zeros, dx_xla], axis=-1)
    row1 = tf.stack([zeros, ones, dy_xla], axis=-1)
    row2 = tf.stack([zeros, zeros, ones], axis=-1)
    M_xla = tf.stack([row0, row1, row2], axis=-2)
    
    print(f"XLA homography matrix:\n{M_xla.numpy()}")
    
    # Convert TFA params to 3x3 for comparison
    M_tfa = tfa_params_to_3x3(tfa_params)
    print(f"TFA converted to 3x3:\n{M_tfa.numpy()}")
    
    # Check if they're the same
    diff = tf.reduce_max(tf.abs(M_xla - M_tfa))
    print(f"Matrix difference: {diff.numpy()}")
    
    return M_xla, M_tfa

def test_simple_translation():
    """Test with a simple image and known translation."""
    print("\n=== Testing Simple Translation ===")
    
    # Create a simple test image
    size = 32
    image = create_test_image(size)
    image = image[np.newaxis, :, :, np.newaxis]  # Add batch and channel dims
    images = tf.constant(image, dtype=tf.float32)
    
    # Test with integer translation
    translations = tf.constant([[5.0, 3.0]], dtype=tf.float32)
    
    print(f"Input image shape: {images.shape}")
    print(f"Translation: {translations.numpy()}")
    
    # Original implementation
    result_orig = translate_core(images, translations, interpolation='bilinear', use_xla_workaround=False)
    
    # XLA implementation
    result_xla = translate_xla(images, translations, interpolation='bilinear', use_jit=False)
    
    # Compare results
    diff = tf.abs(result_orig - result_xla)
    max_diff = tf.reduce_max(diff)
    mean_diff = tf.reduce_mean(diff)
    
    print(f"Max difference: {max_diff.numpy():.6f}")
    print(f"Mean difference: {mean_diff.numpy():.6f}")
    
    # Show specific pixel values for debugging
    print("\nSample pixel values at (10, 10):")
    print(f"Original: {result_orig[0, 10, 10, 0].numpy():.6f}")
    print(f"XLA: {result_xla[0, 10, 10, 0].numpy():.6f}")
    
    # Save images for visual inspection
    np.save('debug_orig.npy', result_orig.numpy())
    np.save('debug_xla.npy', result_xla.numpy())
    np.save('debug_diff.npy', diff.numpy())
    
    return result_orig, result_xla

def test_coordinate_system():
    """Test the coordinate system conventions."""
    print("\n=== Testing Coordinate System ===")
    
    # Create an image with a single bright pixel
    size = 32
    image = np.zeros((1, size, size, 1), dtype=np.float32)
    image[0, 10, 15, 0] = 1.0  # Bright pixel at (y=10, x=15)
    images = tf.constant(image)
    
    # Translate by (dx=5, dy=3)
    translations = tf.constant([[5.0, 3.0]], dtype=tf.float32)
    
    print(f"Bright pixel at (y=10, x=15)")
    print(f"Translating by dx={translations[0, 0].numpy()}, dy={translations[0, 1].numpy()}")
    
    # Original implementation
    result_orig = translate_core(images, translations, interpolation='nearest', use_xla_workaround=False)
    
    # XLA implementation
    result_xla = translate_xla(images, translations, interpolation='nearest', use_jit=False)
    
    # Find where the bright pixel moved to
    orig_pos = tf.where(result_orig[0, :, :, 0] > 0.5)
    xla_pos = tf.where(result_xla[0, :, :, 0] > 0.5)
    
    print(f"\nOriginal implementation - bright pixel at: {orig_pos.numpy()}")
    print(f"XLA implementation - bright pixel at: {xla_pos.numpy()}")
    
    # Expected position after translation
    # If positive dx moves right and positive dy moves down:
    # New position should be (y=10+3=13, x=15+5=20)
    print(f"Expected position: (y=13, x=20)")

def test_homography_application():
    """Test how homographies are applied in both implementations."""
    print("\n=== Testing Homography Application ===")
    
    # Create a simple point
    point = tf.constant([[15.0, 10.0, 1.0]], dtype=tf.float32)  # (x, y, 1)
    
    # Translation
    dx, dy = 5.0, 3.0
    
    # Build homography matrix (same as XLA implementation)
    M = tf.constant([[1.0, 0.0, -dx],
                     [0.0, 1.0, -dy],
                     [0.0, 0.0, 1.0]], dtype=tf.float32)
    
    # Apply homography: dst = M @ src
    # This is the standard homography application
    transformed = tf.matmul(M, tf.transpose(point))
    transformed = tf.transpose(transformed)
    x_new = transformed[0, 0] / transformed[0, 2]
    y_new = transformed[0, 1] / transformed[0, 2]
    
    print(f"Original point: (x={point[0, 0].numpy()}, y={point[0, 1].numpy()})")
    print(f"Translation: (dx={dx}, dy={dy})")
    print(f"Homography matrix:\n{M.numpy()}")
    print(f"Transformed point: (x={x_new.numpy()}, y={y_new.numpy()})")
    print(f"Expected (if moving content): (x={15+dx}, y={10+dy}) = (20, 13)")
    print(f"Expected (if moving coordinates): (x={15-dx}, y={10-dy}) = (10, 7)")

def main():
    """Run all debugging tests."""
    print("Debugging Translation Differences")
    print("=" * 60)
    
    # Test matrix construction
    test_translation_matrix_equivalence()
    
    # Test simple translation
    test_simple_translation()
    
    # Test coordinate system
    test_coordinate_system()
    
    # Test homography application
    test_homography_application()
    
    print("\n" + "=" * 60)
    print("Debug files saved: debug_orig.npy, debug_xla.npy, debug_diff.npy")

if __name__ == "__main__":
    main()