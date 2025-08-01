#!/usr/bin/env python3
"""Debug the exact convention differences between implementations."""

import numpy as np
import tensorflow as tf
import os

os.environ['USE_XLA_TRANSLATE'] = '0'

from ptycho.tf_helper import translate_core
from ptycho.projective_warp_xla import projective_warp_xla

def test_backward_vs_forward_mapping():
    """Test if the issue is backward vs forward mapping."""
    print("=== Testing Mapping Direction ===")
    
    # Create test image
    size = 8
    image = np.zeros((1, size, size, 1), dtype=np.float32)
    # Put a 2x2 bright square in the middle
    image[0, 3:5, 3:5, 0] = 1.0
    
    print("Original image (8x8 with 2x2 bright square at center):")
    print(image[0, :, :, 0])
    
    # Test translation: move right by 2, down by 1
    dx, dy = 2.0, 1.0
    translations = tf.constant([[dx, dy]], dtype=tf.float32)
    
    # For the original implementation, it negates the translation
    # So the homography matrix will have [-dx, -dy]
    
    # Create homography matrix for XLA (with negation like original)
    M = tf.constant([[[1.0, 0.0, -dx],
                      [0.0, 1.0, -dy],
                      [0.0, 0.0, 1.0]]], dtype=tf.float32)
    
    # Run XLA implementation
    result_xla = projective_warp_xla(tf.constant(image), M, interpolation='nearest')
    
    print(f"\nXLA result (translation by dx={dx}, dy={dy}):")
    print(result_xla[0, :, :, 0].numpy())
    
    # The bright square should move from (3:5, 3:5) to (5:7, 4:6)
    # if positive translation moves content in positive direction

def analyze_bilinear_differences():
    """Analyze why bilinear interpolation differs."""
    print("\n=== Analyzing Bilinear Interpolation ===")
    
    # Create a gradient image for better understanding
    size = 8
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    image = xx[np.newaxis, :, :, np.newaxis].astype(np.float32)
    
    print("Test image (horizontal gradient 0 to 1):")
    print(image[0, :, :, 0])
    
    # Small fractional translation
    dx, dy = 0.5, 0.0
    translations = tf.constant([[dx, dy]], dtype=tf.float32)
    
    # Original implementation
    result_orig = translate_core(tf.constant(image), translations, interpolation='bilinear')
    
    # XLA implementation with homography
    M = tf.constant([[[1.0, 0.0, -dx],
                      [0.0, 1.0, -dy],
                      [0.0, 0.0, 1.0]]], dtype=tf.float32)
    result_xla = projective_warp_xla(tf.constant(image), M, interpolation='bilinear')
    
    print(f"\nTranslation by dx={dx}, dy={dy}")
    print("Original result (row 0):", result_orig[0, 0, :, 0].numpy())
    print("XLA result (row 0):", result_xla[0, 0, :, 0].numpy())
    print("Difference:", (result_orig - result_xla)[0, 0, :, 0].numpy())

def test_edge_handling():
    """Test how edges are handled."""
    print("\n=== Testing Edge Handling ===")
    
    # Create image with values at edges
    size = 8
    image = np.zeros((1, size, size, 1), dtype=np.float32)
    image[0, :, 0, 0] = 1.0  # Left edge
    image[0, 0, :, 0] = 2.0  # Top edge
    
    print("Test image with edge values:")
    print(image[0, :, :, 0])
    
    # Translate to show edge behavior
    dx, dy = -1.0, -1.0  # Move content left and up
    translations = tf.constant([[dx, dy]], dtype=tf.float32)
    
    # Original
    result_orig = translate_core(tf.constant(image), translations, interpolation='bilinear')
    
    # XLA
    M = tf.constant([[[1.0, 0.0, -dx],
                      [0.0, 1.0, -dy],
                      [0.0, 0.0, 1.0]]], dtype=tf.float32)
    result_xla = projective_warp_xla(tf.constant(image), M, interpolation='bilinear', fill_mode='edge')
    
    print("\nOriginal result:")
    print(result_orig[0, :, :, 0].numpy())
    print("\nXLA result:")
    print(result_xla[0, :, :, 0].numpy())

def check_tfa_convention():
    """Verify what the TFA convention actually does."""
    print("\n=== Checking TFA Parameter Convention ===")
    
    # According to TFA docs, the transform parameters represent:
    # Output pixel (x', y') = Transform * Input pixel (x, y)
    # Where Transform is:
    # [[a0, a1, a2],
    #  [a3, a4, a5],
    #  [a6, a7, 1]]
    
    # For translation only: a0=1, a1=0, a3=0, a4=1, a6=0, a7=0
    # So: x' = x + a2, y' = y + a5
    
    # In ImageProjectiveTransformV3, the transformation is applied as:
    # (x_out, y_out) maps to (x_in, y_in) via the inverse transform
    # So positive a2, a5 move the image content in the negative direction
    
    print("TFA convention:")
    print("- Transform parameters define OUTPUT = Transform * INPUT")
    print("- ImageProjectiveTransformV3 uses inverse mapping")
    print("- So positive translation parameters move content in negative direction")
    print("- That's why the original code negates dx, dy!")

def main():
    """Run debugging tests."""
    print("Debugging XLA Convention Differences")
    print("=" * 60)
    
    test_backward_vs_forward_mapping()
    analyze_bilinear_differences()
    test_edge_handling()
    check_tfa_convention()

if __name__ == "__main__":
    main()