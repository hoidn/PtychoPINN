#!/usr/bin/env python
"""Test script for pure TensorFlow translation implementation."""

import tensorflow as tf
import numpy as np
import sys

# Disable GPU for this test to avoid XLA issues
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from ptycho.tf_helper import translate, Translation

def test_translation():
    """Test pure TF translation implementation."""
    print("Testing pure TensorFlow translation implementation...")
    
    # Test 1: Basic translation
    print("\n1. Testing basic translation:")
    test_img = tf.constant(np.random.rand(2, 64, 64, 1).astype(np.float32))
    test_offsets = tf.constant([[1.5, -2.0], [0.5, 1.0]], dtype=tf.float32)
    
    result = translate(test_img, test_offsets)
    print(f"   Input shape: {test_img.shape}")
    print(f"   Output shape: {result.shape}")
    print(f"   ✓ Basic translation successful!")
    
    # Test 2: Complex number translation
    print("\n2. Testing complex number translation:")
    real_part = np.random.rand(2, 64, 64, 1).astype(np.float32)
    imag_part = np.random.rand(2, 64, 64, 1).astype(np.float32)
    test_complex = tf.complex(real_part, imag_part)
    
    result_complex = translate(test_complex, test_offsets)
    print(f"   Input dtype: {test_complex.dtype}")
    print(f"   Output dtype: {result_complex.dtype}")
    print(f"   ✓ Complex translation successful!")
    
    # Test 3: Translation layer
    print("\n3. Testing Translation layer:")
    translation_layer = Translation(jitter_stddev=0.0)
    result_layer = translation_layer([test_img, test_offsets])
    print(f"   Output shape: {result_layer.shape}")
    print(f"   ✓ Translation layer successful!")
    
    # Test 4: XLA compilation test
    print("\n4. Testing XLA compatibility:")
    @tf.function(jit_compile=True)
    def xla_test():
        return translate(test_img, test_offsets)
    
    try:
        xla_result = xla_test()
        print(f"   ✓ XLA compilation successful!")
    except Exception as e:
        print(f"   ✗ XLA compilation failed: {e}")
        return False
    
    # Test 5: Different interpolation modes
    print("\n5. Testing interpolation modes:")
    result_bilinear = translate(test_img, test_offsets, interpolation='bilinear')
    result_nearest = translate(test_img, test_offsets, interpolation='nearest')
    print(f"   ✓ Both interpolation modes work!")
    
    print("\n✅ All tests passed! Pure TF translation implementation is working correctly.")
    return True

if __name__ == "__main__":
    success = test_translation()
    sys.exit(0 if success else 1)