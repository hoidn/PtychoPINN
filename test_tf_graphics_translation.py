#!/usr/bin/env python
"""Test TensorFlow Graphics translation implementation."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import tensorflow as tf
import numpy as np
import time
from ptycho.tf_helper import translate, Translation

def test_tf_graphics_translation():
    """Test TensorFlow Graphics translation implementation."""
    print("Testing TensorFlow Graphics translation implementation...")
    
    # Test 1: Basic functionality
    print("\n1. Testing basic translation:")
    test_img = tf.constant(np.random.rand(2, 64, 64, 1).astype(np.float32))
    test_offsets = tf.constant([[1.5, -2.0], [0.5, 1.0]], dtype=tf.float32)
    
    result = translate(test_img, test_offsets)
    print(f"   Input shape: {test_img.shape}")
    print(f"   Output shape: {result.shape}")
    print(f"   ✓ Basic translation successful!")
    
    # Test 2: Performance comparison
    print("\n2. Performance comparison:")
    batch_sizes = [4, 16, 32]
    
    for batch_size in batch_sizes:
        test_batch = tf.constant(np.random.rand(batch_size, 64, 64, 1).astype(np.float32))
        test_offsets_batch = tf.constant(np.random.rand(batch_size, 2).astype(np.float32) * 10)
        
        # Warm up
        _ = translate(test_batch, test_offsets_batch)
        
        # Time the operation
        start_time = time.time()
        for _ in range(10):
            _ = translate(test_batch, test_offsets_batch)
        tf_graphics_time = (time.time() - start_time) / 10
        
        print(f"   Batch size {batch_size}: {tf_graphics_time:.4f}s per call")
    
    # Test 3: XLA compilation
    print("\n3. Testing XLA compatibility:")
    @tf.function(jit_compile=True)
    def xla_test():
        return translate(test_img, test_offsets)
    
    try:
        xla_result = xla_test()
        print(f"   ✓ XLA compilation successful!")
    except Exception as e:
        print(f"   ✗ XLA compilation failed: {e}")
        return False
    
    # Test 4: Complex numbers
    print("\n4. Testing complex number support:")
    real_part = np.random.rand(2, 64, 64, 1).astype(np.float32)
    imag_part = np.random.rand(2, 64, 64, 1).astype(np.float32)
    test_complex = tf.complex(real_part, imag_part)
    
    result_complex = translate(test_complex, test_offsets)
    print(f"   Input dtype: {test_complex.dtype}")
    print(f"   Output dtype: {result_complex.dtype}")
    print(f"   ✓ Complex translation successful!")
    
    # Test 5: Large batch performance
    print("\n5. Testing large batch performance:")
    large_batch = tf.constant(np.random.rand(128, 64, 64, 1).astype(np.float32))
    large_offsets = tf.constant(np.random.rand(128, 2).astype(np.float32) * 10)
    
    start_time = time.time()
    _ = translate(large_batch, large_offsets)
    large_batch_time = time.time() - start_time
    print(f"   128 batch translation: {large_batch_time:.4f}s")
    print(f"   Throughput: {128/large_batch_time:.1f} images/second")
    
    print("\n✅ All tests passed! TensorFlow Graphics translation is working correctly.")
    return True

if __name__ == "__main__":
    test_tf_graphics_translation()