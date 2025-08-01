#!/usr/bin/env python3
"""Test script for XLA-friendly translation implementation."""

import os
import numpy as np
import tensorflow as tf
import time

# Enable XLA translation
os.environ['USE_XLA_TRANSLATE'] = '1'
os.environ['USE_XLA_COMPILE'] = '1'

from ptycho.tf_helper import translate, Translation, should_use_xla
from ptycho.projective_warp_xla import translate_xla, projective_warp_xla_jit

def test_basic_translation():
    """Test basic translation functionality."""
    print("\n=== Testing Basic Translation ===")
    
    # Create test data
    batch_size = 4
    height = width = 128
    channels = 1
    
    # Create a simple test pattern
    images = tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
    translations = tf.constant([[10.0, -5.0], [0.0, 0.0], [-15.0, 20.0], [5.0, 5.0]], dtype=tf.float32)
    
    # Test XLA translation
    print(f"XLA enabled: {should_use_xla()}")
    result = translate(images, translations, use_xla_workaround=True)
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
    
    return result

def test_complex_translation():
    """Test translation with complex-valued tensors."""
    print("\n=== Testing Complex Translation ===")
    
    # Create complex test data
    batch_size = 2
    height = width = 64
    channels = 1
    
    real_part = tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
    imag_part = tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
    complex_images = tf.complex(real_part, imag_part)
    
    translations = tf.constant([[5.0, -3.0], [-7.0, 2.0]], dtype=tf.float32)
    
    # Test complex translation
    result = translate(complex_images, translations, use_xla_workaround=True)
    print(f"Complex result shape: {result.shape}")
    print(f"Complex result dtype: {result.dtype}")
    
    return result

def test_translation_layer():
    """Test Translation layer with XLA."""
    print("\n=== Testing Translation Layer ===")
    
    # Create test data
    batch_size = 4
    height = width = 128
    channels = 1
    
    images = tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
    offsets = tf.random.uniform((batch_size, 2), minval=-10, maxval=10, dtype=tf.float32)
    
    # Create layer with XLA enabled
    layer = Translation(jitter_stddev=2.0, use_xla=True)
    
    # Test the layer
    result = layer([images, offsets])
    print(f"Layer result shape: {result.shape}")
    print(f"Layer result dtype: {result.dtype}")
    
    return result

def benchmark_translation():
    """Benchmark XLA vs non-XLA translation."""
    print("\n=== Benchmarking Translation ===")
    
    # Create larger test data for benchmarking
    batch_size = 8
    height = width = 512
    channels = 3
    
    images = tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
    translations = tf.random.uniform((batch_size, 2), minval=-20, maxval=20, dtype=tf.float32)
    
    # Warm up
    _ = translate(images, translations, use_xla_workaround=False)
    _ = translate(images, translations, use_xla_workaround=True)
    
    # Benchmark non-XLA
    num_iterations = 10
    start_time = time.time()
    for _ in range(num_iterations):
        _ = translate(images, translations, use_xla_workaround=False)
    non_xla_time = (time.time() - start_time) / num_iterations
    print(f"Non-XLA average time: {non_xla_time:.4f}s")
    
    # Benchmark XLA
    start_time = time.time()
    for _ in range(num_iterations):
        _ = translate(images, translations, use_xla_workaround=True)
    xla_time = (time.time() - start_time) / num_iterations
    print(f"XLA average time: {xla_time:.4f}s")
    print(f"Speedup: {non_xla_time/xla_time:.2f}x")

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with zero translations
    images = tf.ones((2, 32, 32, 1), dtype=tf.float32)
    zero_translations = tf.zeros((2, 2), dtype=tf.float32)
    result = translate(images, zero_translations, use_xla_workaround=True)
    print(f"Zero translation - Max difference from original: {tf.reduce_max(tf.abs(result - images)):.6f}")
    
    # Test with large translations
    large_translations = tf.constant([[100.0, 100.0], [-100.0, -100.0]], dtype=tf.float32)
    result_large = translate(images, large_translations, use_xla_workaround=True)
    print(f"Large translation result shape: {result_large.shape}")
    
    # Test with single batch
    single_image = tf.random.uniform((1, 64, 64, 1), dtype=tf.float32)
    single_translation = tf.constant([[5.0, -5.0]], dtype=tf.float32)
    result_single = translate(single_image, single_translation, use_xla_workaround=True)
    print(f"Single batch result shape: {result_single.shape}")

def test_jit_compilation():
    """Test JIT compilation with XLA."""
    print("\n=== Testing JIT Compilation ===")
    
    # Create test function with JIT
    @tf.function(jit_compile=True)
    def translate_jit(images, translations):
        return translate_xla(images, translations, use_jit=True)
    
    # Test data
    images = tf.random.uniform((4, 128, 128, 1), dtype=tf.float32)
    translations = tf.random.uniform((4, 2), minval=-10, maxval=10, dtype=tf.float32)
    
    # First call will compile
    print("Compiling JIT function...")
    start_time = time.time()
    result = translate_jit(images, translations)
    compile_time = time.time() - start_time
    print(f"First call (with compilation): {compile_time:.4f}s")
    
    # Subsequent calls should be faster
    start_time = time.time()
    result = translate_jit(images, translations)
    runtime = time.time() - start_time
    print(f"Second call (compiled): {runtime:.4f}s")
    print(f"Speedup after compilation: {compile_time/runtime:.2f}x")

def main():
    """Run all tests."""
    print("Testing XLA Translation Implementation")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Run tests
    test_basic_translation()
    test_complex_translation()
    test_translation_layer()
    test_edge_cases()
    test_jit_compilation()
    
    # Only run benchmark if not in CI/quick test mode
    if os.environ.get('QUICK_TEST', '').lower() != 'true':
        benchmark_translation()
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()