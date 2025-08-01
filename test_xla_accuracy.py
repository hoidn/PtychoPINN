#!/usr/bin/env python3
"""Test numerical accuracy of XLA translation vs original implementation."""

import os
import numpy as np
import tensorflow as tf

# Temporarily disable XLA for comparison
os.environ['USE_XLA_TRANSLATE'] = '0'

from ptycho.tf_helper import translate_core
from ptycho.projective_warp_xla import translate_xla

def test_numerical_accuracy():
    """Compare XLA and non-XLA implementations for numerical accuracy."""
    print("Testing Numerical Accuracy: XLA vs Original Implementation")
    print("=" * 60)
    
    # Test configurations
    test_cases = [
        {"batch": 1, "size": 32, "channels": 1, "name": "Small single batch"},
        {"batch": 4, "size": 64, "channels": 1, "name": "Medium batch"},
        {"batch": 8, "size": 128, "channels": 3, "name": "Large RGB batch"},
        {"batch": 2, "size": 256, "channels": 1, "name": "High resolution"},
    ]
    
    translation_cases = [
        {"dx": 0.0, "dy": 0.0, "name": "Zero translation"},
        {"dx": 5.5, "dy": -3.3, "name": "Fractional translation"},
        {"dx": 10.0, "dy": 10.0, "name": "Integer translation"},
        {"dx": -15.7, "dy": 8.2, "name": "Mixed translation"},
    ]
    
    all_passed = True
    
    for test_config in test_cases:
        print(f"\n{test_config['name']}:")
        print("-" * 40)
        
        # Create test image with known pattern
        batch = test_config['batch']
        size = test_config['size']
        channels = test_config['channels']
        
        # Create a gradient pattern for better testing
        x = tf.linspace(0.0, 1.0, size)
        y = tf.linspace(0.0, 1.0, size)
        xx, yy = tf.meshgrid(x, y)
        
        # Create different patterns for each channel
        patterns = []
        for c in range(channels):
            if c == 0:
                pattern = xx  # Horizontal gradient
            elif c == 1:
                pattern = yy  # Vertical gradient
            else:
                pattern = xx * yy  # Diagonal gradient
            patterns.append(pattern)
        
        image = tf.stack(patterns, axis=-1)
        image = tf.expand_dims(image, 0)  # Add batch dimension
        images = tf.tile(image, [batch, 1, 1, 1])
        
        for trans_case in translation_cases:
            # Create translations
            translations = tf.constant([[trans_case['dx'], trans_case['dy']]] * batch, dtype=tf.float32)
            
            # Original implementation
            result_original = translate_core(images, translations, interpolation='bilinear', use_xla_workaround=False)
            
            # XLA implementation
            result_xla = translate_xla(images, translations, interpolation='bilinear', use_jit=False)
            
            # Compare results
            diff = tf.abs(result_original - result_xla)
            max_diff = tf.reduce_max(diff)
            mean_diff = tf.reduce_mean(diff)
            
            # Check if within tolerance
            tolerance = 1e-5
            passed = max_diff < tolerance
            all_passed = all_passed and passed
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {trans_case['name']}: {status}")
            print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
            
            if not passed:
                print(f"    WARNING: Difference exceeds tolerance of {tolerance}")
                # Print some debug info
                print(f"    Original min/max: {tf.reduce_min(result_original):.4f}/{tf.reduce_max(result_original):.4f}")
                print(f"    XLA min/max: {tf.reduce_min(result_xla):.4f}/{tf.reduce_max(result_xla):.4f}")
    
    return all_passed

def test_complex_accuracy():
    """Test accuracy with complex-valued tensors."""
    print("\n\nTesting Complex Number Accuracy")
    print("=" * 60)
    
    # Test configuration
    batch_size = 4
    height = width = 64
    channels = 1
    
    # Create complex test pattern
    x = tf.linspace(0.0, 2*np.pi, width)
    y = tf.linspace(0.0, 2*np.pi, height)
    xx, yy = tf.meshgrid(x, y)
    
    # Create sinusoidal pattern
    real_part = tf.cos(xx) * tf.sin(yy)
    imag_part = tf.sin(xx) * tf.cos(yy)
    
    real_part = tf.expand_dims(tf.expand_dims(real_part, 0), -1)
    imag_part = tf.expand_dims(tf.expand_dims(imag_part, 0), -1)
    
    real_images = tf.tile(real_part, [batch_size, 1, 1, 1])
    imag_images = tf.tile(imag_part, [batch_size, 1, 1, 1])
    complex_images = tf.complex(real_images, imag_images)
    
    # Test translations
    translations = tf.constant([[5.0, -3.0], [-7.0, 2.0], [0.0, 0.0], [10.5, -10.5]], dtype=tf.float32)
    
    # Original implementation (with complexify decorator)
    from ptycho.tf_helper import translate
    result_original = translate(complex_images, translations, use_xla_workaround=False)
    
    # XLA implementation
    result_xla = translate_xla(complex_images, translations, interpolation='bilinear', use_jit=False)
    
    # Compare results
    diff = tf.abs(result_original - result_xla)
    max_diff = tf.reduce_max(diff)
    mean_diff = tf.reduce_mean(diff)
    
    tolerance = 1e-5
    passed = max_diff < tolerance
    
    print(f"Complex translation accuracy: {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    # Check amplitude and phase separately
    amp_diff = tf.abs(tf.abs(result_original) - tf.abs(result_xla))
    phase_diff = tf.abs(tf.math.angle(result_original) - tf.math.angle(result_xla))
    
    print(f"Max amplitude difference: {tf.reduce_max(amp_diff):.2e}")
    print(f"Max phase difference: {tf.reduce_max(phase_diff):.2e}")
    
    return passed

def test_interpolation_modes():
    """Test different interpolation modes."""
    print("\n\nTesting Interpolation Modes")
    print("=" * 60)
    
    # Create test image
    images = tf.random.uniform((2, 64, 64, 1), dtype=tf.float32)
    translations = tf.constant([[3.7, -2.3], [-5.1, 4.8]], dtype=tf.float32)
    
    for interpolation in ['bilinear', 'nearest']:
        print(f"\n{interpolation.capitalize()} interpolation:")
        
        # Original implementation
        result_original = translate_core(images, translations, interpolation=interpolation, use_xla_workaround=False)
        
        # XLA implementation
        result_xla = translate_xla(images, translations, interpolation=interpolation, use_jit=False)
        
        # Compare
        diff = tf.abs(result_original - result_xla)
        max_diff = tf.reduce_max(diff)
        mean_diff = tf.reduce_mean(diff)
        
        # Nearest neighbor might have slightly higher tolerance due to rounding
        tolerance = 1e-4 if interpolation == 'nearest' else 1e-5
        passed = max_diff < tolerance
        
        print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
        print(f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

def main():
    """Run all accuracy tests."""
    print("XLA Translation Accuracy Tests")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Run tests
    basic_passed = test_numerical_accuracy()
    complex_passed = test_complex_accuracy()
    test_interpolation_modes()
    
    # Summary
    print("\n" + "=" * 60)
    if basic_passed and complex_passed:
        print("✅ All accuracy tests PASSED!")
        print("The XLA implementation produces numerically equivalent results.")
    else:
        print("❌ Some tests FAILED!")
        print("Please investigate the differences between implementations.")

if __name__ == "__main__":
    main()