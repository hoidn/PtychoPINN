#!/usr/bin/env python3
"""
Verify the scaling bug in illuminate_and_diffract.
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho.diffsim import illuminate_and_diffract, scale_nphotons
from ptycho import params as p
from ptycho import tf_helper as hh

def test_scaling_bug():
    """Test if the scaling bug exists in illuminate_and_diffract."""
    print("=== Testing Scaling Bug ===")
    
    # Create dummy data
    Y_I = tf.ones((5, 32, 32, 1), dtype=tf.float32)  # Uniform amplitude
    Y_phi = tf.zeros((5, 32, 32, 1), dtype=tf.float32)  # Zero phase
    probe = tf.ones((32, 32, 1), dtype=tf.complex64)  # Uniform probe
    
    results = {}
    
    for nphotons in [1e4, 1e9]:
        print(f"\n--- Testing nphotons = {nphotons} ---")
        p.set('nphotons', nphotons)
        p.set('batch_size', 5)
        
        # Call illuminate_and_diffract
        X, Y_I_out, Y_phi_out, intensity_scale = illuminate_and_diffract(Y_I, Y_phi, probe)
        
        print(f"intensity_scale: {intensity_scale:.6e}")
        print(f"X mean: {tf.reduce_mean(X).numpy():.6e}")
        print(f"X max: {tf.reduce_max(X).numpy():.6e}")
        
        results[nphotons] = {
            'intensity_scale': intensity_scale,
            'X_mean': tf.reduce_mean(X).numpy(),
            'X_max': tf.reduce_max(X).numpy()
        }
    
    # Check if the values are proportional to nphotons
    print(f"\n=== COMPARISON ===")
    ratio_scale = results[1e9]['intensity_scale'] / results[1e4]['intensity_scale']
    ratio_X_mean = results[1e9]['X_mean'] / results[1e4]['X_mean']
    ratio_X_max = results[1e9]['X_max'] / results[1e4]['X_max']
    
    expected_ratio = np.sqrt(1e9 / 1e4)  # Should be sqrt(1e5) = 316
    
    print(f"Intensity scale ratio: {ratio_scale:.2e} (expected: {expected_ratio:.2e})")
    print(f"X mean ratio: {ratio_X_mean:.2e} (expected: ~{expected_ratio**2:.2e})")
    print(f"X max ratio: {ratio_X_max:.2e} (expected: ~{expected_ratio**2:.2e})")
    
    if abs(ratio_X_mean - 1.0) < 0.1:
        print("🚨 BUG CONFIRMED: X values are nearly identical despite different nphotons!")
    else:
        print("✓ X values scale correctly with nphotons")

def test_fixed_version():
    """Test what happens if we don't divide by intensity_scale."""
    print(f"\n=== Testing Fixed Version (No Division) ===")
    
    # Simplified version without the division bug
    Y_I = tf.ones((5, 32, 32, 1), dtype=tf.float32)
    Y_phi = tf.zeros((5, 32, 32, 1), dtype=tf.float32) 
    probe = tf.ones((32, 32, 1), dtype=tf.complex64)
    
    results_fixed = {}
    
    for nphotons in [1e4, 1e9]:
        print(f"\n--- Fixed version: nphotons = {nphotons} ---")
        p.set('nphotons', nphotons)
        
        # Manual implementation without the bug
        probe_amplitude = tf.cast(tf.abs(probe), Y_I.dtype)
        intensity_scale = scale_nphotons(Y_I * probe_amplitude[None, ...]).numpy()
        
        obj = intensity_scale * hh.combine_complex(Y_I, Y_phi)
        obj = obj * tf.cast(probe[None, ...], obj.dtype)
        
        # Simulate diffraction (simplified)
        X_fixed = tf.abs(tf.signal.fft2d(obj))**2  # Simple diffraction
        X_fixed = tf.sqrt(X_fixed)  # Convert to amplitude
        
        # DON'T divide by intensity_scale here!
        
        print(f"intensity_scale: {intensity_scale:.6e}")
        print(f"X_fixed mean: {tf.reduce_mean(X_fixed).numpy():.6e}")
        print(f"X_fixed max: {tf.reduce_max(X_fixed).numpy():.6e}")
        
        results_fixed[nphotons] = {
            'X_mean': tf.reduce_mean(X_fixed).numpy(),
            'X_max': tf.reduce_max(X_fixed).numpy()
        }
    
    # Check ratios for fixed version
    ratio_mean_fixed = results_fixed[1e9]['X_mean'] / results_fixed[1e4]['X_mean']
    ratio_max_fixed = results_fixed[1e9]['X_max'] / results_fixed[1e4]['X_max']
    
    expected_ratio_squared = (1e9 / 1e4)  # Should be 1e5
    
    print(f"\nFixed version ratios:")
    print(f"X mean ratio: {ratio_mean_fixed:.2e} (expected: ~{expected_ratio_squared:.2e})")
    print(f"X max ratio: {ratio_max_fixed:.2e} (expected: ~{expected_ratio_squared:.2e})")
    
    if abs(np.log10(ratio_mean_fixed) - 5) < 1:  # Should be around 10^5
        print("✓ FIXED: X values now scale correctly with nphotons!")
    else:
        print("❌ Still not working correctly")

if __name__ == "__main__":
    test_scaling_bug()
    test_fixed_version()