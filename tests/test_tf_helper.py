#!/usr/bin/env python3
"""
Unit tests for ptycho/tf_helper.py, focusing on patch reassembly logic.

This test suite validates the core functionality of the reassemble_position function
by testing its fundamental properties without making assumptions about exact output sizes.
"""

import unittest
import numpy as np
import tensorflow as tf
import os
import sys

# Add project root to path to allow for ptycho imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho.tf_helper import reassemble_position
from ptycho import params as p

class TestReassemblePosition(unittest.TestCase):
    """
    Test suite for the `reassemble_position` function.
    
    These tests verify the core functionality of the patch reassembly logic
    by testing fundamental properties without making assumptions about exact
    output coordinates or sizes.
    """

    def setUp(self):
        """Set a random seed for deterministic tests and default config."""
        np.random.seed(42)
        tf.random.set_seed(42)
        # Set gridsize=2 to mimic real-world multi-channel conditions
        p.set('gridsize', 2)

    def test_perfect_overlap_averages_to_identity(self):
        """
        Test 1: Perfect Overlap Averaging
        
        When two identical patches are placed at the same offset, all non-zero
        pixels in the result should equal the original patch value.
        """
        print("\n--- Test 1: Perfect Overlap Averaging ---")
        N = 32
        M = 16
        
        # Create two identical patches
        patch_value = 1.0 + 2.0j
        patch = tf.constant(patch_value, shape=(1, N, N, 1), dtype=tf.complex64)
        obj_tensor = tf.concat([patch, patch], axis=0)
        
        # Position them at the exact same offset
        offsets = np.array([[0, 0], [0, 0]], dtype=np.float64)
        global_offsets = offsets.reshape((2, 1, 2, 1))
        
        # Execute reassembly
        result = reassemble_position(obj_tensor, global_offsets, M=M)
        result_np = result.numpy()
        
        # All non-zero pixels should equal the original patch value
        non_zero_mask = np.abs(result_np) > 1e-10
        non_zero_values = result_np[non_zero_mask]
        
        if len(non_zero_values) > 0:
            np.testing.assert_allclose(non_zero_values, patch_value, rtol=1e-6)
        
        # Additional checks
        self.assertTrue(np.all(np.isfinite(result_np)))
        self.assertGreater(np.sum(np.abs(result_np)), 0)
        
        print("✅ Perfect overlap averaging test passed.")

    def test_identical_patches_single_vs_double(self):
        """
        Test 2: Single vs Double Identical Patches
        
        When you have one patch vs two identical patches at the same position,
        the result should be identical (because avg(x, x) = x).
        """
        print("\n--- Test 2: Single vs Double Identical Patches ---")
        N = 32
        M = 16
        patch_value = 1.0 + 2.0j
        
        # Single patch
        single_patch = tf.constant(patch_value, shape=(1, N, N, 1), dtype=tf.complex64)
        offsets_single = np.array([[0, 0]], dtype=np.float64)
        global_offsets_single = offsets_single.reshape((1, 1, 2, 1))
        
        result_single = reassemble_position(single_patch, global_offsets_single, M=M)
        
        # Double identical patches at same position
        double_patch = tf.constant(patch_value, shape=(2, N, N, 1), dtype=tf.complex64)
        offsets_double = np.array([[0, 0], [0, 0]], dtype=np.float64)
        global_offsets_double = offsets_double.reshape((2, 1, 2, 1))
        
        result_double = reassemble_position(double_patch, global_offsets_double, M=M)
        
        # Results should be identical
        np.testing.assert_allclose(result_single.numpy(), result_double.numpy(), rtol=1e-6)
        
        print("✅ Single vs double identical patches test passed.")

    def test_basic_functionality(self):
        """
        Test 3: Basic Functionality
        
        Ensures the function runs without errors and produces reasonable output.
        """
        print("\n--- Test 3: Basic Functionality ---")
        N = 32
        M = 16
        
        # Create simple test input
        obj_tensor = tf.ones((4, N, N, 1), dtype=tf.complex64)
        offsets = np.array([[0, 0], [0, 20], [20, 0], [20, 20]], dtype=np.float64)
        global_offsets = offsets.reshape((4, 1, 2, 1))
        
        # Execute reassembly
        result = reassemble_position(obj_tensor, global_offsets, M=M)
        
        # Basic assertions
        self.assertEqual(result.dtype, tf.complex64)
        self.assertEqual(len(result.shape), 3)
        self.assertEqual(result.shape[2], 1)
        self.assertTrue(np.all(np.isfinite(result.numpy())))
        self.assertGreater(np.sum(np.abs(result.numpy())), 0)
        
        print("✅ Basic functionality test passed.")

    def test_different_patch_values_blend(self):
        """
        Test 4: Different Patch Values Blend
        
        When patches with different values overlap, the result should contain
        values that are between the original patch values, indicating averaging.
        """
        print("\n--- Test 4: Different Patch Values Blend ---")
        N = 32
        M = 16
        
        # Create patches with different values
        value1 = 1.0 + 0.0j
        value2 = 5.0 + 0.0j
        
        patch1 = tf.constant(value1, shape=(1, N, N, 1), dtype=tf.complex64)
        patch2 = tf.constant(value2, shape=(1, N, N, 1), dtype=tf.complex64)
        obj_tensor = tf.concat([patch1, patch2], axis=0)
        
        # Position patches to overlap
        offsets = np.array([[0, 0], [0, 0]], dtype=np.float64)  # Perfect overlap
        global_offsets = offsets.reshape((2, 1, 2, 1))
        
        # Execute reassembly
        result = reassemble_position(obj_tensor, global_offsets, M=M)
        result_np = result.numpy()
        
        # Find non-zero values
        non_zero_mask = np.abs(result_np) > 1e-10
        non_zero_values = result_np[non_zero_mask]
        
        if len(non_zero_values) > 0:
            # All non-zero values should be the expected average
            expected_avg = (value1 + value2) / 2.0  # Should be 3.0 + 0.0j
            np.testing.assert_allclose(non_zero_values, expected_avg, rtol=1e-6)
        
        # Additional checks
        self.assertTrue(np.all(np.isfinite(result_np)))
        self.assertGreater(np.sum(np.abs(result_np)), 0)
        
        print("✅ Different patch values blend test passed.")

if __name__ == '__main__':
    unittest.main(verbosity=2)