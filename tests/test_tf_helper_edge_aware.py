#!/usr/bin/env python3
"""
Edge-aware tests for translate functions that account for interpolation differences.

This test suite specifically handles the known differences between TFA and TF raw ops
regarding edge interpolation while ensuring smooth patterns (as used in PtychoPINN)
work correctly.
"""

import unittest
import numpy as np
import tensorflow as tf
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ptycho.tf_helper import translate_core, translate

# Handle optional tensorflow_addons import
try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except ImportError:
    HAS_TFA = False
    # Mock TFA for testing without it
    class MockTFA:
        class image:
            @staticmethod
            def translate(imgs, offsets):
                # Simple mock that returns input (no actual translation)
                return imgs
    tfa = MockTFA()


class TestTranslateSmoothPatterns(unittest.TestCase):
    """Test translation with smooth patterns (relevant for PtychoPINN)."""
    
    def setUp(self):
        """Set a random seed for deterministic tests."""
        np.random.seed(42)
        tf.random.set_seed(42)
        
    @unittest.skipUnless(HAS_TFA, "tensorflow_addons not available")
    def test_gaussian_probe_translation(self):
        """Test translation of Gaussian-like probe patterns."""
        print("\n--- Test: Gaussian probe pattern translation ---")
        
        # Create Gaussian probe pattern
        size = 64
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        xx, yy = np.meshgrid(x, y)
        probe = np.exp(-(xx**2 + yy**2) / 0.5).astype(np.float32)
        
        # Test various offsets
        test_cases = [
            ([0.0, 0.0], "Zero translation"),
            ([2.0, 0.0], "Integer X"),
            ([0.0, -3.0], "Integer Y"),
            ([1.5, 2.5], "Sub-pixel positive"),
            ([-0.7, -1.3], "Sub-pixel negative"),
        ]
        
        for offset, name in test_cases:
            with self.subTest(case=name):
                tf_probe = tf.constant(probe[np.newaxis, :, :, np.newaxis])
                tf_offset = tf.constant([offset], dtype=tf.float32)
                
                result_tfa = tfa.image.translate(tf_probe, tf_offset)
                result_core = translate_core(tf_probe, tf_offset)
                
                max_diff = tf.reduce_max(tf.abs(result_tfa - result_core)).numpy()
                
                # For smooth patterns, we expect good agreement
                self.assertLess(
                    max_diff, 0.05,  # 5% tolerance
                    f"{name}: Max diff {max_diff:.3e} exceeds tolerance"
                )
        
        print("✅ Gaussian probe translation test passed")
    
    @unittest.skipUnless(HAS_TFA, "tensorflow_addons not available")
    def test_smooth_object_translation(self):
        """Test translation of smooth object-like patterns."""
        print("\n--- Test: Smooth object pattern translation ---")
        
        # Create smooth object pattern with multiple features
        size = 64
        x = np.linspace(0, 4*np.pi, size)
        y = np.linspace(0, 4*np.pi, size)
        xx, yy = np.meshgrid(x, y)
        
        # Combine smooth variations
        object_pattern = (
            0.5 * np.sin(xx/2) * np.cos(yy/2) +
            0.3 * np.exp(-((xx-2*np.pi)**2 + (yy-2*np.pi)**2) / 10) +
            0.5
        ).astype(np.float32)
        
        # Normalize to [0, 1]
        object_pattern = (object_pattern - object_pattern.min()) / (object_pattern.max() - object_pattern.min())
        
        tf_object = tf.constant(object_pattern[np.newaxis, :, :, np.newaxis])
        offsets = tf.constant([[2.7, -1.3]], dtype=tf.float32)
        
        result_tfa = tfa.image.translate(tf_object, offsets)
        result_core = translate_core(tf_object, offsets)
        
        max_diff = tf.reduce_max(tf.abs(result_tfa - result_core)).numpy()
        mean_diff = tf.reduce_mean(tf.abs(result_tfa - result_core)).numpy()
        
        # For smooth patterns, differences should be small
        self.assertLess(max_diff, 0.1, f"Max diff {max_diff:.3e} too large")
        self.assertLess(mean_diff, 0.01, f"Mean diff {mean_diff:.3e} too large")
        
        print(f"✅ Smooth object translation: max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e}")
    
    def test_complex_smooth_translation(self):
        """Test translation of complex-valued smooth patterns."""
        print("\n--- Test: Complex smooth pattern translation ---")
        
        # Create complex smooth pattern
        size = 32
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        
        # Amplitude: Gaussian
        amplitude = np.exp(-(xx**2 + yy**2) / 0.3)
        # Phase: smooth variation
        phase = np.pi * (xx + yy) / 2
        
        complex_pattern = amplitude * np.exp(1j * phase)
        # Cast to complex64 for consistent dtype handling with TF 2.19+
        tf_complex = tf.cast(tf.constant(complex_pattern[np.newaxis, :, :, np.newaxis]), tf.complex64)
        
        offsets = tf.constant([[1.5, -0.7]], dtype=tf.float32)
        
        # Use wrapped translate function that handles complex
        result = translate(tf_complex, offsets)
        
        # Verify result is reasonable
        self.assertEqual(result.dtype, tf.complex64)
        self.assertTrue(np.all(np.isfinite(result.numpy())))
        
        # For complex patterns, just verify the magnitude is preserved reasonably
        input_magnitude = tf.abs(tf_complex)
        output_magnitude = tf.abs(result)
        
        # Total energy should be approximately preserved (allowing for boundary effects)
        # Both energies now have consistent float32 dtype
        input_energy = tf.reduce_sum(input_magnitude**2)
        output_energy = tf.reduce_sum(output_magnitude**2)
        energy_ratio = output_energy / input_energy
        
        self.assertGreater(energy_ratio, 0.8, "Too much energy loss")
        self.assertLess(energy_ratio, 1.2, "Energy increased too much")
        
        print(f"✅ Complex smooth translation: energy ratio={energy_ratio:.3f}")


class TestTranslateEdgeCases(unittest.TestCase):
    """Test and document expected edge handling differences."""
    
    @unittest.skipUnless(HAS_TFA, "tensorflow_addons not available")
    def test_document_edge_differences(self):
        """Document the expected differences in edge handling."""
        print("\n--- Documenting Edge Handling Differences ---")
        
        # Create pattern with sharp edge
        size = 32
        sharp_pattern = np.zeros((size, size), dtype=np.float32)
        sharp_pattern[10:20, 10:20] = 1.0  # Sharp square
        
        tf_sharp = tf.constant(sharp_pattern[np.newaxis, :, :, np.newaxis])
        offset = tf.constant([[0.5, 0.5]], dtype=tf.float32)
        
        result_tfa = tfa.image.translate(tf_sharp, offset)
        result_core = translate_core(tf_sharp, offset)
        
        max_diff = tf.reduce_max(tf.abs(result_tfa - result_core)).numpy()
        
        print(f"Sharp edge pattern max difference: {max_diff:.3e}")
        print("This difference is expected due to different interpolation strategies:")
        print("- TFA: Specialized edge handling")
        print("- TF raw ops: Standard bilinear interpolation")
        
        # We expect significant differences for sharp edges
        self.assertGreater(max_diff, 0.1, "Expected significant edge differences")
        self.assertLess(max_diff, 1.0, "Differences should not exceed pixel values")
        
        print("✅ Edge handling differences documented")
    
    @unittest.skipUnless(HAS_TFA, "tensorflow_addons not available")
    def test_boundary_behavior(self):
        """Test behavior at image boundaries."""
        print("\n--- Test: Boundary behavior ---")
        
        # Pattern near boundaries
        size = 32
        boundary_pattern = np.zeros((size, size), dtype=np.float32)
        boundary_pattern[:5, :] = 0.5   # Top edge
        boundary_pattern[-5:, :] = 0.5  # Bottom edge
        boundary_pattern[:, :5] = 0.5   # Left edge
        boundary_pattern[:, -5:] = 0.5  # Right edge
        
        tf_boundary = tf.constant(boundary_pattern[np.newaxis, :, :, np.newaxis])
        offset = tf.constant([[2.3, -1.7]], dtype=tf.float32)
        
        result_tfa = tfa.image.translate(tf_boundary, offset)
        result_core = translate_core(tf_boundary, offset)
        
        # Both should handle boundaries, just potentially differently
        self.assertTrue(np.all(np.isfinite(result_tfa.numpy())))
        self.assertTrue(np.all(np.isfinite(result_core.numpy())))
        
        print("✅ Boundary behavior test passed")


class TestPtychoPINNRelevantCases(unittest.TestCase):
    """Test cases specifically relevant to PtychoPINN usage."""
    
    @unittest.skipUnless(HAS_TFA, "tensorflow_addons not available")
    def test_typical_probe_sizes(self):
        """Test with typical probe sizes used in PtychoPINN."""
        print("\n--- Test: Typical PtychoPINN probe sizes ---")
        
        probe_sizes = [32, 64, 128]
        
        for size in probe_sizes:
            with self.subTest(size=size):
                # Create typical probe
                x = np.linspace(-2, 2, size)
                y = np.linspace(-2, 2, size)
                xx, yy = np.meshgrid(x, y)
                probe = np.exp(-(xx**2 + yy**2) / 0.5).astype(np.float32)
                
                tf_probe = tf.constant(probe[np.newaxis, :, :, np.newaxis])
                offset = tf.constant([[1.23, -0.87]], dtype=tf.float32)
                
                result_tfa = tfa.image.translate(tf_probe, offset)
                result_core = translate_core(tf_probe, offset)
                
                max_diff = tf.reduce_max(tf.abs(result_tfa - result_core)).numpy()
                
                self.assertLess(
                    max_diff, 0.05,
                    f"Size {size}: max_diff={max_diff:.3e}"
                )
        
        print("✅ Typical probe sizes test passed")
    
    @unittest.skipUnless(HAS_TFA, "tensorflow_addons not available")
    def test_batch_smooth_patterns(self):
        """Test batch processing with smooth patterns only."""
        print("\n--- Test: Batch processing with smooth patterns ---")
        
        batch_size = 16
        size = 32
        
        # Create batch of smooth patterns
        batch_patterns = []
        for i in range(batch_size):
            x = np.linspace(-2, 2, size)
            y = np.linspace(-2, 2, size)
            xx, yy = np.meshgrid(x, y)
            # Different width Gaussians
            sigma = 0.3 + i * 0.1
            pattern = np.exp(-(xx**2 + yy**2) / sigma).astype(np.float32)
            batch_patterns.append(pattern)
        
        tf_batch = tf.stack([p[..., np.newaxis] for p in batch_patterns], axis=0)
        
        # Different offsets for each
        offsets = tf.constant([
            [0.0, 0.0],
            [1.5, -0.5],
            [-0.7, 1.3],
            [2.2, -1.8]
        ], dtype=tf.float32)
        
        result_tfa = tfa.image.translate(tf_batch, offsets)
        result_core = translate_core(tf_batch, offsets)
        
        max_diff = tf.reduce_max(tf.abs(result_tfa - result_core)).numpy()
        
        self.assertLess(
            max_diff, 0.1,
            f"Batch smooth patterns max_diff={max_diff:.3e}"
        )
        
        print(f"✅ Batch smooth patterns: max_diff={max_diff:.3e}")


if __name__ == '__main__':
    # Run tests
    print("\n" + "="*70)
    print("EDGE-AWARE TRANSLATION TESTS")
    print("Testing with patterns relevant to PtychoPINN usage")
    print("="*70)
    
    unittest.main(verbosity=2)