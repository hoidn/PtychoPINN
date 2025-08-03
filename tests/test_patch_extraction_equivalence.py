"""
Test suite for validating numerical equivalence between iterative and batched
patch extraction implementations.

This module contains critical tests that prove the new high-performance batched
implementation produces identical results to the legacy iterative implementation.
"""
import unittest
import numpy as np
import tensorflow as tf
from ptycho.raw_data import _get_image_patches_iterative, _get_image_patches_batched


class TestPatchExtractionEquivalence(unittest.TestCase):
    """Tests to ensure batched and iterative implementations are numerically equivalent."""
    
    def setUp(self):
        """Set up test environment."""
        # Enable eager execution for testing
        tf.config.run_functions_eagerly(True)
        
    def test_numerical_equivalence(self):
        """Test that batched and iterative implementations produce identical results."""
        # 1. Generate test data
        N, B, c = 64, 4, 4
        obj_size = 200
        gt_image = tf.complex(
            tf.random.normal((obj_size, obj_size), dtype=tf.float32),
            tf.random.normal((obj_size, obj_size), dtype=tf.float32)
        )
        gt_padded = tf.pad(gt_image[None, ..., None], [[0, 0], [N//2, N//2], [N//2, N//2], [0, 0]])
        offsets_f = tf.random.uniform((B*c, 1, 1, 2), minval=-50, maxval=50, dtype=tf.float32)

        # 2. Run iterative (legacy) implementation
        iterative_result = _get_image_patches_iterative(gt_padded, offsets_f, N, B, c)

        # 3. Run batched (new) implementation
        batched_result = _get_image_patches_batched(gt_padded, offsets_f, N, B, c)

        # 4. Assert numerical equivalence
        np.testing.assert_allclose(
            iterative_result.numpy(),
            batched_result.numpy(),
            atol=1e-6,
            err_msg="Batched implementation output does not match iterative implementation."
        )
        print("✓ Numerical equivalence test passed.")
        
    def test_equivalence_multiple_configurations(self):
        """Test equivalence across various parameter configurations."""
        test_configs = [
            (32, 1, 1),    # N=32, B=1, gridsize=1
            (64, 5, 1),    # N=64, B=5, gridsize=1
            (64, 10, 4),   # N=64, B=10, gridsize=2
            (128, 3, 9),   # N=128, B=3, gridsize=3
        ]
        
        for N, B, c in test_configs:
            with self.subTest(N=N, B=B, c=c):
                # Generate test image
                obj_size = N * 3  # Ensure image is large enough
                gt_image = tf.complex(
                    tf.random.normal((obj_size, obj_size), dtype=tf.float32),
                    tf.random.normal((obj_size, obj_size), dtype=tf.float32)
                )
                gt_padded = tf.pad(gt_image[None, ..., None], 
                                  [[0, 0], [N//2, N//2], [N//2, N//2], [0, 0]])
                
                # Generate random offsets
                offsets_f = tf.random.uniform((B*c, 1, 1, 2), 
                                            minval=-N//4, maxval=N//4, dtype=tf.float32)
                
                # Run both implementations
                iterative_result = _get_image_patches_iterative(gt_padded, offsets_f, N, B, c)
                batched_result = _get_image_patches_batched(gt_padded, offsets_f, N, B, c)
                
                # Assert equivalence
                np.testing.assert_allclose(
                    iterative_result.numpy(),
                    batched_result.numpy(),
                    atol=1e-6,
                    err_msg=f"Mismatch for config N={N}, B={B}, c={c}"
                )
                
    def test_edge_case_zero_offsets(self):
        """Test with zero offsets (no translation)."""
        N, B, c = 64, 2, 4
        obj_size = 200
        
        # Create a simple pattern that's easy to verify
        x = tf.range(obj_size, dtype=tf.float32)
        y = tf.range(obj_size, dtype=tf.float32)
        xx, yy = tf.meshgrid(x, y)
        gt_image = tf.complex(xx, yy)
        gt_padded = tf.pad(gt_image[None, ..., None], 
                          [[0, 0], [N//2, N//2], [N//2, N//2], [0, 0]])
        
        # Zero offsets
        offsets_f = tf.zeros((B*c, 1, 1, 2), dtype=tf.float32)
        
        # Run both implementations
        iterative_result = _get_image_patches_iterative(gt_padded, offsets_f, N, B, c)
        batched_result = _get_image_patches_batched(gt_padded, offsets_f, N, B, c)
        
        # Assert equivalence
        np.testing.assert_allclose(
            iterative_result.numpy(),
            batched_result.numpy(),
            atol=1e-6,
            err_msg="Mismatch for zero offsets"
        )
        
    def test_edge_case_large_offsets(self):
        """Test with large offsets near image boundaries."""
        N, B, c = 64, 3, 1
        obj_size = 200
        
        gt_image = tf.complex(
            tf.random.normal((obj_size, obj_size), dtype=tf.float32),
            tf.random.normal((obj_size, obj_size), dtype=tf.float32)
        )
        gt_padded = tf.pad(gt_image[None, ..., None], 
                          [[0, 0], [N//2, N//2], [N//2, N//2], [0, 0]])
        
        # Large offsets that push patches near boundaries
        large_offset = obj_size // 2 - N // 2 - 5
        offsets_f = tf.constant([
            [[[large_offset, 0]]],      # Far right
            [[[0, large_offset]]],      # Far bottom
            [[[-large_offset, 0]]],     # Far left
        ], dtype=tf.float32)
        
        # Run both implementations
        iterative_result = _get_image_patches_iterative(gt_padded, offsets_f, N, B, c)
        batched_result = _get_image_patches_batched(gt_padded, offsets_f, N, B, c)
        
        # Assert equivalence
        np.testing.assert_allclose(
            iterative_result.numpy(),
            batched_result.numpy(),
            atol=1e-6,
            err_msg="Mismatch for large offsets near boundaries"
        )


if __name__ == '__main__':
    unittest.main()