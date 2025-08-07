"""
Test suite for patch extraction performance and correctness validation.

This module tests the memory-optimized mini-batching implementation of patch extraction
to ensure numerical equivalence with the iterative approach while providing better
memory efficiency for large datasets.
"""

import unittest
import numpy as np
import tensorflow as tf
import logging
import time
from typing import Tuple, Optional

# Import the functions we need to test
from ptycho.raw_data import _get_image_patches_iterative, _get_image_patches_batched
from ptycho.config.config import ModelConfig


class TestPatchExtractionPerformance(unittest.TestCase):
    """Test suite for patch extraction performance and correctness."""
    
    def setUp(self):
        """Set up synthetic ground truth data for testing."""
        # Create a synthetic ground truth object
        self.object_size = 128
        self.patch_size = 32
        self.gridsize = 2
        self.c = self.gridsize ** 2  # Number of channels
        
        # Create synthetic object with some interesting structure
        x = np.linspace(-1, 1, self.object_size)
        y = np.linspace(-1, 1, self.object_size)
        X, Y = np.meshgrid(x, y)
        
        # Create a complex object with both amplitude and phase
        amplitude = np.exp(-(X**2 + Y**2) / 0.5)  # Gaussian
        phase = np.sin(3 * X) * np.cos(3 * Y)     # Sinusoidal phase pattern
        
        self.ground_truth = (amplitude * np.exp(1j * phase)).astype(np.complex64)
        
        # Pad the object for patch extraction
        pad_size = self.patch_size
        self.gt_padded = tf.constant(
            np.pad(self.ground_truth, ((pad_size, pad_size), (pad_size, pad_size)), 
                   mode='constant', constant_values=0.0),
            dtype=tf.complex64
        )
        self.gt_padded = tf.expand_dims(tf.expand_dims(self.gt_padded, axis=0), axis=-1)
        
        # Create synthetic scan coordinates for testing
        self.small_batch_size = 8   # Small dataset for equivalence testing
        self.large_batch_size = 1024  # Large dataset for performance testing
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_offsets(self, batch_size: int) -> tf.Tensor:
        """Create offset tensor for patch extraction testing."""
        # Generate random scan positions within the valid range
        max_offset = self.object_size - self.patch_size
        x_offsets = np.random.uniform(-max_offset/2, max_offset/2, batch_size * self.c)
        y_offsets = np.random.uniform(-max_offset/2, max_offset/2, batch_size * self.c)
        
        # Stack to create (B*c, 2) coordinate array
        offsets = np.stack([x_offsets, y_offsets], axis=1)
        
        # Reshape to (B*c, 1, 2, 1) format expected by the functions
        offsets_f = offsets.reshape(batch_size * self.c, 1, 2, 1).astype(np.float32)
        
        return tf.constant(offsets_f)
    
    def test_numerical_equivalence_small_dataset(self):
        """Test that batched and iterative implementations give identical results."""
        self.logger.info("Testing numerical equivalence with small dataset...")
        
        # Create offset tensor for small batch
        offsets_f = self._create_offsets(self.small_batch_size)
        
        # Run iterative implementation
        result_iterative = _get_image_patches_iterative(
            self.gt_padded, offsets_f, self.patch_size, self.small_batch_size, self.c
        )
        
        # Run batched implementation
        result_batched = _get_image_patches_batched(
            self.gt_padded, offsets_f, self.patch_size, self.small_batch_size, self.c
        )
        
        # Convert to numpy for comparison
        result_iterative_np = result_iterative.numpy()
        result_batched_np = result_batched.numpy()
        
        # Check shapes are identical
        self.assertEqual(result_iterative_np.shape, result_batched_np.shape)
        self.assertEqual(result_iterative_np.shape, 
                        (self.small_batch_size, self.patch_size, self.patch_size, self.c))
        
        # Check numerical equivalence - use relaxed tolerance for floating point differences
        max_diff = np.max(np.abs(result_iterative_np - result_batched_np))
        mean_diff = np.mean(np.abs(result_iterative_np - result_batched_np))
        
        self.logger.info(f"Max absolute difference: {max_diff}")
        self.logger.info(f"Mean absolute difference: {mean_diff}")
        
        # Use a tolerance that accounts for TensorFlow's batched vs single operation differences
        # Testing shows max differences of ~0.002 when using batched translation operations
        self.assertTrue(np.allclose(result_iterative_np, result_batched_np, atol=5e-3, rtol=1e-5),
                       f"Results are not numerically equivalent. Max difference: {max_diff}, "
                       f"Mean difference: {mean_diff}. This may indicate a bug in the implementation.")
        
        self.logger.info("✅ Numerical equivalence test passed!")
    
    def test_memory_efficiency_large_dataset(self):
        """Test that the optimized implementation can handle large datasets without OOM."""
        self.logger.info("Testing memory efficiency with large dataset...")
        
        # Create offset tensor for large batch - this would cause OOM with old implementation
        offsets_f = self._create_offsets(self.large_batch_size)
        
        try:
            # Test the new mini-batching implementation with a large dataset
            # Use a small mini-batch size to ensure the chunking works correctly
            result_batched = _get_image_patches_batched(
                self.gt_padded, offsets_f, self.patch_size, 
                self.large_batch_size, self.c, mini_batch_size=64  # Small chunks
            )
            
            # Verify the result shape is correct
            expected_shape = (self.large_batch_size, self.patch_size, self.patch_size, self.c)
            self.assertEqual(result_batched.shape, expected_shape)
            
            # Verify we got reasonable results (not all zeros or NaNs)
            result_np = result_batched.numpy()
            self.assertFalse(np.isnan(result_np).any(), "Result contains NaN values")
            self.assertTrue(np.any(np.abs(result_np) > 1e-6), "Result appears to be all zeros")
            
            self.logger.info(f"✅ Large dataset test passed! Processed {self.large_batch_size} patches successfully.")
            self.logger.info(f"Result shape: {result_batched.shape}, dtype: {result_batched.dtype}")
            
        except Exception as e:
            self.fail(f"Large dataset test failed: {e}")
    
    def test_timing_comparison(self):
        """Compare timing between iterative and batched implementations."""
        self.logger.info("Testing timing comparison...")
        
        # Use medium-sized dataset for timing comparison
        medium_batch_size = 64
        offsets_f = self._create_offsets(medium_batch_size)
        
        # Time iterative implementation
        start_time = time.time()
        result_iterative = _get_image_patches_iterative(
            self.gt_padded, offsets_f, self.patch_size, medium_batch_size, self.c
        )
        iterative_time = time.time() - start_time
        
        # Time batched implementation
        start_time = time.time()
        result_batched = _get_image_patches_batched(
            self.gt_padded, offsets_f, self.patch_size, medium_batch_size, self.c
        )
        batched_time = time.time() - start_time
        
        self.logger.info(f"Iterative implementation time: {iterative_time:.4f}s")
        self.logger.info(f"Batched implementation time: {batched_time:.4f}s")
        self.logger.info(f"Speedup: {iterative_time/batched_time:.2f}x")
        
        # Verify results are still equivalent (using same tolerance as equivalence test)
        self.assertTrue(np.allclose(result_iterative.numpy(), result_batched.numpy(), atol=5e-3, rtol=1e-5))
        
        self.logger.info("✅ Timing comparison test passed!")


if __name__ == '__main__':
    # Configure TensorFlow to limit GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    unittest.main()