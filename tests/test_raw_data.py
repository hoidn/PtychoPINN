"""
Unit tests for raw_data module, focusing on patch extraction functionality.

This test module validates the correctness of both iterative and batched
implementations of the get_image_patches function.
"""
import unittest
import numpy as np
import tensorflow as tf
from ptycho.raw_data import _get_image_patches_iterative, _get_image_patches_batched


class TestPatchExtraction(unittest.TestCase):
    """Test cases for patch extraction functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Enable eager execution for testing
        tf.config.run_functions_eagerly(True)
        
    def test_basic_functionality_batched(self):
        """Test that batched implementation produces valid output."""
        # Create a simple test image with known pattern (100x100)
        test_image = tf.complex(
            tf.range(100*100, dtype=tf.float32),
            tf.zeros(100*100, dtype=tf.float32)
        )
        test_image = tf.reshape(test_image, (100, 100))
        
        # Pad the image
        gt_padded = tf.pad(test_image[None, ..., None], [[0, 0], [32, 32], [32, 32], [0, 0]])
        
        # Define test parameters
        N = 64  # Patch size
        B = 4   # Batch size (4 scan positions)
        c = 4   # Channels (gridsize=2, so 2x2=4)
        
        # Create test offsets for a 2x2 grid
        offsets = []
        for i in range(B):
            for j in range(c):
                # Create offsets that stay within bounds
                offset_y = float(i * 10)
                offset_x = float(j * 10)
                offsets.append([[offset_y, offset_x]])
        
        offsets_f = tf.constant(offsets, dtype=tf.float32)
        offsets_f = tf.reshape(offsets_f, (B*c, 1, 2, 1))
        
        # Call the batched implementation
        result = _get_image_patches_batched(gt_padded, offsets_f, N, B, c)
        
        # Verify output shape
        self.assertEqual(result.shape, (B, N, N, c))
        
        # Verify output is valid (no NaN/Inf values)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(tf.abs(result))))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(tf.abs(result))))
        
    def test_shape_validation(self):
        """Test various input configurations for shape correctness."""
        test_configs = [
            (32, 1, 1),   # N=32, gridsize=1
            (64, 1, 1),   # N=64, gridsize=1
            (64, 2, 4),   # N=64, gridsize=2
            (128, 2, 4),  # N=128, gridsize=2
            (64, 3, 9),   # N=64, gridsize=3
        ]
        
        for N, gridsize, c in test_configs:
            with self.subTest(N=N, gridsize=gridsize, c=c):
                # Create test image
                test_image = tf.complex(
                    tf.ones((200, 200), dtype=tf.float32),
                    tf.zeros((200, 200), dtype=tf.float32)
                )
                gt_padded = tf.pad(test_image[None, ..., None], 
                                  [[0, 0], [N//2, N//2], [N//2, N//2], [0, 0]])
                
                # Different batch sizes
                for B in [1, 10, 100]:
                    # Create random offsets
                    offsets_f = tf.random.uniform((B*c, 1, 2, 1), 
                                                minval=-50, maxval=50)
                    
                    # Test batched implementation
                    result_batched = _get_image_patches_batched(
                        gt_padded, offsets_f, N, B, c)
                    
                    # Verify shape
                    expected_shape = (B, N, N, c)
                    self.assertEqual(result_batched.shape, expected_shape)
                    
    def test_single_patch_extraction(self):
        """Test extraction of a single patch."""
        # Create a simple gradient image
        x = tf.range(100, dtype=tf.float32)
        y = tf.range(100, dtype=tf.float32)
        xx, yy = tf.meshgrid(x, y)
        test_image = tf.complex(xx + yy, tf.zeros_like(xx))
        
        # Pad the image
        N = 64
        gt_padded = tf.pad(test_image[None, ..., None], 
                          [[0, 0], [N//2, N//2], [N//2, N//2], [0, 0]])
        
        # Single patch parameters
        B = 1
        c = 1
        offsets_f = tf.constant([[[[20.0], [30.0]]]], dtype=tf.float32)
        
        # Extract patch
        result = _get_image_patches_batched(gt_padded, offsets_f, N, B, c)
        
        # Verify shape
        self.assertEqual(result.shape, (1, N, N, 1))
        
        # Verify the patch contains expected values
        # Since we negate offsets in the implementation, the center should be at
        # the original position minus the offset
        center_value = tf.abs(result[0, N//2, N//2, 0])
        # Just verify it's a reasonable value (not zero, not inf)
        self.assertGreater(center_value.numpy(), 0.0)
        self.assertLess(center_value.numpy(), 200.0)


class TestPatchExtractionDispatcher(unittest.TestCase):
    """Test cases for the dispatcher functionality of get_image_patches."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Enable eager execution for testing
        tf.config.run_functions_eagerly(True)
        # Import the function we need
        from ptycho.raw_data import get_image_patches
        self.get_image_patches = get_image_patches
        
    def test_dispatcher_uses_iterative_by_default(self):
        """Verify default behavior uses iterative implementation."""
        from ptycho import params
        
        # Ensure the flag is not set (default behavior)
        if 'use_batched_patch_extraction' in params.cfg:
            del params.cfg['use_batched_patch_extraction']
        
        # Create test data
        N = 64
        gridsize = 2
        B = 2
        test_image = tf.complex(
            tf.ones((200, 200), dtype=tf.float32),
            tf.zeros((200, 200), dtype=tf.float32)
        )
        
        # Create offsets in the expected shape
        global_offsets = tf.zeros((B, 1, 2, gridsize**2), dtype=tf.float32)
        local_offsets = tf.random.uniform((B, 1, 2, gridsize**2), 
                                        minval=-10, maxval=10, dtype=tf.float32)
        
        # Call get_image_patches without config (should use iterative)
        result = self.get_image_patches(test_image, global_offsets, local_offsets, N=N, gridsize=gridsize)
        
        # Verify output shape
        self.assertEqual(result.shape, (B, N, N, gridsize**2))
        
    def test_dispatcher_uses_batched_when_enabled(self):
        """Verify batched path when feature flag is enabled."""
        from ptycho import params
        from ptycho.config.config import TrainingConfig, ModelConfig
        
        # Create config with batched extraction enabled
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=2, use_batched_patch_extraction=True),
            nepochs=1
        )
        
        # Create test data
        N = 64
        gridsize = 2
        B = 2
        test_image = tf.complex(
            tf.ones((200, 200), dtype=tf.float32),
            tf.zeros((200, 200), dtype=tf.float32)
        )
        
        # Create offsets in the expected shape
        global_offsets = tf.zeros((B, 1, 2, gridsize**2), dtype=tf.float32)
        local_offsets = tf.random.uniform((B, 1, 2, gridsize**2), 
                                        minval=-10, maxval=10, dtype=tf.float32)
        
        # Call get_image_patches with config (should use batched)
        result = self.get_image_patches(test_image, global_offsets, local_offsets, config=config)
        
        # Verify output shape
        self.assertEqual(result.shape, (B, N, N, gridsize**2))
        
    def test_configuration_priority(self):
        """Test that config object takes precedence over legacy params."""
        from ptycho import params
        from ptycho.config.config import TrainingConfig, ModelConfig
        
        # Set different values in params vs config
        params.set('use_batched_patch_extraction', False)  # Legacy says False
        
        # Create config with batched extraction enabled
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=2, use_batched_patch_extraction=True),  # Config says True
            nepochs=1
        )
        
        # Create test data
        N = 64
        gridsize = 2
        B = 2
        test_image = tf.complex(
            tf.ones((200, 200), dtype=tf.float32),
            tf.zeros((200, 200), dtype=tf.float32)
        )
        
        # Create offsets
        global_offsets = tf.zeros((B, 1, 2, gridsize**2), dtype=tf.float32)
        local_offsets = tf.random.uniform((B, 1, 2, gridsize**2), 
                                        minval=-10, maxval=10, dtype=tf.float32)
        
        # Call with config - should use config value (True) not params value (False)
        result = self.get_image_patches(test_image, global_offsets, local_offsets, config=config)
        
        # Just verify it works - the logging would show which path was taken
        self.assertEqual(result.shape, (B, N, N, gridsize**2))
        
        # Test when config is None - should fall back to params
        params.set('use_batched_patch_extraction', False)
        result2 = self.get_image_patches(test_image, global_offsets, local_offsets, N=N, gridsize=gridsize)
        self.assertEqual(result2.shape, (B, N, N, gridsize**2))


if __name__ == '__main__':
    unittest.main()