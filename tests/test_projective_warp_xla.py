"""Test cases for projective_warp_xla module."""

import unittest
import tensorflow as tf
import numpy as np
from ptycho.projective_warp_xla import (
    projective_warp_xla,
    projective_warp_xla_jit,
    translate_xla,
    tfa_params_to_3x3
)


class TestProjectiveWarpXLA(unittest.TestCase):
    """Test suite for XLA-compatible projective warp implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test images
        self.batch_size = 2
        self.height = 64
        self.width = 64
        self.channels = 1
        
        # Create test images with a simple pattern
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        pattern = np.sin(5 * xx) * np.cos(5 * yy)
        
        self.test_image_f32 = tf.constant(
            pattern[np.newaxis, :, :, np.newaxis],
            dtype=tf.float32
        )
        self.test_image_f32 = tf.tile(self.test_image_f32, [self.batch_size, 1, 1, 1])
        
        self.test_image_f64 = tf.cast(self.test_image_f32, tf.float64)
        
        # Create identity transform
        self.identity_transform = tf.constant([
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]]
        ], dtype=tf.float32)
        self.identity_transform = tf.tile(self.identity_transform, [self.batch_size, 1, 1])
        
        # Create translation transform
        self.translation_transform = tf.constant([
            [[1.0, 0.0, 5.0],
             [0.0, 1.0, 5.0],
             [0.0, 0.0, 1.0]]
        ], dtype=tf.float32)
        self.translation_transform = tf.tile(self.translation_transform, [self.batch_size, 1, 1])
    
    def test_float32_dtype(self):
        """Test that float32 inputs work correctly."""
        # This test would have passed even with the bug
        result = projective_warp_xla(
            self.test_image_f32,
            self.identity_transform,
            interpolation='bilinear',
            fill_mode='zeros'
        )
        self.assertEqual(result.dtype, tf.float32)
        self.assertEqual(result.shape, self.test_image_f32.shape)
        
        # Check that identity transform preserves the image
        np.testing.assert_allclose(
            result.numpy(),
            self.test_image_f32.numpy(),
            rtol=1e-5,
            atol=1e-5
        )
    
    def test_float64_dtype(self):
        """Test that float64 inputs work correctly.
        
        This test would have caught the original bug where float64 images
        caused a dtype mismatch error with float32 grid computations.
        """
        # Convert transform to float64 to match image dtype
        transform_f64 = tf.cast(self.identity_transform, tf.float64)
        
        # This call would have failed with the original bug
        result = projective_warp_xla(
            self.test_image_f64,
            transform_f64,
            interpolation='bilinear',
            fill_mode='zeros'
        )
        
        self.assertEqual(result.dtype, tf.float64)
        self.assertEqual(result.shape, self.test_image_f64.shape)
        
        # Check that identity transform preserves the image
        np.testing.assert_allclose(
            result.numpy(),
            self.test_image_f64.numpy(),
            rtol=1e-10,
            atol=1e-10
        )
    
    def test_float64_with_translation(self):
        """Test float64 with actual translation to ensure interpolation works."""
        transform_f64 = tf.cast(self.translation_transform, tf.float64)
        
        result = projective_warp_xla(
            self.test_image_f64,
            transform_f64,
            interpolation='bilinear',
            fill_mode='zeros'
        )
        
        self.assertEqual(result.dtype, tf.float64)
        self.assertEqual(result.shape, self.test_image_f64.shape)
        
        # Check that translation moved the image
        self.assertFalse(np.allclose(result.numpy(), self.test_image_f64.numpy()))
    
    def test_complex64_dtype(self):
        """Test that complex64 inputs work correctly."""
        # Create complex image
        complex_image = tf.complex(self.test_image_f32, self.test_image_f32 * 0.5)
        
        # Create translations for translate_xla
        translations = tf.constant([[5.0, 5.0]], dtype=tf.float32)
        translations = tf.tile(translations, [self.batch_size, 1])
        
        result = translate_xla(
            complex_image,
            translations,
            interpolation='bilinear',
            use_jit=False
        )
        
        self.assertEqual(result.dtype, tf.complex64)
        self.assertEqual(result.shape, complex_image.shape)
        
        # Check that real and imaginary parts are processed correctly
        real_result = tf.math.real(result)
        imag_result = tf.math.imag(result)
        
        # Both parts should be translated
        self.assertFalse(np.allclose(real_result.numpy(), tf.math.real(complex_image).numpy()))
        self.assertFalse(np.allclose(imag_result.numpy(), tf.math.imag(complex_image).numpy()))
    
    def test_complex128_dtype(self):
        """Test that complex128 inputs work correctly."""
        # Create complex128 image
        complex_image_f64 = tf.complex(self.test_image_f64, self.test_image_f64 * 0.5)
        
        # Create translations for translate_xla
        translations = tf.constant([[5.0, 5.0]], dtype=tf.float64)
        translations = tf.tile(translations, [self.batch_size, 1])
        
        result = translate_xla(
            complex_image_f64,
            translations,
            interpolation='bilinear',
            use_jit=False
        )
        
        self.assertEqual(result.dtype, tf.complex128)
        self.assertEqual(result.shape, complex_image_f64.shape)
    
    def test_jit_compilation_float32(self):
        """Test that JIT compilation works with float32."""
        result = projective_warp_xla_jit(
            self.test_image_f32,
            self.identity_transform,
            interpolation='bilinear',
            fill_mode='zeros'
        )
        
        self.assertEqual(result.dtype, tf.float32)
        self.assertEqual(result.shape, self.test_image_f32.shape)
        
        # Compare with non-JIT version
        result_no_jit = projective_warp_xla(
            self.test_image_f32,
            self.identity_transform,
            interpolation='bilinear',
            fill_mode='zeros'
        )
        
        np.testing.assert_allclose(
            result.numpy(),
            result_no_jit.numpy(),
            rtol=1e-5,
            atol=1e-5
        )
    
    def test_jit_compilation_float64(self):
        """Test that JIT compilation works with float64.
        
        This test specifically checks that the dtype fixes allow
        JIT compilation to work with float64 inputs.
        """
        transform_f64 = tf.cast(self.identity_transform, tf.float64)
        
        # This would have failed with the original bug
        result = projective_warp_xla_jit(
            self.test_image_f64,
            transform_f64,
            interpolation='bilinear',
            fill_mode='zeros'
        )
        
        self.assertEqual(result.dtype, tf.float64)
        self.assertEqual(result.shape, self.test_image_f64.shape)
        
        # Compare with non-JIT version
        result_no_jit = projective_warp_xla(
            self.test_image_f64,
            transform_f64,
            interpolation='bilinear',
            fill_mode='zeros'
        )
        
        np.testing.assert_allclose(
            result.numpy(),
            result_no_jit.numpy(),
            rtol=1e-10,
            atol=1e-10
        )
    
    def test_mixed_precision_translation(self):
        """Test translation with mixed precision inputs."""
        # Use float32 translations with float64 image
        translations_f32 = tf.constant([[5.0, 5.0]], dtype=tf.float32)
        translations_f32 = tf.tile(translations_f32, [self.batch_size, 1])
        
        result = translate_xla(
            self.test_image_f64,
            translations_f32,
            interpolation='bilinear',
            use_jit=True
        )
        
        # Result should maintain image dtype
        self.assertEqual(result.dtype, tf.float64)
        self.assertEqual(result.shape, self.test_image_f64.shape)
    
    def test_tfa_params_conversion(self):
        """Test TFA parameter conversion."""
        # Create TFA-style parameters
        params = tf.constant([
            [1.0, 0.0, 5.0, 0.0, 1.0, 5.0, 0.0, 0.0],
            [1.0, 0.0, -5.0, 0.0, 1.0, -5.0, 0.0, 0.0]
        ], dtype=tf.float32)
        
        matrices = tfa_params_to_3x3(params)
        
        self.assertEqual(matrices.shape, (2, 3, 3))
        self.assertEqual(matrices.dtype, tf.float32)
        
        # Check that the conversion is correct
        expected = tf.constant([
            [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, -5.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]]
        ], dtype=tf.float32)
        
        np.testing.assert_allclose(matrices.numpy(), expected.numpy())
    
    def test_interpolation_modes(self):
        """Test different interpolation modes."""
        for interpolation in ['nearest', 'bilinear']:
            result = projective_warp_xla(
                self.test_image_f32,
                self.translation_transform,
                interpolation=interpolation,
                fill_mode='zeros'
            )
            self.assertEqual(result.dtype, tf.float32)
            self.assertEqual(result.shape, self.test_image_f32.shape)
    
    def test_fill_modes(self):
        """Test different fill modes."""
        for fill_mode in ['zeros', 'edge']:
            result = projective_warp_xla(
                self.test_image_f32,
                self.translation_transform,
                interpolation='bilinear',
                fill_mode=fill_mode
            )
            self.assertEqual(result.dtype, tf.float32)
            self.assertEqual(result.shape, self.test_image_f32.shape)
    
    def test_batch_processing(self):
        """Test that batched images are processed correctly."""
        # Create batch with different transforms
        transforms = tf.constant([
            [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, -5.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]]
        ], dtype=tf.float32)
        
        result = projective_warp_xla(
            self.test_image_f32,
            transforms,
            interpolation='bilinear',
            fill_mode='zeros'
        )
        
        self.assertEqual(result.shape[0], 2)
        
        # Check that the two results are different (different transforms)
        self.assertFalse(np.allclose(result[0].numpy(), result[1].numpy()))


if __name__ == '__main__':
    unittest.main()