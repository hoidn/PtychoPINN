"""
Unit tests for the image registration module.

Tests cover offset detection, shift application, cropping, and edge cases
for both real and complex-valued images.
"""

import unittest
import numpy as np
import numpy.testing as npt
from ptycho.image.registration import (
    find_translation_offset,
    apply_shift_and_crop,
    register_and_align
)


class TestRegistration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with various image types."""
        # Create test images
        np.random.seed(42)  # For reproducible tests
        
        # Real-valued test image with distinct features
        self.real_ref = np.zeros((64, 64))
        self.real_ref[20:40, 20:40] = 1.0  # square
        self.real_ref[10:15, 50:55] = 0.5  # small rectangle
        
        # Complex-valued test image
        amplitude = self.real_ref.copy()
        phase = np.random.rand(64, 64) * 2 * np.pi
        self.complex_ref = amplitude * np.exp(1j * phase)
        
        # Random noise image for testing edge cases
        self.noise_ref = np.random.rand(64, 64)
        
    def test_find_offset_known_shift_real(self):
        """Test offset detection with known shifts on real images."""
        test_cases = [
            (5, 3),   # down 5, right 3
            (-3, 7),  # up 3, right 7
            (0, -5),  # no vertical, left 5
            (8, 0),   # down 8, no horizontal
            (0, 0),   # no shift
        ]
        
        for dy_true, dx_true in test_cases:
            with self.subTest(shift=(dy_true, dx_true)):
                # Create shifted image
                shifted = np.roll(np.roll(self.real_ref, dy_true, axis=0), dx_true, axis=1)
                
                # Detect offset
                dy_detected, dx_detected = find_translation_offset(shifted, self.real_ref)
                
                # Should detect the applied shift
                self.assertEqual(dy_detected, dy_true, 
                               f"Y offset detection failed: expected {dy_true}, got {dy_detected}")
                self.assertEqual(dx_detected, dx_true,
                               f"X offset detection failed: expected {dx_true}, got {dx_detected}")
    
    def test_find_offset_known_shift_complex(self):
        """Test offset detection with known shifts on complex images."""
        dy_true, dx_true = 7, -4
        
        # Create shifted complex image
        shifted = np.roll(np.roll(self.complex_ref, dy_true, axis=0), dx_true, axis=1)
        
        # Detect offset
        dy_detected, dx_detected = find_translation_offset(shifted, self.complex_ref)
        
        # Should detect the applied shift
        self.assertEqual(dy_detected, dy_true)
        self.assertEqual(dx_detected, dx_true)
    
    def test_apply_shift_and_crop_basic(self):
        """Test basic shift application and cropping."""
        offset = (5, -3)  # down 5, left 3
        
        # Apply shift and crop
        shifted_img, cropped_ref = apply_shift_and_crop(
            self.real_ref, self.real_ref, offset
        )
        
        # Check shapes are identical
        self.assertEqual(shifted_img.shape, cropped_ref.shape)
        
        # Check resulting shape is correct (should be reduced by offset amounts)
        expected_h = self.real_ref.shape[0] - abs(offset[0])
        expected_w = self.real_ref.shape[1] - abs(offset[1])
        self.assertEqual(shifted_img.shape, (expected_h, expected_w))
    
    def test_apply_shift_and_crop_zero_offset(self):
        """Test that zero offset doesn't unnecessarily crop images."""
        offset = (0, 0)
        
        shifted_img, cropped_ref = apply_shift_and_crop(
            self.real_ref, self.real_ref, offset
        )
        
        # Should have same shape as original
        self.assertEqual(shifted_img.shape, self.real_ref.shape)
        self.assertEqual(cropped_ref.shape, self.real_ref.shape)
        
        # Should be identical to original
        npt.assert_array_equal(shifted_img, self.real_ref)
        npt.assert_array_equal(cropped_ref, self.real_ref)
    
    def test_shift_and_crop_preserves_data_type(self):
        """Test that shift and crop preserves complex data types."""
        offset = (3, 2)
        
        shifted_img, cropped_ref = apply_shift_and_crop(
            self.complex_ref, self.complex_ref, offset
        )
        
        # Should preserve complex type
        self.assertTrue(np.iscomplexobj(shifted_img))
        self.assertTrue(np.iscomplexobj(cropped_ref))
        self.assertEqual(shifted_img.dtype, self.complex_ref.dtype)
    
    def test_round_trip_registration(self):
        """Test that applying detected offset correctly aligns images."""
        # Create a shifted version
        true_offset = (6, -4)
        shifted_orig = np.roll(np.roll(self.real_ref, true_offset[0], axis=0), 
                              true_offset[1], axis=1)
        
        # Detect offset
        detected_offset = find_translation_offset(shifted_orig, self.real_ref)
        
        # Apply inverse shift to align
        aligned_img, aligned_ref = apply_shift_and_crop(
            shifted_orig, self.real_ref, detected_offset
        )
        
        # The overlapping regions should now be very similar
        # (accounting for the fact that we lose some border region)
        correlation = np.corrcoef(aligned_img.flatten(), aligned_ref.flatten())[0, 1]
        self.assertGreater(correlation, 0.95, 
                          f"Aligned images should be highly correlated, got {correlation:.3f}")
    
    def test_register_and_align_convenience(self):
        """Test the convenience function that combines detection and alignment."""
        # Create shifted image
        true_offset = (4, 7)
        shifted = np.roll(np.roll(self.real_ref, true_offset[0], axis=0), 
                         true_offset[1], axis=1)
        
        # Use convenience function
        aligned_img, aligned_ref = register_and_align(shifted, self.real_ref)
        
        # Should have identical shapes
        self.assertEqual(aligned_img.shape, aligned_ref.shape)
        
        # Should be well-aligned (high correlation)
        correlation = np.corrcoef(aligned_img.flatten(), aligned_ref.flatten())[0, 1]
        self.assertGreater(correlation, 0.95)
    
    def test_input_validation_2d_requirement(self):
        """Test that functions reject non-2D inputs."""
        img_1d = np.random.rand(64)
        img_3d = np.random.rand(32, 32, 3)
        
        with self.assertRaises(ValueError):
            find_translation_offset(img_1d, self.real_ref)
        
        with self.assertRaises(ValueError):
            find_translation_offset(self.real_ref, img_3d)
            
        with self.assertRaises(ValueError):
            apply_shift_and_crop(img_1d, self.real_ref, (0, 0))
    
    def test_input_validation_shape_matching(self):
        """Test that functions require matching image shapes."""
        wrong_shape = np.random.rand(32, 32)
        
        with self.assertRaises(ValueError):
            find_translation_offset(wrong_shape, self.real_ref)
            
        with self.assertRaises(ValueError):
            apply_shift_and_crop(wrong_shape, self.real_ref, (0, 0))
    
    def test_input_validation_excessive_offset(self):
        """Test that excessively large offsets are rejected."""
        large_offset = (70, 50)  # Larger than 64x64 image
        
        with self.assertRaises(ValueError):
            apply_shift_and_crop(self.real_ref, self.real_ref, large_offset)
    
    def test_edge_case_single_pixel_shift(self):
        """Test detection of single-pixel shifts."""
        offset = (1, 1)
        shifted = np.roll(np.roll(self.real_ref, offset[0], axis=0), offset[1], axis=1)
        
        detected = find_translation_offset(shifted, self.real_ref)
        self.assertEqual(detected, offset)
    
    def test_edge_case_maximum_shift(self):
        """Test near-maximum shifts (but still valid)."""
        h, w = self.real_ref.shape
        
        # Test shifts that are large but still valid
        test_offsets = [
            (h-2, 0),     # Near-maximum vertical
            (0, w-2),     # Near-maximum horizontal  
            (-(h-2), 0),  # Near-maximum negative vertical
            (0, -(w-2)),  # Near-maximum negative horizontal
        ]
        
        for offset in test_offsets:
            with self.subTest(offset=offset):
                shifted_img, cropped_ref = apply_shift_and_crop(
                    self.real_ref, self.real_ref, offset
                )
                
                # Should not crash and should produce valid shapes
                self.assertEqual(shifted_img.shape, cropped_ref.shape)
                self.assertGreater(shifted_img.size, 0)  # Non-empty result
    
    def test_noise_robustness(self):
        """Test that registration works reasonably with noisy images."""
        # Add noise to reference
        noise_level = 0.1
        noisy_ref = self.real_ref + noise_level * np.random.rand(*self.real_ref.shape)
        
        # Create shifted version with same noise
        true_offset = (3, -2)
        shifted_noisy = np.roll(np.roll(noisy_ref, true_offset[0], axis=0), 
                               true_offset[1], axis=1)
        
        # Should still detect offset reasonably well
        detected_offset = find_translation_offset(shifted_noisy, noisy_ref)
        
        # Allow some tolerance for noisy data
        dy_error = abs(detected_offset[0] - true_offset[0])
        dx_error = abs(detected_offset[1] - true_offset[1])
        
        self.assertLessEqual(dy_error, 1, "Y offset should be within 1 pixel for noisy data")
        self.assertLessEqual(dx_error, 1, "X offset should be within 1 pixel for noisy data")
    
    def test_different_image_content(self):
        """Test registration between different (but related) image content."""
        # Create two different but related images
        img1 = self.real_ref.copy()
        img2 = self.real_ref * 0.8 + 0.1  # scaled and offset version
        
        # Apply known shift to img2
        true_offset = (5, -3)
        shifted_img2 = np.roll(np.roll(img2, true_offset[0], axis=0), 
                              true_offset[1], axis=1)
        
        # Should still detect the shift
        detected_offset = find_translation_offset(shifted_img2, img1)
        
        # May not be perfect due to content differences, but should be close
        dy_error = abs(detected_offset[0] - true_offset[0])
        dx_error = abs(detected_offset[1] - true_offset[1])
        
        self.assertLessEqual(dy_error, 2, "Should detect offset within 2 pixels for related content")
        self.assertLessEqual(dx_error, 2, "Should detect offset within 2 pixels for related content")


if __name__ == '__main__':
    unittest.main()