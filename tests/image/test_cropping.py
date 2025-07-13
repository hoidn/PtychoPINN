"""
Tests for the cropping module, focusing on the align_for_evaluation function.
"""

import unittest
import numpy as np
from ptycho.image.cropping import align_for_evaluation, _center_crop


class TestCroppingAlignment(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with known input sizes."""
        # Create test reconstruction (64x64)
        self.reconstruction = np.random.random((64, 64)) + 1j * np.random.random((64, 64))
        
        # Create larger ground truth (200x200)
        self.ground_truth = np.random.random((200, 200)) + 1j * np.random.random((200, 200))
        
        # Create scan coordinates that would produce a 64x64 reconstruction
        # Place scan coords in the center region of the ground truth
        x_coords = np.linspace(80, 120, 20)  # 40 pixel span
        y_coords = np.linspace(80, 120, 20)  # 40 pixel span
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.scan_coords_yx = np.column_stack([yy.ravel(), xx.ravel()])
        
        # Stitch patch size (M parameter)
        self.stitch_patch_size = 20
    
    def test_align_for_evaluation_shapes(self):
        """Test that align_for_evaluation returns identically shaped arrays."""
        aligned_recon, aligned_gt = align_for_evaluation(
            self.reconstruction,
            self.ground_truth, 
            self.scan_coords_yx,
            self.stitch_patch_size
        )
        
        # Should return identical shapes
        self.assertEqual(aligned_recon.shape, aligned_gt.shape)
        
        # Should be 2D arrays
        self.assertEqual(len(aligned_recon.shape), 2)
        self.assertEqual(len(aligned_gt.shape), 2)
        
        # Should preserve complex dtype
        self.assertTrue(np.iscomplexobj(aligned_recon))
        self.assertTrue(np.iscomplexobj(aligned_gt))
    
    def test_align_for_evaluation_coordinates_format(self):
        """Test that the function validates coordinate format."""
        # Wrong coordinate format (should be n_positions x 2)
        bad_coords = np.random.random((10, 3))
        
        with self.assertRaises(ValueError):
            align_for_evaluation(
                self.reconstruction,
                self.ground_truth,
                bad_coords,
                self.stitch_patch_size
            )
    
    def test_center_crop_exact_size(self):
        """Test center crop helper function."""
        # Test image that needs cropping
        img = np.random.random((100, 100))
        target_h, target_w = 50, 60
        
        cropped = _center_crop(img, target_h, target_w)
        
        # Check output shape
        self.assertEqual(cropped.shape, (target_h, target_w))
        
        # Test image that doesn't need cropping
        exact_img = np.random.random((50, 60))
        exact_cropped = _center_crop(exact_img, target_h, target_w)
        
        # Should return the same array
        np.testing.assert_array_equal(exact_img, exact_cropped)
    
    def test_align_for_evaluation_with_squeeze(self):
        """Test that the function handles extra dimensions correctly."""
        # Add extra dimensions to inputs
        reconstruction_3d = self.reconstruction[None, ..., None]  # (1, 64, 64, 1)
        ground_truth_3d = self.ground_truth[..., None]  # (200, 200, 1)
        
        aligned_recon, aligned_gt = align_for_evaluation(
            reconstruction_3d,
            ground_truth_3d,
            self.scan_coords_yx,
            self.stitch_patch_size
        )
        
        # Should still return 2D arrays with identical shapes
        self.assertEqual(len(aligned_recon.shape), 2)
        self.assertEqual(len(aligned_gt.shape), 2)
        self.assertEqual(aligned_recon.shape, aligned_gt.shape)
    
    def test_align_for_evaluation_bounding_box(self):
        """Test that the bounding box calculation is reasonable."""
        aligned_recon, aligned_gt = align_for_evaluation(
            self.reconstruction,
            self.ground_truth,
            self.scan_coords_yx,
            self.stitch_patch_size
        )
        
        # The aligned ground truth should be smaller than the original
        self.assertLess(aligned_gt.shape[0], self.ground_truth.shape[0])
        self.assertLess(aligned_gt.shape[1], self.ground_truth.shape[1])
        
        # But should be in a reasonable size range given our scan coordinates
        # (scan coords span ~40 pixels, plus effective_radius of 10)
        expected_size_range = (40, 80)  # Reasonable range
        self.assertGreater(aligned_gt.shape[0], expected_size_range[0])
        self.assertLess(aligned_gt.shape[0], expected_size_range[1])


if __name__ == '__main__':
    unittest.main()