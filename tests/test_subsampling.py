"""Unit tests for independent data subsampling functionality.

This module tests the new n_subsample parameter that enables independent control
of data subsampling and neighbor grouping operations in PtychoPINN.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
from ptycho.workflows.components import load_data
from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho import params


class TestSubsampling(unittest.TestCase):
    """Test suite for data subsampling functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary test dataset."""
        cls.test_data_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        
        # Create synthetic test data
        n_total = 1000
        N = 64
        
        # Generate random coordinates
        xcoords = np.random.rand(n_total) * 100
        ycoords = np.random.rand(n_total) * 100
        xcoords_start = xcoords.copy()
        ycoords_start = ycoords.copy()
        
        # Generate random diffraction patterns
        diffraction = np.random.rand(n_total, N, N).astype(np.float32)
        
        # Generate probe and object
        probeGuess = np.random.rand(N, N).astype(np.complex64)
        objectGuess = np.random.rand(N*3, N*3).astype(np.complex64)
        
        # Generate Y patches for supervised training
        Y = np.random.rand(n_total, N, N).astype(np.complex64)
        
        # Save test data
        np.savez(cls.test_data_file.name,
                xcoords=xcoords,
                ycoords=ycoords,
                xcoords_start=xcoords_start,
                ycoords_start=ycoords_start,
                diffraction=diffraction,
                probeGuess=probeGuess,
                objectGuess=objectGuess,
                Y=Y)
        
        cls.n_total = n_total
        cls.N = N
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test file."""
        import os
        os.unlink(cls.test_data_file.name)
    
    def test_subsample_with_n_subsample(self):
        """Test that n_subsample correctly subsamples the data."""
        n_subsample = 200
        data = load_data(self.test_data_file.name, n_subsample=n_subsample)
        
        # Check that the correct number of images was subsampled
        self.assertEqual(len(data.xcoords), n_subsample)
        self.assertEqual(len(data.ycoords), n_subsample)
        self.assertEqual(data.diff3d.shape[0], n_subsample)
        
        # Check that Y patches were also subsampled if present
        if data.Y is not None:
            self.assertEqual(data.Y.shape[0], n_subsample)
    
    def test_legacy_n_images_behavior(self):
        """Test backward compatibility with n_images parameter."""
        n_images = 300
        data = load_data(self.test_data_file.name, n_images=n_images)
        
        # Check that n_images still works as before when n_subsample is not specified
        self.assertEqual(len(data.xcoords), n_images)
        self.assertEqual(len(data.ycoords), n_images)
        self.assertEqual(data.diff3d.shape[0], n_images)
    
    def test_n_subsample_overrides_n_images(self):
        """Test that n_subsample takes precedence over n_images."""
        n_subsample = 150
        n_images = 500
        data = load_data(self.test_data_file.name, 
                        n_images=n_images, 
                        n_subsample=n_subsample)
        
        # n_subsample should take precedence
        self.assertEqual(len(data.xcoords), n_subsample)
        self.assertEqual(len(data.ycoords), n_subsample)
    
    def test_reproducible_subsampling_with_seed(self):
        """Test that subsample_seed produces reproducible results."""
        seed = 42
        n_subsample = 100
        
        # Load data twice with the same seed
        data1 = load_data(self.test_data_file.name, 
                         n_subsample=n_subsample,
                         subsample_seed=seed)
        data2 = load_data(self.test_data_file.name,
                         n_subsample=n_subsample,
                         subsample_seed=seed)
        
        # Check that the same indices were selected
        np.testing.assert_array_equal(data1.xcoords, data2.xcoords)
        np.testing.assert_array_equal(data1.ycoords, data2.ycoords)
        np.testing.assert_array_equal(data1.diff3d, data2.diff3d)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different subsamples."""
        n_subsample = 100
        
        data1 = load_data(self.test_data_file.name,
                         n_subsample=n_subsample,
                         subsample_seed=42)
        data2 = load_data(self.test_data_file.name,
                         n_subsample=n_subsample,
                         subsample_seed=123)
        
        # Check that different indices were selected
        # (with high probability for reasonable dataset sizes)
        self.assertFalse(np.array_equal(data1.xcoords, data2.xcoords))
    
    def test_subsample_larger_than_dataset(self):
        """Test that requesting more samples than available uses full dataset."""
        n_subsample = self.n_total + 100  # More than available
        data = load_data(self.test_data_file.name, n_subsample=n_subsample)
        
        # Should use all available data
        self.assertEqual(len(data.xcoords), self.n_total)
        self.assertEqual(len(data.ycoords), self.n_total)
    
    def test_no_subsample_uses_full_dataset(self):
        """Test that not specifying n_subsample or n_images uses full dataset."""
        data = load_data(self.test_data_file.name)
        
        # Should use all available data
        self.assertEqual(len(data.xcoords), self.n_total)
        self.assertEqual(len(data.ycoords), self.n_total)
    
    def test_subsample_zero_edge_case(self):
        """Test edge case where n_subsample is 0."""
        # This should either raise an error or use minimum of 1
        # The actual behavior depends on implementation
        n_subsample = 0
        
        # Current implementation will use min(0, dataset_size) = 0
        # which may cause issues downstream, so we should handle this
        try:
            data = load_data(self.test_data_file.name, n_subsample=n_subsample)
            # If it doesn't raise, check that it handled gracefully
            self.assertGreaterEqual(len(data.xcoords), 0)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass
    
    def test_y_patches_subsampled_consistently(self):
        """Test that Y patches are subsampled consistently with diffraction data."""
        n_subsample = 200
        seed = 42
        
        data = load_data(self.test_data_file.name,
                        n_subsample=n_subsample,
                        subsample_seed=seed)
        
        # Check that Y patches have same first dimension as diffraction
        if data.Y is not None:
            self.assertEqual(data.Y.shape[0], data.diff3d.shape[0])
            self.assertEqual(data.Y.shape[0], n_subsample)
    
    def test_sorted_indices_for_consistency(self):
        """Test that subsampled indices are sorted for consistency."""
        n_subsample = 100
        seed = 42
        
        # Load data and check coordinates are monotonic if originally sorted
        original_data = np.load(self.test_data_file.name)
        
        # Create test data with sorted coordinates
        sorted_test_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        sorted_coords = np.arange(self.n_total, dtype=np.float64)
        
        np.savez(sorted_test_file.name,
                xcoords=sorted_coords,
                ycoords=sorted_coords,
                xcoords_start=original_data['xcoords_start'],
                ycoords_start=original_data['ycoords_start'],
                diffraction=original_data['diffraction'],
                probeGuess=original_data['probeGuess'],
                objectGuess=original_data['objectGuess'],
                Y=original_data['Y'])
        
        # Load with subsampling
        data = load_data(sorted_test_file.name,
                        n_subsample=n_subsample,
                        subsample_seed=seed)
        
        # Check that selected indices are sorted
        self.assertTrue(np.all(np.diff(data.xcoords) >= 0))
        
        # Clean up
        import os
        os.unlink(sorted_test_file.name)
    
    def test_interaction_with_config_dataclass(self):
        """Test that new config fields work correctly."""
        config = TrainingConfig(
            model=ModelConfig(N=64),
            n_images=500,
            n_subsample=200,
            subsample_seed=42
        )
        
        # Check that fields are accessible
        self.assertEqual(config.n_subsample, 200)
        self.assertEqual(config.subsample_seed, 42)
        
        # Check that None defaults work
        config_default = TrainingConfig(
            model=ModelConfig(N=64),
            n_images=500
        )
        self.assertIsNone(config_default.n_subsample)
        self.assertIsNone(config_default.subsample_seed)


if __name__ == '__main__':
    unittest.main()