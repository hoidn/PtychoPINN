#!/usr/bin/env python
"""
Test automatic K choose C oversampling functionality.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptycho.raw_data import RawData
from ptycho import params


class TestAutomaticOversampling(unittest.TestCase):
    """Test automatic K choose C oversampling when requesting more groups than points."""
    
    def setUp(self):
        """Create a small synthetic dataset for testing."""
        # Create a small dataset with 100 points
        np.random.seed(42)
        n_points = 100
        N = 64  # Diffraction pattern size
        
        # Generate random coordinates
        self.xcoords = np.random.uniform(0, 200, n_points)
        self.ycoords = np.random.uniform(0, 200, n_points)
        
        # Generate synthetic diffraction patterns
        self.diffraction = np.random.rand(n_points, N, N).astype(np.float32)
        
        # Generate synthetic probe and object
        self.probeGuess = np.ones((N, N), dtype=np.complex64)
        self.objectGuess = np.ones((232, 232), dtype=np.complex64)
        
        # Set gridsize for testing
        params.cfg['gridsize'] = 2  # C = 4
        
    def test_standard_sampling_no_oversampling(self):
        """Test that requesting fewer groups than points works normally."""
        # Create RawData instance
        raw_data = RawData(
            xcoords=self.xcoords,
            ycoords=self.ycoords,
            xcoords_start=self.xcoords,
            ycoords_start=self.ycoords,
            diff3d=self.diffraction,
            probeGuess=self.probeGuess,
            scan_index=np.arange(len(self.xcoords)),
            objectGuess=self.objectGuess
        )
        
        # Request 50 groups (less than 100 points available)
        nsamples = 50
        K = 4
        N = 64
        
        dataset = raw_data.generate_grouped_data(N, K=K, nsamples=nsamples, seed=42)
        
        # Check that we got the requested number of groups
        self.assertEqual(dataset['diffraction'].shape[0], nsamples)
        self.assertEqual(dataset['nn_indices'].shape[0], nsamples)
        self.assertEqual(dataset['nn_indices'].shape[1], 4)  # gridsize² = 4
        
    def test_automatic_oversampling_triggers(self):
        """Test that oversampling automatically triggers when requesting more groups than points."""
        # Create RawData instance
        raw_data = RawData(
            xcoords=self.xcoords,
            ycoords=self.ycoords,
            xcoords_start=self.xcoords,
            ycoords_start=self.ycoords,
            diff3d=self.diffraction,
            probeGuess=self.probeGuess,
            scan_index=np.arange(len(self.xcoords)),
            objectGuess=self.objectGuess
        )
        
        # Request 200 groups (more than 100 points available)
        nsamples = 200
        K = 7  # Higher K for more combinations
        N = 64
        
        dataset = raw_data.generate_grouped_data(N, K=K, nsamples=nsamples, seed=42)
        
        # Check that we got the requested number of groups through oversampling
        self.assertEqual(dataset['diffraction'].shape[0], nsamples)
        self.assertEqual(dataset['nn_indices'].shape[0], nsamples)
        self.assertEqual(dataset['nn_indices'].shape[1], 4)  # gridsize² = 4
        
        print(f"Successfully generated {nsamples} groups from {len(self.xcoords)} points using K={K}")
        
    def test_oversampling_with_different_k_values(self):
        """Test oversampling with different K values."""
        raw_data = RawData(
            xcoords=self.xcoords,
            ycoords=self.ycoords,
            xcoords_start=self.xcoords,
            ycoords_start=self.ycoords,
            diff3d=self.diffraction,
            probeGuess=self.probeGuess,
            scan_index=np.arange(len(self.xcoords)),
            objectGuess=self.objectGuess
        )
        
        nsamples = 150
        N = 64
        
        # Test with different K values
        for K in [5, 6, 7, 8]:
            with self.subTest(K=K):
                dataset = raw_data.generate_grouped_data(N, K=K, nsamples=nsamples, seed=42)
                
                # Should always get requested number of groups
                self.assertEqual(dataset['diffraction'].shape[0], nsamples)
                
                # Higher K should allow more diverse combinations
                unique_indices = len(np.unique(dataset['nn_indices']))
                print(f"K={K}: Generated {nsamples} groups using {unique_indices} unique indices")
                
    def test_gridsize_1_no_oversampling(self):
        """Test that gridsize=1 doesn't trigger oversampling logic."""
        params.cfg['gridsize'] = 1  # C = 1
        
        raw_data = RawData(
            xcoords=self.xcoords,
            ycoords=self.ycoords,
            xcoords_start=self.xcoords,
            ycoords_start=self.ycoords,
            diff3d=self.diffraction,
            probeGuess=self.probeGuess,
            scan_index=np.arange(len(self.xcoords)),
            objectGuess=self.objectGuess
        )
        
        # Request more samples than points
        nsamples = 150
        K = 7
        N = 64
        
        dataset = raw_data.generate_grouped_data(N, K=K, nsamples=nsamples, seed=42)
        
        # For gridsize=1, no oversampling - capped at available points
        expected_samples = min(nsamples, len(self.xcoords))
        self.assertEqual(dataset['diffraction'].shape[0], expected_samples)
        self.assertEqual(dataset['nn_indices'].shape[0], expected_samples)
        self.assertEqual(dataset['nn_indices'].shape[1], 1)  # gridsize² = 1
        
    def test_reproducibility_with_seed(self):
        """Test that oversampling is reproducible with the same seed."""
        raw_data = RawData(
            xcoords=self.xcoords,
            ycoords=self.ycoords,
            xcoords_start=self.xcoords,
            ycoords_start=self.ycoords,
            diff3d=self.diffraction,
            probeGuess=self.probeGuess,
            scan_index=np.arange(len(self.xcoords)),
            objectGuess=self.objectGuess
        )
        
        nsamples = 200
        K = 7
        N = 64
        seed = 123
        
        # Generate two datasets with same seed
        dataset1 = raw_data.generate_grouped_data(N, K=K, nsamples=nsamples, seed=seed)
        dataset2 = raw_data.generate_grouped_data(N, K=K, nsamples=nsamples, seed=seed)
        
        # Should be identical
        np.testing.assert_array_equal(dataset1['nn_indices'], dataset2['nn_indices'])
        
        # Generate with different seed
        dataset3 = raw_data.generate_grouped_data(N, K=K, nsamples=nsamples, seed=456)
        
        # Should be different
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(dataset1['nn_indices'], dataset3['nn_indices'])
    
    def tearDown(self):
        """Reset gridsize to default."""
        params.cfg['gridsize'] = 1


if __name__ == '__main__':
    unittest.main()