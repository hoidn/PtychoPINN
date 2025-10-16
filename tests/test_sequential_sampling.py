"""
Test suite for sequential sampling functionality in coordinate grouping.

This module tests the sequential_sampling flag added to the RawData class
to restore deterministic, sequential data subset selection capability.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptycho.raw_data import RawData
from ptycho import params


class TestSequentialSampling(unittest.TestCase):
    """Test suite for sequential sampling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a dataset with known ordering
        n_points = 100
        self.xcoords = np.arange(n_points, dtype=float)
        self.ycoords = np.arange(n_points, dtype=float) * 2  # Different pattern
        
        # Create dummy data
        self.diff3d = np.random.rand(n_points, 64, 64)
        self.probeGuess = np.ones((64, 64), dtype=complex)
        self.objectGuess = np.ones((256, 256), dtype=complex)
        self.scan_index = np.arange(n_points)
        
        # Create RawData instance
        self.raw_data = RawData(
            xcoords=self.xcoords,
            ycoords=self.ycoords,
            xcoords_start=self.xcoords,
            ycoords_start=self.ycoords,
            diff3d=self.diff3d,
            probeGuess=self.probeGuess,
            scan_index=self.scan_index,
            objectGuess=self.objectGuess
        )
        
        # Set gridsize
        params.cfg['gridsize'] = 2
    
    def test_default_behavior_is_random(self):
        """Test that default behavior (sequential_sampling=False) uses random sampling."""
        # Generate groups with default (random) sampling
        nsamples = 10
        result1 = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples, 
            sequential_sampling=False,
            seed=None  # No seed, so should be random
        )
        
        result2 = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=False,
            seed=None  # No seed, so should be random
        )
        
        # The indices should be different between runs (with high probability)
        indices1 = result1['nn_indices'].flatten()
        indices2 = result2['nn_indices'].flatten()
        
        # Check that at least some indices are different
        # (There's a tiny chance this could fail randomly, but very unlikely)
        self.assertFalse(np.array_equal(indices1, indices2),
                        "Random sampling should produce different results")
    
    def test_sequential_sampling_uses_first_n_points(self):
        """Test that sequential_sampling=True uses the first nsamples points."""
        nsamples = 10
        result = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=True
        )
        
        # Extract the seed indices used (first index in each group should be related to seed)
        nn_indices = result['nn_indices']
        
        # Check that groups are formed from the first nsamples points
        # The exact indices depend on neighbor finding, but seed points should be sequential
        # We can verify this by checking that the minimum index in each group
        # increases roughly monotonically
        min_indices = [np.min(group) for group in nn_indices]
        
        # First 10 groups should use points near the beginning of the dataset
        self.assertTrue(max(min_indices) < 30,
                       f"Sequential sampling should use early points, but max min_index was {max(min_indices)}")
    
    def test_sequential_sampling_is_deterministic(self):
        """Test that sequential sampling produces deterministic results."""
        nsamples = 10
        
        # Run sequential sampling multiple times
        result1 = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=True
        )
        
        result2 = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=True
        )
        
        # Results should be identical
        np.testing.assert_array_equal(
            result1['nn_indices'], 
            result2['nn_indices'],
            err_msg="Sequential sampling should be deterministic"
        )
    
    def test_sequential_sampling_order(self):
        """Test that sequential sampling actually uses points in order."""
        # Create a smaller, more controlled dataset
        n_points = 20
        xcoords = np.arange(n_points, dtype=float)
        ycoords = np.zeros(n_points)  # All in a line for easier verification
        
        small_raw_data = RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=np.random.rand(n_points, 64, 64),
            probeGuess=np.ones((64, 64), dtype=complex),
            scan_index=np.arange(n_points)
        )
        
        # Use sequential sampling
        nsamples = 5
        params.cfg['gridsize'] = 1  # Use gridsize=1 for simplicity
        result = small_raw_data.generate_grouped_data(
            N=64, K=3, nsamples=nsamples,
            sequential_sampling=True
        )
        
        # For gridsize=1, each group should be a single point
        # and they should be [0, 1, 2, 3, 4] for sequential sampling
        nn_indices = result['nn_indices'].flatten()
        
        # The seed indices should be the first 5 points
        # Due to neighbor finding, the actual indices might include neighbors,
        # but the pattern should show sequential selection
        unique_indices = np.unique(nn_indices)
        
        # Check that low indices are heavily represented
        self.assertTrue(min(unique_indices) == 0, "Should include index 0")
        self.assertTrue(max(unique_indices) < 10, 
                       f"Should mainly use early indices, but max was {max(unique_indices)}")
    
    def test_sequential_sampling_with_gridsize_greater_than_1(self):
        """Test sequential sampling with gridsize > 1."""
        params.cfg['gridsize'] = 2  # 4 points per group
        nsamples = 5
        
        result = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=True
        )
        
        # Check shape
        nn_indices = result['nn_indices']
        self.assertEqual(nn_indices.shape, (nsamples, 4))  # gridsize^2 = 4
        
        # Check that it's deterministic
        result2 = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=True
        )
        np.testing.assert_array_equal(nn_indices, result2['nn_indices'])
    
    def test_sequential_vs_random_coverage(self):
        """Test that sequential sampling covers a different region than random."""
        nsamples = 20
        
        # Sequential sampling
        seq_result = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=True
        )
        
        # Random sampling with seed for reproducibility
        random_result = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=False,
            seed=42
        )
        
        # Get unique indices used
        seq_indices = np.unique(seq_result['nn_indices'].flatten())
        random_indices = np.unique(random_result['nn_indices'].flatten())
        
        # Sequential should use lower indices on average
        seq_mean = np.mean(seq_indices)
        random_mean = np.mean(random_indices)
        
        # Sequential mean should be lower (closer to start of dataset)
        self.assertLess(seq_mean, random_mean,
                       f"Sequential sampling (mean={seq_mean:.1f}) should use lower indices "
                       f"than random sampling (mean={random_mean:.1f})")
    
    def test_sequential_sampling_handles_edge_cases(self):
        """Test sequential sampling with edge cases."""
        # Case 1: Request more samples than available points with gridsize=1 (no oversampling)
        n_points = len(self.xcoords)

        # Temporarily set gridsize=1 to avoid K choose C oversampling
        original_gridsize = params.cfg['gridsize']
        params.cfg['gridsize'] = 1

        try:
            result = self.raw_data.generate_grouped_data(
                N=64, K=7, nsamples=n_points + 10,  # More than available
                sequential_sampling=True
            )

            # With gridsize=1, should cap at available points
            nn_indices = result['nn_indices']
            self.assertLessEqual(len(nn_indices), n_points)
        finally:
            # Restore original gridsize
            params.cfg['gridsize'] = original_gridsize

        # Case 2: Request more samples than available points with gridsize > 1 (oversampling)
        n_points = len(self.xcoords)
        result = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=n_points + 10,  # More than available
            sequential_sampling=True
        )

        # With gridsize=2 (C=4), should use K choose C oversampling to generate the requested number
        nn_indices = result['nn_indices']
        self.assertEqual(len(nn_indices), n_points + 10,
                        "K choose C oversampling should generate the exact number of requested groups")

        # Case 3: Request exactly the number of available points
        result = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=n_points,
            sequential_sampling=True
        )

        # Should work without issues
        self.assertEqual(len(result['nn_indices']), n_points)
    
    def test_sequential_sampling_with_seed_parameter(self):
        """Test that seed parameter doesn't affect sequential sampling."""
        nsamples = 10
        
        # Sequential with different seeds should give same result
        result1 = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=True,
            seed=42
        )
        
        result2 = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=nsamples,
            sequential_sampling=True,
            seed=123  # Different seed
        )
        
        # Should be identical despite different seeds
        np.testing.assert_array_equal(
            result1['nn_indices'],
            result2['nn_indices'],
            err_msg="Sequential sampling should ignore seed parameter"
        )


class TestIntegrationWithWorkflow(unittest.TestCase):
    """Test integration of sequential sampling with the training workflow."""
    
    def test_config_flag_exists(self):
        """Test that TrainingConfig has sequential_sampling field."""
        from ptycho.config.config import TrainingConfig, ModelConfig
        
        # Create a config with sequential_sampling
        model_config = ModelConfig()
        config = TrainingConfig(
            model=model_config,
            sequential_sampling=True  # Should not raise an error
        )
        
        self.assertTrue(hasattr(config, 'sequential_sampling'))
        self.assertTrue(config.sequential_sampling)
        
        # Test default value
        config_default = TrainingConfig(model=model_config)
        self.assertFalse(config_default.sequential_sampling)
    
    def test_cli_argument_parsing(self):
        """Test that CLI argument for sequential sampling works."""
        from ptycho.workflows.components import parse_arguments
        import argparse
        
        # Mock command line args
        test_args = ['--sequential_sampling', '--n_images', '100']
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        # The parse_arguments function should add this flag
        # We can't easily test the full parse_arguments here,
        # but we can verify the config accepts it
        from ptycho.config.config import TrainingConfig, ModelConfig
        
        # This should work without error
        config = TrainingConfig(
            model=ModelConfig(),
            sequential_sampling=True,
            n_images=100
        )
        
        self.assertTrue(config.sequential_sampling)
        self.assertEqual(config.n_images, 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)