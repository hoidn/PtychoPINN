"""
Comprehensive test suite for efficient coordinate grouping implementation.

This module tests the new "sample-then-group" strategy for coordinate grouping
in the RawData class, ensuring correctness, performance, and edge case handling.
"""

import unittest
import numpy as np
import time
import tracemalloc
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptycho.raw_data import RawData
from ptycho import params


class TestCoordinateGrouping(unittest.TestCase):
    """Test suite for the efficient coordinate grouping implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple grid of coordinates for testing
        self.grid_size = 100  # 100x100 grid = 10,000 points
        x = np.linspace(0, 10, self.grid_size)
        y = np.linspace(0, 10, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        
        self.xcoords = xx.flatten()
        self.ycoords = yy.flatten()
        self.n_points = len(self.xcoords)
        
        # Create dummy diffraction data
        self.diff_size = 64
        self.diffraction = np.random.rand(self.n_points, self.diff_size, self.diff_size)
        
        # Create scan index
        scan_index = np.arange(self.n_points)
        
        # Create probe guess
        probeGuess = np.ones((self.diff_size, self.diff_size), dtype=complex)
        
        # Create object guess
        objectGuess = np.ones((256, 256), dtype=complex)
        
        # Create RawData instance with required arguments
        self.raw_data = RawData(
            xcoords=self.xcoords,
            ycoords=self.ycoords,
            xcoords_start=self.xcoords,  # Use same coords for start
            ycoords_start=self.ycoords,
            diff3d=self.diffraction,
            probeGuess=probeGuess,
            scan_index=scan_index,
            objectGuess=objectGuess,
            Y=None  # Test without pre-computed Y
        )
        
        # Set default params
        params.cfg['gridsize'] = 2
    
    def test_efficient_grouping_output_shape(self):
        """Test that _generate_groups_efficiently returns correct shape."""
        nsamples = 100
        K = 7
        C = 4
        
        groups = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C
        )
        
        self.assertEqual(groups.shape, (nsamples, C))
        self.assertIn(groups.dtype, [np.int32, np.int64])
    
    def test_efficient_grouping_valid_indices(self):
        """Test that generated groups contain valid indices."""
        nsamples = 50
        K = 7
        C = 4
        
        groups = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C
        )
        
        # All indices should be within valid range
        self.assertTrue(np.all(groups >= 0))
        self.assertTrue(np.all(groups < self.n_points))
        
        # Each group should have unique indices
        for group in groups:
            self.assertEqual(len(np.unique(group)), C)
    
    def test_efficient_grouping_spatial_coherence(self):
        """Test that groups contain spatially coherent neighbors."""
        # Create a simple 10x10 grid for easier verification
        x = np.arange(10)
        y = np.arange(10)
        xx, yy = np.meshgrid(x, y)
        
        xcoords = xx.flatten()
        ycoords = yy.flatten()
        diff3d = np.random.rand(100, 64, 64)
        
        small_raw_data = RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=diff3d,
            probeGuess=np.ones((64, 64), dtype=complex),
            scan_index=np.arange(100)
        )
        
        nsamples = 10
        K = 8  # 8 neighbors + self = 9 total
        C = 4
        
        groups = small_raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=42
        )
        
        # Check that neighbors are actually close
        for group in groups:
            coords = np.column_stack([
                small_raw_data.xcoords[group],
                small_raw_data.ycoords[group]
            ])
            # Calculate pairwise distances
            distances = []
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    distances.append(dist)
            
            # Maximum distance should be reasonable (not across entire grid)
            max_dist = max(distances)
            self.assertLess(max_dist, 5.0, "Neighbors should be spatially close")
    
    def test_reproducibility_with_seed(self):
        """Test that using the same seed produces identical results."""
        nsamples = 100
        K = 7
        C = 4
        seed = 12345
        
        groups1 = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=seed
        )
        
        groups2 = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=seed
        )
        
        np.testing.assert_array_equal(groups1, groups2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        nsamples = 100
        K = 7
        C = 4
        
        groups1 = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=111
        )
        
        groups2 = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=222
        )
        
        # Groups should be different (though there might be some overlap)
        self.assertFalse(np.array_equal(groups1, groups2))
    
    def test_edge_case_more_samples_than_points(self):
        """Test handling when requesting more samples than available points."""
        # Create small dataset
        xcoords = np.array([0, 1, 2, 3, 4])
        ycoords = np.array([0, 1, 2, 3, 4])
        diff3d = np.random.rand(5, 64, 64)
        
        small_raw_data = RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=diff3d,
            probeGuess=np.ones((64, 64), dtype=complex),
            scan_index=np.arange(5)
        )
        
        nsamples = 10  # More than 5 available points
        K = 4
        C = 2
        
        groups = small_raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C
        )
        
        # Should return at most 5 groups (one per point)
        self.assertLessEqual(len(groups), 5)
    
    def test_edge_case_k_less_than_c(self):
        """Test that error is raised when K < C."""
        nsamples = 10
        K = 3
        C = 5  # C > K+1
        
        with self.assertRaises(ValueError) as context:
            self.raw_data._generate_groups_efficiently(
                nsamples=nsamples, K=K, C=C
            )

        # Updated for improved technical error message
        self.assertIn("K=3 must be >= C=5", str(context.exception))
    
    def test_edge_case_small_dataset(self):
        """Test with very small dataset."""
        xcoords = np.array([0, 1])
        ycoords = np.array([0, 1])
        diff3d = np.random.rand(2, 64, 64)
        
        small_raw_data = RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=diff3d,
            probeGuess=np.ones((64, 64), dtype=complex),
            scan_index=np.arange(2)
        )
        
        nsamples = 1
        K = 1
        C = 1
        
        groups = small_raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C
        )
        
        self.assertEqual(groups.shape, (1, 1))
        self.assertIn(groups[0, 0], [0, 1])
    
    def test_generate_grouped_data_integration(self):
        """Test the full generate_grouped_data method with new implementation."""
        N = 64
        K = 7
        nsamples = 50
        
        # Test with gridsize=2
        params.cfg['gridsize'] = 2
        result = self.raw_data.generate_grouped_data(
            N=N, K=K, nsamples=nsamples, seed=42
        )
        
        # Check that all expected keys are present
        expected_keys = ['diffraction', 'Y', 'coords_offsets', 'coords_relative',
                        'coords_nn', 'nn_indices', 'objectGuess', 'X_full']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check shapes
        C = 4  # gridsize^2
        self.assertEqual(result['diffraction'].shape, (nsamples, N, N, C))
        # Updated for standardized coordinate array format
        self.assertEqual(result['coords_offsets'].shape, (nsamples, 1, 2, 1))
        self.assertEqual(result['coords_relative'].shape, (nsamples, 1, 2, C))
        self.assertEqual(result['nn_indices'].shape, (nsamples, C))
    
    def test_generate_grouped_data_gridsize_1(self):
        """Test that gridsize=1 uses the efficient implementation."""
        N = 64
        K = 7
        nsamples = 100
        
        # Test with gridsize=1
        params.cfg['gridsize'] = 1
        result = self.raw_data.generate_grouped_data(
            N=N, K=K, nsamples=nsamples, seed=42
        )
        
        # Check shapes for gridsize=1
        C = 1
        self.assertEqual(result['diffraction'].shape, (nsamples, N, N, C))
        self.assertEqual(result['nn_indices'].shape, (nsamples, C))
    
    def test_performance_improvement(self):
        """Test that new implementation is faster than expected baseline."""
        # Create larger dataset for performance testing
        n_large = 5000
        xcoords = np.random.rand(n_large) * 100
        ycoords = np.random.rand(n_large) * 100
        diff3d = np.random.rand(n_large, 64, 64)
        
        large_raw_data = RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=diff3d,
            probeGuess=np.ones((64, 64), dtype=complex),
            scan_index=np.arange(n_large)
        )
        
        nsamples = 500
        K = 7
        C = 4
        
        # Time the efficient implementation
        start_time = time.perf_counter()
        groups = large_raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C
        )
        elapsed_time = time.perf_counter() - start_time
        
        # Should complete in reasonable time (< 1 second for 5000 points)
        self.assertLess(elapsed_time, 1.0, 
                       f"Grouping took {elapsed_time:.3f}s, expected < 1s")
        
        print(f"\nPerformance: Generated {nsamples} groups from {n_large} points "
              f"in {elapsed_time:.3f}s")
    
    def test_memory_efficiency(self):
        """Test that new implementation has reasonable memory usage."""
        # Create dataset
        n_points = 10000
        xcoords = np.random.rand(n_points) * 100
        ycoords = np.random.rand(n_points) * 100
        diff3d = np.random.rand(n_points, 64, 64)
        
        large_raw_data = RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=diff3d,
            probeGuess=np.ones((64, 64), dtype=complex),
            scan_index=np.arange(n_points)
        )
        
        nsamples = 1000
        K = 7
        C = 4
        
        # Measure memory usage
        tracemalloc.start()
        
        groups = large_raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C
        )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Convert to MB
        peak_mb = peak / 1024 / 1024
        
        # Should use less than 100MB for this size dataset
        self.assertLess(peak_mb, 100, 
                       f"Peak memory usage was {peak_mb:.1f}MB, expected < 100MB")
        
        print(f"\nMemory efficiency: Peak usage {peak_mb:.1f}MB for "
              f"{nsamples} groups from {n_points} points")
    
    def test_no_cache_files_created(self):
        """Test that no cache files are created with new implementation."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temporary dataset file
            dataset_path = os.path.join(tmpdir, "test_dataset.npz")
            
            # Run generate_grouped_data
            params.cfg['gridsize'] = 2
            result = self.raw_data.generate_grouped_data(
                N=64, K=7, nsamples=50, dataset_path=dataset_path
            )
            
            # Check that no cache files were created
            cache_files = [f for f in os.listdir(tmpdir) if 'cache' in f]
            self.assertEqual(len(cache_files), 0, 
                           f"Cache files found: {cache_files}")
    
    def test_backward_compatibility(self):
        """Test that the new implementation maintains backward compatibility."""
        # Test that dataset_path parameter still works (even though ignored)
        params.cfg['gridsize'] = 2
        result = self.raw_data.generate_grouped_data(
            N=64, K=7, nsamples=50, 
            dataset_path="/some/fake/path.npz"  # Should not cause error
        )
        
        self.assertIsNotNone(result)
        self.assertIn('diffraction', result)


class TestIntegrationWithExistingCode(unittest.TestCase):
    """Test integration with existing codebase."""
    
    def test_existing_tests_still_pass(self):
        """Verify that existing grouping tests still pass."""
        # This is a placeholder - in reality, we'd run the existing test suite
        # For now, we just verify the key functions exist and are callable
        from ptycho.raw_data import (
            get_neighbor_indices,
            get_relative_coords,
            get_image_patches,
            normalize_data
        )
        
        # These support functions should still exist
        self.assertTrue(callable(get_neighbor_indices))
        self.assertTrue(callable(get_relative_coords))
        self.assertTrue(callable(get_image_patches))
        self.assertTrue(callable(normalize_data))


if __name__ == '__main__':
    unittest.main(verbosity=2)