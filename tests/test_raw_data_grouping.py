"""
Unit tests for the efficient coordinate grouping implementation in RawData.

This test module validates the new _generate_groups_efficiently method
that implements the "sample-then-group" strategy for improved performance.
"""

import unittest
import numpy as np
import tempfile
import os
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptycho.raw_data import RawData


class TestRawDataGrouping(unittest.TestCase):
    """Test suite for the efficient grouping implementation."""
    
    def setUp(self):
        """Set up test fixtures with known coordinate patterns."""
        # Create a simple grid of coordinates for testing
        self.grid_size = 20  # 20x20 grid = 400 points
        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        xx, yy = np.meshgrid(x, y)
        
        self.xcoords = xx.flatten()
        self.ycoords = yy.flatten()
        self.n_points = len(self.xcoords)
        
        # Create minimal diffraction data for RawData
        self.diff3d = np.random.rand(self.n_points, 64, 64).astype(np.float32)
        
        # Create a test NPZ file with all required fields
        self.test_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        np.savez(self.test_file.name,
                 xcoords=self.xcoords,
                 ycoords=self.ycoords,
                 xcoords_start=self.xcoords,  # Use same coords for start
                 ycoords_start=self.ycoords,  # Use same coords for start
                 diff3d=self.diff3d,  # Note: key is 'diff3d' not 'diffraction'
                 objectGuess=np.ones((256, 256), dtype=np.complex64),
                 probeGuess=np.ones((64, 64), dtype=np.complex64),
                 scan_index=np.zeros(self.n_points, dtype=np.int32))  # Required field
        
        # Load as RawData instance
        self.raw_data = RawData.from_file(self.test_file.name)
    
    def tearDown(self):
        """Clean up test files."""
        if hasattr(self, 'test_file'):
            os.unlink(self.test_file.name)
    
    def test_output_shape(self):
        """Test that the function returns the correct number and shape of groups."""
        nsamples = 100
        K = 7
        C = 4
        
        groups = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=42
        )
        
        # Check shape
        self.assertEqual(groups.shape, (nsamples, C),
                        f"Expected shape ({nsamples}, {C}), got {groups.shape}")
        
        # Check data type
        self.assertEqual(groups.dtype, np.int32,
                        f"Expected dtype int32, got {groups.dtype}")
    
    def test_content_validity(self):
        """Test that generated groups contain valid neighbor indices."""
        nsamples = 50
        K = 8
        C = 4
        
        groups = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=42
        )
        
        # All indices should be within valid range
        self.assertTrue(np.all(groups >= 0),
                       "Found negative indices in groups")
        self.assertTrue(np.all(groups < self.n_points),
                       f"Found indices >= {self.n_points} in groups")
        
        # Check that indices in each group are spatially close
        coords = np.column_stack([self.xcoords, self.ycoords])
        
        for group in groups[:10]:  # Check first 10 groups
            group_coords = coords[group]
            # Calculate pairwise distances within group
            center = group_coords.mean(axis=0)
            distances = np.linalg.norm(group_coords - center, axis=1)
            max_dist = distances.max()
            
            # Neighbors should be reasonably close (within sqrt(K) grid units typically)
            self.assertLess(max_dist, np.sqrt(K) * 2,
                          f"Group has maximum distance {max_dist}, seems too large for K={K}")
    
    def test_edge_case_more_samples_than_points(self):
        """Test behavior when requesting more samples than available points."""
        nsamples = self.n_points + 100  # Request more than available
        K = 4
        C = 2
        
        groups = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=42
        )
        
        # Should return exactly n_points groups
        self.assertEqual(groups.shape[0], self.n_points,
                        f"Expected {self.n_points} groups when requesting {nsamples}")
    
    def test_edge_case_k_less_than_c(self):
        """Test that K < C raises appropriate error."""
        with self.assertRaises(ValueError) as context:
            self.raw_data._generate_groups_efficiently(
                nsamples=10, K=3, C=5, seed=42
            )
        
        self.assertIn("must be >=", str(context.exception),
                     "Error message should explain K must be >= C")
    
    def test_edge_case_small_dataset(self):
        """Test with a very small dataset."""
        # Create tiny dataset with just 5 points
        small_xcoords = np.array([0, 1, 0, 1, 0.5])
        small_ycoords = np.array([0, 0, 1, 1, 0.5])
        small_diff = np.random.rand(5, 32, 32)
        
        # Create temporary file with all required fields
        small_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        np.savez(small_file.name,
                 xcoords=small_xcoords,
                 ycoords=small_ycoords,
                 xcoords_start=small_xcoords,
                 ycoords_start=small_ycoords,
                 diff3d=small_diff,
                 objectGuess=np.ones((128, 128), dtype=np.complex64),
                 probeGuess=np.ones((32, 32), dtype=np.complex64),
                 scan_index=np.zeros(5, dtype=np.int32))
        
        try:
            small_data = RawData.from_file(small_file.name)
            
            # Should work with C <= 5
            groups = small_data._generate_groups_efficiently(
                nsamples=3, K=4, C=3, seed=42
            )
            self.assertEqual(groups.shape, (3, 3))
            
            # Should work even when requesting more samples
            groups = small_data._generate_groups_efficiently(
                nsamples=10, K=4, C=2, seed=42
            )
            self.assertEqual(groups.shape[0], 5)  # Only 5 points available
            
        finally:
            os.unlink(small_file.name)
    
    def test_reproducibility(self):
        """Test that the same seed produces identical results."""
        nsamples = 100
        K = 6
        C = 4
        seed = 12345
        
        # Generate groups twice with same seed
        groups1 = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=seed
        )
        groups2 = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=seed
        )
        
        # Should be identical
        np.testing.assert_array_equal(groups1, groups2,
                                     "Same seed should produce identical results")
        
        # Different seed should produce different results
        groups3 = self.raw_data._generate_groups_efficiently(
            nsamples=nsamples, K=K, C=C, seed=seed + 1
        )
        
        # Should be different (with high probability)
        self.assertFalse(np.array_equal(groups1, groups3),
                        "Different seeds should produce different results")
    
    def test_performance_improvement(self):
        """Test that the new method is faster than the old approach (when not cached)."""
        # Create a larger dataset for performance testing
        large_size = 100  # 100x100 = 10,000 points
        x = np.arange(large_size)
        y = np.arange(large_size) 
        xx, yy = np.meshgrid(x, y)
        
        large_xcoords = xx.flatten()
        large_ycoords = yy.flatten()
        large_diff = np.random.rand(len(large_xcoords), 32, 32).astype(np.float32)
        
        # Create large test file with all required fields
        large_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        np.savez(large_file.name,
                 xcoords=large_xcoords,
                 ycoords=large_ycoords,
                 xcoords_start=large_xcoords,
                 ycoords_start=large_ycoords,
                 diff3d=large_diff,
                 objectGuess=np.ones((512, 512), dtype=np.complex64),
                 probeGuess=np.ones((32, 32), dtype=np.complex64),
                 scan_index=np.zeros(len(large_xcoords), dtype=np.int32))
        
        try:
            large_data = RawData.from_file(large_file.name)
            
            # Time the new efficient method
            start_time = time.time()
            groups_efficient = large_data._generate_groups_efficiently(
                nsamples=512, K=8, C=4, seed=42
            )
            efficient_time = time.time() - start_time
            
            print(f"\nEfficient method time: {efficient_time:.4f} seconds")
            print(f"Generated {groups_efficient.shape[0]} groups")
            
            # The new method should be very fast (typically < 0.1 seconds)
            self.assertLess(efficient_time, 1.0,
                          f"Efficient method took {efficient_time:.2f}s, expected < 1s")
            
            # Note: We're not comparing with the old method here because:
            # 1. It would require running the inefficient code
            # 2. The old method with caching might be fast on subsequent runs
            # 3. The real improvement is on first-run performance
            
        finally:
            os.unlink(large_file.name)
    
    def test_memory_efficiency(self):
        """Test that memory usage is reasonable for large datasets."""
        import tracemalloc
        
        # Create a moderate dataset
        moderate_size = 50  # 50x50 = 2,500 points
        x = np.arange(moderate_size)
        y = np.arange(moderate_size)
        xx, yy = np.meshgrid(x, y)
        
        mod_xcoords = xx.flatten()
        mod_ycoords = yy.flatten()
        mod_diff = np.random.rand(len(mod_xcoords), 32, 32).astype(np.float32)
        
        # Create test file with all required fields
        mod_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        np.savez(mod_file.name,
                 xcoords=mod_xcoords,
                 ycoords=mod_ycoords,
                 xcoords_start=mod_xcoords,
                 ycoords_start=mod_ycoords,
                 diff3d=mod_diff,
                 objectGuess=np.ones((256, 256), dtype=np.complex64),
                 probeGuess=np.ones((32, 32), dtype=np.complex64),
                 scan_index=np.zeros(len(mod_xcoords), dtype=np.int32))
        
        try:
            mod_data = RawData.from_file(mod_file.name)
            
            # Measure memory usage
            tracemalloc.start()
            snapshot_before = tracemalloc.take_snapshot()
            
            groups = mod_data._generate_groups_efficiently(
                nsamples=256, K=8, C=4, seed=42
            )
            
            snapshot_after = tracemalloc.take_snapshot()
            tracemalloc.stop()
            
            # Calculate memory difference
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            total_memory = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
            memory_mb = total_memory / 1024 / 1024
            
            print(f"\nMemory used for 256 groups from 2,500 points: {memory_mb:.2f} MB")
            
            # Memory usage should be minimal (< 10 MB for this size)
            self.assertLess(memory_mb, 10.0,
                          f"Memory usage {memory_mb:.2f} MB seems excessive")
            
        finally:
            os.unlink(mod_file.name)
    
    def test_uniform_sampling(self):
        """Test that sampling is reasonably uniform across the dataset."""
        nsamples = self.n_points // 4  # Sample 25% of points
        K = 6
        C = 1  # Use C=1 to track which points are sampled
        
        # Run multiple times to check distribution
        n_runs = 100
        sample_counts = np.zeros(self.n_points)
        
        for run in range(n_runs):
            groups = self.raw_data._generate_groups_efficiently(
                nsamples=nsamples, K=K, C=C, seed=run
            )
            # Count how often each point is sampled
            unique_indices = np.unique(groups.flatten())
            sample_counts[unique_indices] += 1
        
        # Check that sampling is reasonably uniform
        # Each point should be sampled roughly (nsamples/n_points) * n_runs times
        expected_count = (nsamples / self.n_points) * n_runs
        
        # Allow 3x variation from expected
        min_count = expected_count / 3
        max_count = expected_count * 3
        
        # Most points should be within expected range
        within_range = np.sum((sample_counts >= min_count) & (sample_counts <= max_count))
        fraction_within = within_range / self.n_points
        
        self.assertGreater(fraction_within, 0.8,
                          f"Only {fraction_within:.1%} of points sampled uniformly")


if __name__ == '__main__':
    unittest.main(verbosity=2)