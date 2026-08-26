"""
Unit tests for the efficient coordinate grouping implementation in RawData.

This test module validates the new _generate_groups_efficiently method
that implements the "sample-then-group" strategy for improved performance.
"""

import unittest
from unittest import mock
import numpy as np
import tempfile
import os
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptycho.raw_data import RawData


def assert_numpy_random_state_equal(test_case, before, after):
    """Assert equality for the tuple returned by ``np.random.get_state``."""
    test_case.assertEqual(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    test_case.assertEqual(before[2:], after[2:])


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

    def test_neighbor_groups_never_cross_object_bank_index(self):
        coords = np.tile(np.arange(9, dtype=np.float64), 2)
        object_index = np.repeat(np.arange(2, dtype=np.int64), 9)
        raw = RawData(
            xcoords=coords,
            ycoords=np.zeros_like(coords),
            xcoords_start=coords.copy(),
            ycoords_start=np.zeros_like(coords),
            diff3d=np.ones((18, 4, 4), dtype=np.float32),
            probeGuess=np.ones((4, 4), dtype=np.complex64),
            scan_index=np.arange(18, dtype=np.int64),
            object_index=object_index,
            Y=np.ones((18, 4, 4), dtype=np.complex64),
        )

        groups = raw._generate_groups_efficiently(
            nsamples=18,
            K=4,
            C=4,
            seed=17,
        )

        for group in groups:
            self.assertEqual(np.unique(object_index[group]).size, 1)

    def test_neighbor_grouping_builds_one_tree_per_object_partition(self):
        from ptycho import grouping as grouping_module

        coords = np.tile(np.arange(9, dtype=np.float64), 2)
        object_index = np.repeat(np.arange(2, dtype=np.int64), 9)
        raw = RawData(
            xcoords=coords,
            ycoords=np.zeros_like(coords),
            xcoords_start=coords.copy(),
            ycoords_start=np.zeros_like(coords),
            diff3d=np.ones((18, 4, 4), dtype=np.float32),
            probeGuess=np.ones((4, 4), dtype=np.complex64),
            scan_index=np.arange(18, dtype=np.int64),
            object_index=object_index,
        )
        real_tree = grouping_module.cKDTree
        tree_inputs = []

        def counting_tree(points):
            tree_inputs.append(np.asarray(points))
            return real_tree(points)

        with mock.patch.object(grouping_module, "cKDTree", side_effect=counting_tree):
            raw._generate_groups_efficiently(
                nsamples=18,
                K=4,
                C=4,
                seed=17,
            )

        self.assertEqual(len(tree_inputs), 2)
        self.assertEqual({len(points) for points in tree_inputs}, {9})

    def test_generate_grouped_data_delegates_to_grouping_plan_owner(self):
        from ptycho.grouping import GroupingPlan

        expected = np.asarray(
            [[0, 1, 20, 21], [22, 23, 40, 41]],
            dtype=np.int64,
        )
        plan = GroupingPlan(
            neighbor_indices=expected,
            center_indices=np.asarray([0, 22]),
            center_available=np.ones(2, dtype=bool),
            eligible_indices=np.arange(self.n_points),
            source_indices=np.arange(self.n_points),
            object_index=np.zeros(2, dtype=np.int64),
            experiment_id=np.zeros(2, dtype=np.int64),
            policy="raw_sequential_sample_then_group",
            coverage_complete=False,
        )
        self.raw_data.Y = np.ones(
            (self.n_points, 64, 64),
            dtype=np.complex64,
        )

        with mock.patch(
            "ptycho.grouping.plan_sample_then_group",
            return_value=plan,
        ) as planner:
            grouped = self.raw_data.generate_grouped_data(
                N=64,
                K=7,
                nsamples=2,
                seed=17,
                sequential_sampling=True,
                gridsize=2,
                neighbor_pool_size=9,
            )

        planner.assert_called_once_with(
            self.raw_data.xcoords,
            self.raw_data.ycoords,
            object_index=self.raw_data.object_index,
            experiment_id=self.raw_data.experiment_id,
            count=2,
            neighbor_count=7,
            group_size=4,
            seed=17,
            rng=None,
            sequential=True,
            enable_oversampling=False,
            neighbor_pool_size=9,
        )
        np.testing.assert_array_equal(grouped["nn_indices"], expected)
        self.assertEqual(grouped["nn_indices"].dtype, np.int32)

    def test_raw_data_rejects_lossy_or_invalid_object_identity(self):
        invalid_values = (
            np.asarray([0.25, 0.75]),
            np.asarray([0.0, np.nan]),
            np.asarray([True, False]),
            np.asarray([0, -1]),
        )

        for object_index in invalid_values:
            with self.subTest(object_index=object_index):
                with self.assertRaisesRegex(
                    ValueError,
                    "object_index.*nonnegative integer",
                ):
                    RawData(
                        xcoords=np.asarray([0.0, 1.0]),
                        ycoords=np.asarray([0.0, 0.0]),
                        xcoords_start=np.asarray([0.0, 1.0]),
                        ycoords_start=np.asarray([0.0, 0.0]),
                        diff3d=np.ones((2, 4, 4), dtype=np.float32),
                        probeGuess=np.ones((4, 4), dtype=np.complex64),
                        scan_index=np.arange(2, dtype=np.int64),
                        object_index=object_index,
                    )

    def test_to_file_omits_absent_optional_arrays_for_strict_npz_loading(self):
        raw = RawData(
            xcoords=np.asarray([0.0, 1.0]),
            ycoords=np.asarray([0.0, 0.0]),
            xcoords_start=np.asarray([0.0, 1.0]),
            ycoords_start=np.asarray([0.0, 0.0]),
            diff3d=np.ones((2, 4, 4), dtype=np.float32),
            probeGuess=np.ones((4, 4), dtype=np.complex64),
            scan_index=np.arange(2, dtype=np.int64),
            objectGuess=np.ones((6, 6), dtype=np.complex64),
            Y=None,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.npz"
            raw.to_file(path)

            with np.load(path, allow_pickle=False) as archive:
                self.assertNotIn("Y", archive.files)
                self.assertIn("objectGuess", archive.files)
                for name in archive.files:
                    np.asarray(archive[name])

            raw.objectGuess = None
            raw.to_file(path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertNotIn("Y", archive.files)
                self.assertNotIn("objectGuess", archive.files)
                for name in archive.files:
                    np.asarray(archive[name])

    def test_unique_scan_ids_do_not_partition_neighbor_grouping(self):
        raw = self.raw_data
        raw.scan_index = np.arange(len(raw.xcoords), dtype=np.int64)

        groups = raw._generate_groups_efficiently(
            nsamples=4,
            K=4,
            C=4,
            seed=17,
        )

        self.assertEqual(groups.shape, (4, 4))
    
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

    def test_generate_grouped_data_seed_does_not_mutate_ambient_numpy_state(self):
        """Seeded public grouping must not read or overwrite NumPy's global RNG."""
        np.random.seed(20260803)
        state_before = np.random.get_state()

        self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=32,
            seed=41,
            gridsize=2,
        )

        assert_numpy_random_state_equal(
            self,
            state_before,
            np.random.get_state(),
        )

    def test_efficient_grouping_consumes_passed_generator(self):
        """Ordinary grouping must draw from the caller-owned Generator."""
        groups1 = self.raw_data._generate_groups_efficiently(
            nsamples=64,
            K=7,
            C=4,
            rng=np.random.default_rng(29),
        )
        groups2 = self.raw_data._generate_groups_efficiently(
            nsamples=64,
            K=7,
            C=4,
            rng=np.random.default_rng(29),
        )
        groups3 = self.raw_data._generate_groups_efficiently(
            nsamples=64,
            K=7,
            C=4,
            rng=np.random.default_rng(30),
        )

        np.testing.assert_array_equal(groups1, groups2)
        self.assertFalse(np.array_equal(groups1, groups3))

    def test_oversampling_consumes_passed_generator_without_mutating_ambient_state(self):
        """K-choose-C grouping must use one passed Generator for every draw."""
        np.random.seed(20260803)
        state_before = np.random.get_state()

        groups1 = self.raw_data._generate_groups_with_oversampling(
            nsamples=self.n_points + 32,
            K=7,
            C=4,
            rng=np.random.default_rng(37),
        )
        groups2 = self.raw_data._generate_groups_with_oversampling(
            nsamples=self.n_points + 32,
            K=7,
            C=4,
            rng=np.random.default_rng(37),
        )

        np.testing.assert_array_equal(groups1, groups2)
        assert_numpy_random_state_equal(
            self,
            state_before,
            np.random.get_state(),
        )

    def test_grouping_rejects_seed_and_generator_together(self):
        """The legacy seed and new Generator inputs are mutually exclusive."""
        with self.assertRaisesRegex(ValueError, "seed.*rng|rng.*seed"):
            self.raw_data.generate_grouped_data(
                N=64,
                K=7,
                nsamples=32,
                seed=41,
                gridsize=2,
                rng=np.random.default_rng(41),
            )

    def test_fresh_grouping_generators_do_not_consume_one_anothers_streams(self):
        """Train-like draws must not perturb an independently seeded validation draw."""
        expected_validation = self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=48,
            gridsize=2,
            rng=np.random.default_rng(53),
        )["nn_indices"]

        train_rng = np.random.default_rng(53)
        self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=96,
            gridsize=2,
            rng=train_rng,
        )
        self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=12,
            gridsize=2,
            rng=train_rng,
        )

        actual_validation = self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=48,
            gridsize=2,
            rng=np.random.default_rng(53),
        )["nn_indices"]

        np.testing.assert_array_equal(expected_validation, actual_validation)

    def test_sequential_grouping_uses_fixed_local_generator(self):
        """Sequential anchors remain seed-independent without touching global state."""
        np.random.seed(20260803)
        state_before = np.random.get_state()

        groups1 = self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=24,
            sequential_sampling=True,
            gridsize=2,
            rng=np.random.default_rng(61),
        )["nn_indices"]
        groups2 = self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=24,
            sequential_sampling=True,
            gridsize=2,
            rng=np.random.default_rng(62),
        )["nn_indices"]

        np.testing.assert_array_equal(groups1, groups2)
        assert_numpy_random_state_equal(
            self,
            state_before,
            np.random.get_state(),
        )
    
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
