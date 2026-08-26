"""
Unit tests for the public RawData grouping contract.

This module validates the centered-nearest grouping behavior of
``RawData.generate_grouped_data``: exact unique center selection, one
split-local generator consumed for both center and neighbor selection,
sequential first-centers with a fixed seed-0 neighbor stream, exact-count
rejection, and the public grouped-dictionary dtype/layout contract.
"""

import unittest
from unittest import mock
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptycho.raw_data import RawData
from ptycho.grouping import plan_nearest_groups


def assert_numpy_random_state_equal(test_case, before, after):
    """Assert equality for the tuple returned by ``np.random.get_state``."""
    test_case.assertEqual(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    test_case.assertEqual(before[2:], after[2:])


class TestRawDataGrouping(unittest.TestCase):
    """Test suite for the public grouping implementation."""

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

    def test_random_grouping_selects_exact_unique_centers_then_uses_advanced_rng(self):
        """Random grouping draws exact centers, then the SAME advanced RNG for neighbors."""
        N = 64
        K = 7
        gridsize = 2
        seed = 42
        size = self.n_points

        for nsamples in (32, size):
            with self.subTest(nsamples=nsamples):
                # Reconstruct the exact expected RNG stream: the choice draw
                # always happens first -- including the all-row case, where the
                # permutation draw is consumed and then canonicalized to arange.
                expected_rng = np.random.default_rng(seed)
                drawn_centers = expected_rng.choice(size, nsamples, replace=False)
                expected_centers = (
                    np.arange(size, dtype=np.int64)
                    if nsamples == size
                    else drawn_centers
                )
                expected = plan_nearest_groups(
                    self.raw_data.xcoords,
                    self.raw_data.ycoords,
                    center_indices=expected_centers,
                    candidate_indices=np.arange(size),
                    group_size=gridsize**2,
                    neighbor_count=K,
                    rng=expected_rng,
                )
                actual = self.raw_data.generate_grouped_data(
                    N=N, K=K, nsamples=nsamples, seed=seed, gridsize=gridsize
                )

                nn_indices = actual["nn_indices"]
                self.assertEqual(nn_indices.shape, (nsamples, gridsize**2))
                self.assertEqual(nn_indices.dtype, np.int32)
                # The public dictionary's first column maps to the exact drawn
                # (or canonicalized all-row) centers.
                np.testing.assert_array_equal(
                    nn_indices[:, 0], expected.center_indices
                )
                # One generator was consumed for centers then neighbors: full parity.
                np.testing.assert_array_equal(nn_indices, expected.neighbor_indices)
                self.assertEqual(len(np.unique(nn_indices[:, 0])), nsamples)

    def test_sequential_grouping_uses_first_centers_and_seed_zero_neighbors(self):
        """Sequential grouping uses the first rows as centers and a fixed seed-0 stream."""
        N = 64
        K = 7
        nsamples = 24
        gridsize = 2
        size = self.n_points

        expected = plan_nearest_groups(
            self.raw_data.xcoords,
            self.raw_data.ycoords,
            center_indices=np.arange(nsamples, dtype=np.int64),
            candidate_indices=np.arange(size),
            group_size=gridsize**2,
            neighbor_count=K,
            rng=np.random.default_rng(0),
        )
        actual = self.raw_data.generate_grouped_data(
            N=N,
            K=K,
            nsamples=nsamples,
            sequential_sampling=True,
            gridsize=gridsize,
        )

        nn_indices = actual["nn_indices"]
        self.assertEqual(nn_indices.shape, (nsamples, gridsize**2))
        # First centers: column zero is exactly rows 0..nsamples-1.
        np.testing.assert_array_equal(
            nn_indices[:, 0], np.arange(nsamples, dtype=np.int64)
        )
        # Seed-zero neighbor stream: full parity with the expected plan.
        np.testing.assert_array_equal(nn_indices, expected.neighbor_indices)

        # C=1 sequential: each row is the exact ordered center identity.
        c1 = self.raw_data.generate_grouped_data(
            N=N,
            K=3,
            nsamples=5,
            sequential_sampling=True,
            gridsize=1,
        )
        self.assertEqual(c1["nn_indices"].dtype, np.int64)
        np.testing.assert_array_equal(
            c1["nn_indices"], np.arange(5, dtype=np.int64).reshape(-1, 1)
        )

    def test_grouping_rejects_more_centers_than_candidates(self):
        """A request larger than the candidate pool fails instead of oversampling."""
        with self.assertRaisesRegex(
            ValueError,
            "requested 401 unique centers from only 400 candidates",
        ):
            self.raw_data.generate_grouped_data(
                N=64,
                K=4,
                nsamples=self.n_points + 1,
                seed=17,
                gridsize=2,
            )

        # C=1 is subject to the same exact-count contract.
        with self.assertRaisesRegex(ValueError, "unique centers from only"):
            self.raw_data.generate_grouped_data(
                N=64,
                K=4,
                nsamples=self.n_points + 1,
                seed=17,
                gridsize=1,
            )

    def test_grouping_rejects_seed_and_rng_together_before_center_selection(self):
        """seed plus rng is rejected by RawData before any planner call."""
        from ptycho import grouping

        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                with mock.patch.object(
                    grouping,
                    "plan_nearest_groups",
                    side_effect=AssertionError("planner must not run"),
                ):
                    with self.assertRaisesRegex(ValueError, "seed.*rng|rng.*seed"):
                        self.raw_data.generate_grouped_data(
                            N=64,
                            K=7,
                            nsamples=32,
                            seed=41,
                            sequential_sampling=sequential,
                            gridsize=2,
                            rng=np.random.default_rng(41),
                        )

    def test_grouped_data_is_center_first_same_object_and_preserves_layout(self):
        """Every C>1 row begins with its center, stays in one object, and stays int32."""
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
        size = len(coords)
        seed = 17
        nsamples = 12
        gridsize = 2

        expected_rng = np.random.default_rng(seed)
        drawn_centers = expected_rng.choice(size, nsamples, replace=False)
        expected = plan_nearest_groups(
            raw.xcoords,
            raw.ycoords,
            center_indices=drawn_centers,
            candidate_indices=np.arange(size),
            group_size=gridsize**2,
            neighbor_count=4,
            object_index=object_index,
            rng=expected_rng,
        )
        grouped = raw.generate_grouped_data(
            N=4, K=4, nsamples=nsamples, seed=seed, gridsize=gridsize
        )
        nn_indices = grouped["nn_indices"]

        self.assertEqual(nn_indices.shape, (nsamples, gridsize**2))
        self.assertEqual(nn_indices.dtype, np.int32)
        np.testing.assert_array_equal(nn_indices[:, 0], expected.center_indices)
        # Center-first: column zero equals the plan's designated center rows.
        np.testing.assert_array_equal(nn_indices, expected.neighbor_indices)
        for row in nn_indices:
            self.assertEqual(np.unique(object_index[row]).size, 1)
        # All members are distinct within each row.
        for row in nn_indices:
            self.assertEqual(len(np.unique(row)), gridsize**2)

        # C=1 retains the int64 public layout and exact center identity.
        c1 = raw.generate_grouped_data(
            N=4, K=1, nsamples=nsamples, seed=seed, gridsize=1
        )
        self.assertEqual(c1["nn_indices"].dtype, np.int64)
        np.testing.assert_array_equal(c1["nn_indices"][:, 0], expected.center_indices)
        self.assertEqual(c1["nn_indices"].shape, (nsamples, 1))

    def test_train_and_validation_grouping_use_independent_generators(self):
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

    def test_c1_all_row_selection_preserves_canonical_order_and_int64(self):
        """Random all-row C=1 selection emits canonical order and exact int64 arrays."""
        size = self.n_points
        grouped = self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=size,
            seed=11,
            gridsize=1,
        )
        self.assertEqual(grouped["nn_indices"].dtype, np.int64)
        np.testing.assert_array_equal(
            grouped["nn_indices"],
            np.arange(size, dtype=np.int64).reshape(-1, 1),
        )
        self.assertEqual(grouped["nn_indices"].shape, (size, 1))

    def test_content_validity(self):
        """Test that generated groups contain valid, spatially close indices."""
        nsamples = 50
        K = 8
        gridsize = 2

        nn_indices = self.raw_data.generate_grouped_data(
            N=64, K=K, nsamples=nsamples, seed=42, gridsize=gridsize
        )["nn_indices"]

        # All indices should be within valid range
        self.assertTrue(np.all(nn_indices >= 0),
                        "Found negative indices in groups")
        self.assertTrue(np.all(nn_indices < self.n_points),
                        f"Found indices >= {self.n_points} in groups")

        # Check that indices in each group are spatially close
        coords = np.column_stack([self.xcoords, self.ycoords])

        for group in nn_indices[:10]:  # Check first 10 groups
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

        nn_indices = raw.generate_grouped_data(
            N=4,
            K=4,
            nsamples=18,
            seed=17,
            gridsize=2,
        )["nn_indices"]

        for group in nn_indices:
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
            raw.generate_grouped_data(
                N=4,
                K=4,
                nsamples=18,
                seed=17,
                gridsize=2,
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
            source_indices=np.arange(self.n_points),
            object_index=np.zeros(2, dtype=np.int64),
            experiment_id=np.zeros(2, dtype=np.int64),
        )
        self.raw_data.Y = np.ones(
            (self.n_points, 64, 64),
            dtype=np.complex64,
        )

        with mock.patch(
            "ptycho.grouping.plan_nearest_groups",
            return_value=plan,
        ) as planner:
            grouped = self.raw_data.generate_grouped_data(
                N=64,
                K=7,
                nsamples=2,
                seed=17,
                sequential_sampling=True,
                gridsize=2,
            )

        planner.assert_called_once()
        call = planner.call_args
        self.assertEqual(call.kwargs["group_size"], 4)
        self.assertEqual(call.kwargs["neighbor_count"], 7)
        np.testing.assert_array_equal(
            call.kwargs["center_indices"], np.arange(2, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            call.kwargs["candidate_indices"],
            np.arange(self.n_points, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            call.kwargs["source_indices"],
            np.arange(self.n_points, dtype=np.int64),
        )
        self.assertIsInstance(call.kwargs["rng"], np.random.Generator)
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

        nn_indices = raw.generate_grouped_data(
            N=64,
            K=4,
            nsamples=4,
            seed=17,
            gridsize=2,
        )["nn_indices"]

        self.assertEqual(nn_indices.shape, (4, 4))

    def test_edge_case_k_less_than_c(self):
        """Test that K < C-1 raises an appropriate error."""
        with self.assertRaises(ValueError) as context:
            self.raw_data.generate_grouped_data(
                N=64, K=4, nsamples=10, seed=42, gridsize=3
            )

        self.assertIn("K=4", str(context.exception),
                     "Error message should name the offending K")

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
            nn_indices = small_data.generate_grouped_data(
                N=32, K=4, nsamples=3, seed=42, gridsize=2
            )["nn_indices"]
            self.assertEqual(nn_indices.shape, (3, 4))

            # Requesting more centers than the pool fails (no oversampling).
            with self.assertRaisesRegex(ValueError, "unique centers from only"):
                small_data.generate_grouped_data(
                    N=32, K=4, nsamples=10, seed=42, gridsize=2
                )

        finally:
            os.unlink(small_file.name)

    def test_reproducibility(self):
        """Test that the same seed produces identical results."""
        nsamples = 100
        K = 6
        gridsize = 1
        seed = 12345

        # Generate groups twice with same seed
        nn_indices1 = self.raw_data.generate_grouped_data(
            N=64, K=K, nsamples=nsamples, seed=seed, gridsize=gridsize
        )["nn_indices"]
        nn_indices2 = self.raw_data.generate_grouped_data(
            N=64, K=K, nsamples=nsamples, seed=seed, gridsize=gridsize
        )["nn_indices"]

        # Should be identical
        np.testing.assert_array_equal(nn_indices1, nn_indices2,
                                     "Same seed should produce identical results")

        # Different seed should produce different results
        nn_indices3 = self.raw_data.generate_grouped_data(
            N=64, K=K, nsamples=nsamples, seed=seed + 1, gridsize=gridsize
        )["nn_indices"]

        # Should be different (with high probability)
        self.assertFalse(np.array_equal(nn_indices1, nn_indices3),
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

    def test_grouping_consumes_passed_generator(self):
        """Ordinary grouping must draw from the caller-owned Generator."""
        nn_indices1 = self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=64,
            gridsize=2,
            rng=np.random.default_rng(29),
        )["nn_indices"]
        nn_indices2 = self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=64,
            gridsize=2,
            rng=np.random.default_rng(29),
        )["nn_indices"]
        nn_indices3 = self.raw_data.generate_grouped_data(
            N=64,
            K=7,
            nsamples=64,
            gridsize=2,
            rng=np.random.default_rng(30),
        )["nn_indices"]

        np.testing.assert_array_equal(nn_indices1, nn_indices2)
        self.assertFalse(np.array_equal(nn_indices1, nn_indices3))

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
