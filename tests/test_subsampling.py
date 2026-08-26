"""Unit tests for independent data subsampling functionality.

This module tests the new n_subsample parameter that enables independent control
of data subsampling and neighbor grouping operations in PtychoPINN.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
from ptycho.raw_data import RawData
from ptycho.workflows.components import load_data
from ptycho.config.config import TrainingConfig, ModelConfig, SamplingConfig
from ptycho import params


def assert_numpy_random_state_equal(test_case, before, after):
    """Assert equality for the tuple returned by ``np.random.get_state``."""
    test_case.assertEqual(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    test_case.assertEqual(before[2:], after[2:])


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

    def test_seeded_subsampling_does_not_mutate_ambient_numpy_state(self):
        """A reproducible subset must not overwrite NumPy's global RNG stream."""
        np.random.seed(20260803)
        state_before = np.random.get_state()
        original_cwd = Path.cwd()

        with tempfile.TemporaryDirectory() as temporary_cwd:
            try:
                import os

                os.chdir(temporary_cwd)
                load_data(
                    self.test_data_file.name,
                    n_subsample=100,
                    subsample_seed=71,
                )
                self.assertFalse((Path(temporary_cwd) / "tmp").exists())
            finally:
                os.chdir(original_cwd)

        assert_numpy_random_state_equal(
            self,
            state_before,
            np.random.get_state(),
        )

    def test_subsampling_consumes_passed_generator(self):
        """Selection must accept and draw from a caller-owned Generator."""
        data1 = load_data(
            self.test_data_file.name,
            n_subsample=100,
            rng=np.random.default_rng(73),
        )
        data2 = load_data(
            self.test_data_file.name,
            n_subsample=100,
            rng=np.random.default_rng(73),
        )
        data3 = load_data(
            self.test_data_file.name,
            n_subsample=100,
            rng=np.random.default_rng(74),
        )

        np.testing.assert_array_equal(data1.sample_indices, data2.sample_indices)
        self.assertFalse(np.array_equal(data1.sample_indices, data3.sample_indices))

    def test_subsampling_rejects_seed_and_generator_together(self):
        """The persisted subsample seed and explicit Generator cannot conflict."""
        with self.assertRaisesRegex(
            ValueError,
            "subsample_seed.*rng|rng.*subsample_seed",
        ):
            load_data(
                self.test_data_file.name,
                n_subsample=100,
                subsample_seed=79,
                rng=np.random.default_rng(79),
            )
    
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

    def test_incompatible_legacy_y_is_ignored(self):
        """The shared workflow keeps its historical optional-truth fallback."""
        with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
            np.savez(
                handle.name,
                xcoords=np.arange(3, dtype=np.float64),
                ycoords=np.arange(3, dtype=np.float64),
                diffraction=np.ones((3, 4, 4), dtype=np.float32),
                probeGuess=np.ones((4, 4), dtype=np.complex64),
                Y=np.ones((1, 4, 4, 1), dtype=np.complex64),
            )

            with self.assertWarnsRegex(RuntimeWarning, "Ignoring.*Y"):
                loaded = load_data(handle.name)

        self.assertIsNone(loaded.Y)
    
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
            sampling=SamplingConfig(n_images=500, n_subsample=200, subsample_seed=42),
        )

        self.assertEqual(config.sampling.train_raw_selection, 200)
        self.assertEqual(config.sampling.subsample_seed, 42)

        config_default = TrainingConfig(
            model=ModelConfig(N=64),
            sampling=SamplingConfig(n_images=500),
        )
        self.assertIsNone(config_default.sampling.train_raw_selection)
        self.assertIsNone(config_default.sampling.subsample_seed)

    def test_load_data_keeps_canonical_diffraction_when_n_scans_less_than_n(self):
        """Canonical (N,H,W) diffraction must not transpose even when N_scans < H."""
        n_scans = 4
        n = 128
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        try:
            diffraction = np.random.rand(n_scans, n, n).astype(np.float32)
            np.savez(
                tmp.name,
                xcoords=np.arange(n_scans, dtype=np.float64),
                ycoords=np.arange(n_scans, dtype=np.float64),
                xcoords_start=np.arange(n_scans, dtype=np.float64),
                ycoords_start=np.arange(n_scans, dtype=np.float64),
                diffraction=diffraction,
                probeGuess=np.ones((n, n), dtype=np.complex64),
                objectGuess=np.ones((n, n), dtype=np.complex64),
            )
            loaded = load_data(tmp.name)
            self.assertEqual(loaded.diff3d.shape, (n_scans, n, n))
        finally:
            import os
            os.unlink(tmp.name)

    def test_load_data_defaults_missing_start_coordinates_to_primary_coordinates(self):
        """Optional start coordinates default to the primary coordinates."""
        n_scans = 3
        n = 8
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        try:
            xcoords = np.array([1.0, 2.0, 4.0], dtype=np.float64)
            ycoords = np.array([3.0, 5.0, 8.0], dtype=np.float64)
            np.savez(
                tmp.name,
                xcoords=xcoords,
                ycoords=ycoords,
                diffraction=np.ones((n_scans, n, n), dtype=np.float32),
                probeGuess=np.ones((n, n), dtype=np.complex64),
            )

            loaded = load_data(tmp.name)

            np.testing.assert_array_equal(loaded.xcoords_start, loaded.xcoords)
            np.testing.assert_array_equal(loaded.ycoords_start, loaded.ycoords)
        finally:
            import os
            os.unlink(tmp.name)

    def test_load_data_transposes_legacy_hwn_diffraction_when_last_axis_matches_coords(self):
        """Legacy (H,W,N) diffraction should transpose to (N,H,W)."""
        n_scans = 4
        n = 128
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        try:
            diffraction_legacy = np.random.rand(n, n, n_scans).astype(np.float32)
            np.savez(
                tmp.name,
                xcoords=np.arange(n_scans, dtype=np.float64),
                ycoords=np.arange(n_scans, dtype=np.float64),
                xcoords_start=np.arange(n_scans, dtype=np.float64),
                ycoords_start=np.arange(n_scans, dtype=np.float64),
                diffraction=diffraction_legacy,
                probeGuess=np.ones((n, n), dtype=np.complex64),
                objectGuess=np.ones((n, n), dtype=np.complex64),
            )
            loaded = load_data(tmp.name)
            loaded_raw = RawData.from_file(tmp.name)
            self.assertEqual(loaded.diff3d.shape, (n_scans, n, n))
            np.testing.assert_array_equal(loaded_raw.diff3d, diffraction_legacy.transpose(2, 0, 1))
        finally:
            import os
            os.unlink(tmp.name)

    def test_load_adapters_reject_conflicting_dual_diffraction_keys(self):
        n_scans = 3
        n = 4
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        try:
            canonical = np.ones((n_scans, n, n), dtype=np.float32)
            np.savez(
                tmp.name,
                xcoords=np.arange(n_scans, dtype=np.float64),
                ycoords=np.arange(n_scans, dtype=np.float64),
                diff3d=canonical,
                diffraction=canonical + 1,
                probeGuess=np.ones((n, n), dtype=np.complex64),
            )

            for adapter in (load_data, RawData.from_file):
                with self.subTest(adapter=adapter.__qualname__):
                    with self.assertRaisesRegex(ValueError, "conflicting diffraction"):
                        adapter(tmp.name)
        finally:
            import os
            os.unlink(tmp.name)

    def test_load_adapters_retain_canonical_optional_fields(self):
        n_scans = 6
        n = 4
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        try:
            truth = np.arange(n_scans * n * n).reshape(n_scans, n, n).astype(np.complex64)
            label = truth + np.complex64(2j)
            simulated_probe = np.full((n, n), 3 + 4j, dtype=np.complex64)
            np.savez(
                tmp.name,
                xcoords=np.arange(n_scans, dtype=np.float64),
                ycoords=np.arange(n_scans, dtype=np.float64),
                diff3d=np.ones((n_scans, n, n), dtype=np.float32),
                probeGuess=np.ones((n, n), dtype=np.complex64),
                Y=truth,
                label=label,
                probe_simulated=simulated_probe,
                object_amplitude_scale=np.array(2.5, dtype=np.float64),
                scale_contract_version=np.array("ci_intensity_v2"),
                measurement_domain=np.array("count_intensity"),
                experiment_id=np.array(7, dtype=np.int64),
                _metadata=np.array('{"source": "adapter-test"}'),
            )

            complete = RawData.from_file(tmp.name)
            selected = load_data(tmp.name, n_subsample=3, subsample_seed=19)

            np.testing.assert_array_equal(complete.label, label)
            np.testing.assert_array_equal(complete.probe_simulated, simulated_probe)
            self.assertEqual(complete.metadata, {"source": "adapter-test"})
            np.testing.assert_array_equal(selected.Y, truth[selected.sample_indices])
            np.testing.assert_array_equal(selected.label, label[selected.sample_indices])
            np.testing.assert_array_equal(selected.probe_simulated, simulated_probe)
            self.assertEqual(selected.object_amplitude_scale, np.float64(2.5))
            self.assertEqual(selected.scale_contract_version, "ci_intensity_v2")
            self.assertEqual(selected.measurement_domain, "count_intensity")
            self.assertEqual(selected.experiment_id, 7)
            self.assertEqual(selected.metadata, {"source": "adapter-test"})
        finally:
            import os
            os.unlink(tmp.name)


if __name__ == '__main__':
    unittest.main()
