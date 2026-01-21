"""Tests for ptycho/data_preprocessing.py dataset_intensity_stats attachment.

Per Phase D4f.3: create_ptycho_dataset() must compute and attach dataset_intensity_stats
to both train and test PtychoDataContainer instances so the spec-compliant dataset-derived
intensity_scale formula can be used instead of the closed-form fallback.

See: specs/spec-ptycho-core.md §Normalization Invariants
"""
import numpy as np

from ptycho.loader import compute_dataset_intensity_stats


class TestCreatePtychoDataset:
    """Test suite for create_ptycho_dataset dataset_intensity_stats attachment."""

    def test_attaches_dataset_stats(self):
        """Verify create_ptycho_dataset attaches dataset_intensity_stats to containers.

        Per Phase D4f.3: The function must:
        1. Call compute_dataset_intensity_stats for both train and test data
        2. Pass the resulting dicts to each PtychoDataContainer
        3. Use is_normalized=True with the provided intensity_scale to back-compute raw stats

        This ensures training/inference can use the dataset-derived formula from
        specs/spec-ptycho-core.md §Normalization Invariants instead of falling back to
        the closed-form 988.21 constant.
        """
        from ptycho.data_preprocessing import create_ptycho_dataset
        from ptycho import params as p
        from ptycho import probe as probe_module

        # Setup params.cfg for the test
        N = 64
        gridsize = 2
        p.cfg['N'] = N
        p.cfg['gridsize'] = gridsize
        p.cfg['default_probe_scale'] = 4.0

        # Set up probe in params.cfg (required by create_ptycho_dataset -> probe.get_probe)
        # probe.get_probe expects shape (N, N, 2) where last dim is [real, imag]
        test_probe = np.ones((N, N, 2), dtype=np.float32)
        p.cfg['probe'] = test_probe

        # Create deterministic test data
        np.random.seed(42)
        B_train = 8
        B_test = 4
        C = gridsize ** 2  # 4 channels for gridsize=2

        # Simulate normalized diffraction arrays (as would come from mk_simdata)
        X_train = np.random.rand(B_train, N, N, C).astype(np.float32) + 0.1
        X_test = np.random.rand(B_test, N, N, C).astype(np.float32) + 0.1

        Y_I_train = np.random.rand(B_train, N, N, C).astype(np.float32)
        Y_I_test = np.random.rand(B_test, N, N, C).astype(np.float32)

        Y_phi_train = np.zeros_like(Y_I_train)
        Y_phi_test = np.zeros_like(Y_I_test)

        # Mock intensity_scale as would be returned by mk_simdata
        intensity_scale = 500.0

        # YY_full can be None for this test
        YY_train_full = None
        YY_test_full = None

        # Coords in channel format (B, 1, 2, C)
        coords_train_nominal = np.zeros((B_train, 1, 2, C), dtype=np.float32)
        coords_train_true = np.zeros((B_train, 1, 2, C), dtype=np.float32)
        coords_test_nominal = np.zeros((B_test, 1, 2, C), dtype=np.float32)
        coords_test_true = np.zeros((B_test, 1, 2, C), dtype=np.float32)

        # Call create_ptycho_dataset
        ptycho_dataset = create_ptycho_dataset(
            X_train, Y_I_train, Y_phi_train, intensity_scale,
            YY_train_full, coords_train_nominal, coords_train_true,
            X_test, Y_I_test, Y_phi_test, YY_test_full,
            coords_test_nominal, coords_test_true
        )

        # Verify train container has stats
        train_container = ptycho_dataset.train_data
        assert hasattr(train_container, 'dataset_intensity_stats'), \
            "train_container should have dataset_intensity_stats attribute"
        assert train_container.dataset_intensity_stats is not None, \
            "train_container.dataset_intensity_stats should not be None"

        train_stats = train_container.dataset_intensity_stats
        assert 'batch_mean_sum_intensity' in train_stats, \
            "Train stats missing batch_mean_sum_intensity key"
        assert 'n_samples' in train_stats, \
            "Train stats missing n_samples key"
        assert train_stats['batch_mean_sum_intensity'] > 0, \
            "Train batch_mean_sum_intensity should be positive"
        assert train_stats['n_samples'] == B_train, \
            f"Train n_samples should be {B_train}, got {train_stats['n_samples']}"

        # Verify test container has stats
        test_container = ptycho_dataset.test_data
        assert hasattr(test_container, 'dataset_intensity_stats'), \
            "test_container should have dataset_intensity_stats attribute"
        assert test_container.dataset_intensity_stats is not None, \
            "test_container.dataset_intensity_stats should not be None"

        test_stats = test_container.dataset_intensity_stats
        assert 'batch_mean_sum_intensity' in test_stats, \
            "Test stats missing batch_mean_sum_intensity key"
        assert 'n_samples' in test_stats, \
            "Test stats missing n_samples key"
        assert test_stats['batch_mean_sum_intensity'] > 0, \
            "Test batch_mean_sum_intensity should be positive"
        assert test_stats['n_samples'] == B_test, \
            f"Test n_samples should be {B_test}, got {test_stats['n_samples']}"

        # Verify stats match compute_dataset_intensity_stats with is_normalized=True
        expected_train_stats = compute_dataset_intensity_stats(
            X_train, intensity_scale=intensity_scale, is_normalized=True
        )
        expected_test_stats = compute_dataset_intensity_stats(
            X_test, intensity_scale=intensity_scale, is_normalized=True
        )

        # Allow small tolerance for floating point
        rtol = 1e-5
        assert abs(train_stats['batch_mean_sum_intensity'] - expected_train_stats['batch_mean_sum_intensity']) \
               / expected_train_stats['batch_mean_sum_intensity'] < rtol, \
            f"Train stats mismatch: got {train_stats['batch_mean_sum_intensity']}, " \
            f"expected {expected_train_stats['batch_mean_sum_intensity']}"

        assert abs(test_stats['batch_mean_sum_intensity'] - expected_test_stats['batch_mean_sum_intensity']) \
               / expected_test_stats['batch_mean_sum_intensity'] < rtol, \
            f"Test stats mismatch: got {test_stats['batch_mean_sum_intensity']}, " \
            f"expected {expected_test_stats['batch_mean_sum_intensity']}"

    def test_attaches_stats_with_yy_full(self):
        """Verify stats attachment works when YY_full arrays are provided."""
        from ptycho.data_preprocessing import create_ptycho_dataset
        from ptycho import params as p

        N = 64
        gridsize = 2
        p.cfg['N'] = N
        p.cfg['gridsize'] = gridsize
        p.cfg['default_probe_scale'] = 4.0

        # Set up probe in params.cfg
        # probe.get_probe expects shape (N, N, 2) where last dim is [real, imag]
        test_probe = np.ones((N, N, 2), dtype=np.float32)
        p.cfg['probe'] = test_probe

        np.random.seed(123)
        B_train = 4
        B_test = 2
        C = gridsize ** 2

        X_train = np.random.rand(B_train, N, N, C).astype(np.float32) + 0.1
        X_test = np.random.rand(B_test, N, N, C).astype(np.float32) + 0.1

        Y_I_train = np.random.rand(B_train, N, N, C).astype(np.float32)
        Y_I_test = np.random.rand(B_test, N, N, C).astype(np.float32)

        Y_phi_train = np.zeros_like(Y_I_train)
        Y_phi_test = np.zeros_like(Y_I_test)

        intensity_scale = 300.0

        # Provide actual YY_full arrays (shape varies, often (nimgs, size, size))
        YY_train_full = np.random.rand(1, 200, 200).astype(np.complex64)
        YY_test_full = np.random.rand(1, 200, 200).astype(np.complex64)

        coords_train_nominal = np.zeros((B_train, 1, 2, C), dtype=np.float32)
        coords_train_true = coords_train_nominal.copy()
        coords_test_nominal = np.zeros((B_test, 1, 2, C), dtype=np.float32)
        coords_test_true = coords_test_nominal.copy()

        ptycho_dataset = create_ptycho_dataset(
            X_train, Y_I_train, Y_phi_train, intensity_scale,
            YY_train_full, coords_train_nominal, coords_train_true,
            X_test, Y_I_test, Y_phi_test, YY_test_full,
            coords_test_nominal, coords_test_true
        )

        # Both containers should have valid stats
        assert ptycho_dataset.train_data.dataset_intensity_stats is not None
        assert ptycho_dataset.test_data.dataset_intensity_stats is not None

        # Stats should be correctly computed
        assert ptycho_dataset.train_data.dataset_intensity_stats['n_samples'] == B_train
        assert ptycho_dataset.test_data.dataset_intensity_stats['n_samples'] == B_test
