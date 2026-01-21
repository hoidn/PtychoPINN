"""Regression tests for normalize_data() fixed L2 norm normalization.

Tests verify that normalize_data() in ptycho.raw_data and ptycho.loader
computes the fixed L2 norm normalization:

    norm_factor = sqrt((N/2)² / E_batch[Σ_xy |X|²])
    X_normalized = norm_factor * X

This ensures post-normalization mean-sum-of-squares = (N/2)², providing
consistent input scale for the model regardless of original photon counts.

NOTE: This is DIFFERENT from the intensity_scale formula in spec-ptycho-core.md
§Normalization Invariants. The normalize_data() function normalizes to a fixed
L2 target, while intensity_scale handles nphotons-dependent scaling for loss.
"""
import numpy as np


class TestNormalizeData:
    """Test suite for normalize_data() fixed L2 norm computation."""

    def test_fixed_l2_norm_raw_data(self):
        """Verify raw_data.normalize_data() normalizes to (N/2)² target."""
        from ptycho.raw_data import normalize_data

        N = 64
        B = 4  # batch size

        # Create synthetic diffraction data (rank-3 for raw_data)
        np.random.seed(42)
        X = np.random.rand(B, N, N).astype(np.float32) * 0.1

        # Compute expected norm factor manually
        # norm = sqrt((N/2)² / mean(sum(X², axes=[1,2])))
        sum_x2 = np.sum(X ** 2, axis=(1, 2))  # shape (B,)
        mean_sum_x2 = np.mean(sum_x2)
        expected_norm = np.sqrt((N / 2) ** 2 / mean_sum_x2)

        # Create dataset dict
        dset = {'diffraction': X}

        # Call normalize_data
        X_normalized = normalize_data(dset, N)

        # Verify: normalized data = norm * original data
        actual_norm = float(np.mean(X_normalized / X))

        # Allow 1% relative tolerance
        assert abs(actual_norm - expected_norm) / expected_norm < 0.01, \
            f"Norm mismatch: got {actual_norm}, expected {expected_norm}"

        # Verify post-normalization mean-sum-of-squares ≈ (N/2)²
        post_norm_sum_x2 = np.sum(X_normalized ** 2, axis=(1, 2))
        post_norm_mean = np.mean(post_norm_sum_x2)
        expected_target = (N / 2) ** 2

        assert abs(post_norm_mean - expected_target) / expected_target < 0.01, \
            f"Post-norm target mismatch: got {post_norm_mean}, expected {expected_target}"

    def test_fixed_l2_norm_loader(self):
        """Verify loader.normalize_data() normalizes to (N/2)² target."""
        from ptycho.loader import normalize_data

        N = 64
        B = 4

        np.random.seed(42)
        X = np.random.rand(B, N, N).astype(np.float32) * 0.1

        sum_x2 = np.sum(X ** 2, axis=(1, 2))
        mean_sum_x2 = np.mean(sum_x2)
        expected_norm = np.sqrt((N / 2) ** 2 / mean_sum_x2)

        dset = {'diffraction': X}
        X_normalized = normalize_data(dset, N)

        actual_norm = float(np.mean(X_normalized / X))

        assert abs(actual_norm - expected_norm) / expected_norm < 0.01, \
            f"Norm mismatch: got {actual_norm}, expected {expected_norm}"

    def test_varying_input_scales(self):
        """Verify normalization works for different input amplitudes."""
        from ptycho.raw_data import normalize_data

        N = 64
        B = 4

        # Test with different input scales
        for scale_factor in [0.001, 0.1, 1.0, 10.0, 100.0]:
            np.random.seed(123)
            X = np.random.rand(B, N, N).astype(np.float32) * scale_factor

            dset = {'diffraction': X}
            X_normalized = normalize_data(dset, N)

            # Post-normalization target should always be (N/2)²
            post_norm_sum_x2 = np.sum(X_normalized ** 2, axis=(1, 2))
            post_norm_mean = np.mean(post_norm_sum_x2)
            expected_target = (N / 2) ** 2

            assert abs(post_norm_mean - expected_target) / expected_target < 0.01, \
                f"Scale {scale_factor}: post-norm={post_norm_mean}, expected={expected_target}"

    def test_raw_data_and_loader_consistency(self):
        """Verify raw_data and loader normalize_data produce identical results."""
        from ptycho.raw_data import normalize_data as normalize_raw
        from ptycho.loader import normalize_data as normalize_loader

        N = 64
        B = 8

        np.random.seed(456)
        X = np.random.rand(B, N, N).astype(np.float32) * 0.2

        dset = {'diffraction': X}

        X_norm_raw = normalize_raw(dset, N)
        X_norm_loader = normalize_loader(dset, N)

        # Results should be identical
        np.testing.assert_allclose(
            X_norm_raw, X_norm_loader,
            rtol=1e-6, atol=1e-12,
            err_msg="raw_data and loader normalize_data produce different results"
        )

    def test_formula_compliance(self):
        """Verify the formula matches the documented L2 normalization exactly.

        Formula: norm = sqrt((N/2)² / E_batch[Σ_xy |X|²])
        """
        from ptycho.raw_data import normalize_data

        N = 32
        B = 2

        # Create simple data: constant value for easy calculation
        val = 0.01
        X = np.full((B, N, N), val, dtype=np.float32)

        # Manual calculation per formula:
        # Σ_xy |X|² for one sample = N*N * val²
        sum_per_sample = N * N * (val ** 2)
        # E_batch[...] = mean over batch = sum_per_sample (constant across batch)
        E_batch = sum_per_sample
        # norm = sqrt((N/2)² / E_batch)
        expected_norm = np.sqrt((N / 2) ** 2 / E_batch)

        dset = {'diffraction': X}
        X_normalized = normalize_data(dset, N)

        # Actual norm
        actual_norm = X_normalized[0, 0, 0] / X[0, 0, 0]

        # Should match exactly (within float precision)
        assert abs(actual_norm - expected_norm) < 1e-4, \
            f"Formula violation: got norm={actual_norm}, expected={expected_norm}"

    def test_nphotons_not_used(self):
        """Verify normalize_data does NOT depend on nphotons parameter.

        This is important because the L2 normalization is intentionally
        independent of photon count - that's handled by intensity_scale.
        """
        from ptycho import params as p
        from ptycho.raw_data import normalize_data

        N = 64
        B = 4

        np.random.seed(789)
        X = np.random.rand(B, N, N).astype(np.float32) * 0.1
        dset = {'diffraction': X}

        # Save original nphotons
        original_nphotons = p.cfg.get('nphotons')

        try:
            # Test with different nphotons values
            p.cfg['nphotons'] = 1e6
            X_norm_1e6 = normalize_data(dset, N)

            p.cfg['nphotons'] = 1e12
            X_norm_1e12 = normalize_data(dset, N)

            # Results should be identical regardless of nphotons
            np.testing.assert_allclose(
                X_norm_1e6, X_norm_1e12,
                rtol=1e-6, atol=1e-12,
                err_msg="normalize_data should NOT depend on nphotons"
            )

        finally:
            if original_nphotons is not None:
                p.cfg['nphotons'] = original_nphotons

    def test_dataset_stats_attachment(self):
        """Verify loader.load() attaches pre-normalization diffraction stats.

        Per Phase D4f: loader must compute E_batch[Σ_xy |X|²] from raw 'diffraction'
        BEFORE calling normalize_data() and attach it to the container as
        dataset_intensity_stats. This allows calculate_intensity_scale() to use
        the spec-compliant dataset-derived formula instead of the closed-form fallback.

        See: specs/spec-ptycho-core.md §Normalization Invariants
        """
        from ptycho.loader import load

        N = 64
        B = 8

        # Create synthetic grouped data dict with both 'diffraction' and 'X_full'
        np.random.seed(42)
        raw_diffraction = np.random.rand(B, N, N, 1).astype(np.float32) * 0.5

        # Manually compute expected batch_mean_sum_intensity from raw diffraction
        raw_f64 = raw_diffraction.astype(np.float64)
        sum_intensity = np.sum(raw_f64 ** 2, axis=(1, 2, 3))  # shape (B,)
        expected_batch_mean = float(np.mean(sum_intensity))

        # Create X_full (normalized version - distinct values to verify we use raw)
        X_full = raw_diffraction * 10.0  # Different scale to ensure we detect which is used

        # Build grouped data dict matching raw_data.generate_grouped_data() output
        coords_offsets = np.zeros((B, 1, 2, 1), dtype=np.float32)
        coords_relative = np.zeros((B, 1, 2, 1), dtype=np.float32)
        nn_indices = np.zeros((B, 1), dtype=np.int32)

        dset = {
            'diffraction': raw_diffraction,
            'X_full': X_full,
            'Y': None,
            'coords_offsets': coords_offsets,
            'coords_start_offsets': coords_offsets,
            'coords_relative': coords_relative,
            'coords_start_relative': coords_relative,
            'nn_indices': nn_indices,
        }

        # Create a simple probe
        probe = np.ones((N, N), dtype=np.complex64)

        # Call loader.load without splitting
        container = load(lambda: dset, probe, which=None, create_split=False)

        # Verify dataset_intensity_stats is attached
        assert hasattr(container, 'dataset_intensity_stats'), \
            "Container should have dataset_intensity_stats attribute"
        assert container.dataset_intensity_stats is not None, \
            "dataset_intensity_stats should not be None when 'diffraction' key present"

        # Verify the computed value matches expected
        actual_batch_mean = container.dataset_intensity_stats.get('batch_mean_sum_intensity', 0.0)
        assert abs(actual_batch_mean - expected_batch_mean) / expected_batch_mean < 0.001, \
            f"batch_mean_sum_intensity mismatch: got {actual_batch_mean}, expected {expected_batch_mean}"

        # Verify n_samples is correct
        actual_n_samples = container.dataset_intensity_stats.get('n_samples', 0)
        assert actual_n_samples == B, \
            f"n_samples mismatch: got {actual_n_samples}, expected {B}"

    def test_dataset_stats_attachment_with_split(self):
        """Verify loader.load() recomputes stats correctly for train/test splits.

        When create_split=True, the stats should be recomputed for the specific
        split (train or test), not the full dataset.
        """
        from ptycho.loader import load

        N = 64
        B = 10  # Use 10 samples for easy split math
        train_frac = 0.6  # 6 train, 4 test

        np.random.seed(123)
        raw_diffraction = np.random.rand(B, N, N, 1).astype(np.float32) * 0.5

        # Compute expected stats for train split (first 6 samples)
        n_train = int(B * train_frac)
        train_diff = raw_diffraction[:n_train].astype(np.float64)
        expected_train_batch_mean = float(np.mean(np.sum(train_diff ** 2, axis=(1, 2, 3))))

        # Compute expected stats for test split (last 4 samples)
        test_diff = raw_diffraction[n_train:].astype(np.float64)
        expected_test_batch_mean = float(np.mean(np.sum(test_diff ** 2, axis=(1, 2, 3))))

        # Build grouped data dict
        X_full = raw_diffraction * 10.0
        coords_offsets = np.zeros((B, 1, 2, 1), dtype=np.float32)
        coords_relative = np.zeros((B, 1, 2, 1), dtype=np.float32)
        nn_indices = np.zeros((B, 1), dtype=np.int32)

        dset = {
            'diffraction': raw_diffraction,
            'X_full': X_full,
            'Y': None,
            'coords_offsets': coords_offsets,
            'coords_start_offsets': coords_offsets,
            'coords_relative': coords_relative,
            'coords_start_relative': coords_relative,
            'nn_indices': nn_indices,
        }

        probe = np.ones((N, N), dtype=np.complex64)

        # Load train split
        train_container = load(lambda: (dset, train_frac), probe, which='train', create_split=True)
        assert train_container.dataset_intensity_stats is not None
        train_actual = train_container.dataset_intensity_stats['batch_mean_sum_intensity']
        assert abs(train_actual - expected_train_batch_mean) / expected_train_batch_mean < 0.001, \
            f"Train batch_mean mismatch: got {train_actual}, expected {expected_train_batch_mean}"
        assert train_container.dataset_intensity_stats['n_samples'] == n_train

        # Load test split
        test_container = load(lambda: (dset, train_frac), probe, which='test', create_split=True)
        assert test_container.dataset_intensity_stats is not None
        test_actual = test_container.dataset_intensity_stats['batch_mean_sum_intensity']
        assert abs(test_actual - expected_test_batch_mean) / expected_test_batch_mean < 0.001, \
            f"Test batch_mean mismatch: got {test_actual}, expected {expected_test_batch_mean}"
        assert test_container.dataset_intensity_stats['n_samples'] == B - n_train
