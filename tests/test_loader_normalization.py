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
