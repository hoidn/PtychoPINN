"""Tests for ptycho/train_pinn.py intensity scale calculation.

Per specs/spec-ptycho-core.md Normalization Invariants:
1) Dataset-derived (preferred): s = sqrt(nphotons / E_batch[sum_xy |X|^2])
2) Closed-form fallback: s = sqrt(nphotons) / (N/2) when batch_mean is near zero
"""

import numpy as np
import pytest
import tensorflow as tf


class StubDataContainer:
    """Stub container exposing .X for testing calculate_intensity_scale."""

    def __init__(self, X_data: np.ndarray):
        """Initialize with NumPy array that will be converted to tf.Tensor."""
        self._X = tf.constant(X_data, dtype=tf.float32)

    @property
    def X(self) -> tf.Tensor:
        return self._X


class LazyStubDataContainer:
    """Stub mimicking PtychoDataContainer with _X_np, _tensor_cache, and lazy .X.

    This stub tests that calculate_intensity_scale() uses _X_np and does NOT
    populate _tensor_cache (which would trigger GPU memory allocation).

    See: docs/findings.md PINN-CHUNKED-001 for lazy-container requirements.
    """

    def __init__(self, X_data: np.ndarray):
        """Initialize with NumPy array stored in _X_np; _tensor_cache starts empty."""
        self._X_np = X_data
        self._tensor_cache = {}

    @property
    def X(self) -> tf.Tensor:
        """Lazy property that populates _tensor_cache on first access."""
        if 'X' not in self._tensor_cache:
            self._tensor_cache['X'] = tf.convert_to_tensor(self._X_np, dtype=tf.float32)
        return self._tensor_cache['X']


class TestIntensityScale:
    """Test suite for train_pinn.calculate_intensity_scale per spec-ptycho-core.md."""

    def test_dataset_derived_path(self):
        """Verify dataset-derived formula: s = sqrt(nphotons / E_batch[sum_xy |X|^2])."""
        from ptycho import params
        from ptycho.train_pinn import calculate_intensity_scale

        # Small deterministic tensor: shape (2, 4, 4, 1)
        np.random.seed(42)
        X_data = np.random.rand(2, 4, 4, 1).astype(np.float32) + 0.1  # Ensure nonzero

        container = StubDataContainer(X_data)

        # Save original params and set test values
        original_nphotons = params.get('nphotons')
        original_N = params.get('N')
        try:
            test_nphotons = 1e6
            test_N = 64
            params.set('nphotons', test_nphotons)
            params.set('N', test_N)

            # Compute expected value using the spec formula
            X_f64 = X_data.astype(np.float64)
            sum_intensity = np.sum(X_f64 ** 2, axis=(1, 2, 3))  # shape (2,)
            batch_mean = float(np.mean(sum_intensity))
            expected_scale = float(np.sqrt(test_nphotons / batch_mean))

            # Call the function
            actual_scale = calculate_intensity_scale(container)

            # Verify the result matches the dataset-derived formula
            assert actual_scale == pytest.approx(expected_scale, rel=1e-6), (
                f"Dataset-derived scale mismatch: expected {expected_scale}, got {actual_scale}"
            )

        finally:
            # Restore original params
            params.set('nphotons', original_nphotons)
            params.set('N', original_N)

    def test_fallback_path_zero_tensor(self):
        """Verify fallback engages when batch_mean is near zero."""
        from ptycho import params
        from ptycho.train_pinn import calculate_intensity_scale

        # Zero tensor to trigger fallback: shape (2, 4, 4, 1)
        X_data = np.zeros((2, 4, 4, 1), dtype=np.float32)
        container = StubDataContainer(X_data)

        # Save original params and set test values
        original_nphotons = params.get('nphotons')
        original_N = params.get('N')
        try:
            test_nphotons = 1e6
            test_N = 64
            params.set('nphotons', test_nphotons)
            params.set('N', test_N)

            # Expected fallback: s = sqrt(nphotons) / (N/2)
            expected_fallback = float(np.sqrt(test_nphotons) / (test_N / 2))

            # Call the function
            actual_scale = calculate_intensity_scale(container)

            # Verify fallback path was used
            assert actual_scale == pytest.approx(expected_fallback, rel=1e-6), (
                f"Fallback scale mismatch: expected {expected_fallback}, got {actual_scale}"
            )

        finally:
            # Restore original params
            params.set('nphotons', original_nphotons)
            params.set('N', original_N)

    def test_rank3_tensor_handling(self):
        """Verify function handles rank-3 tensors (B, N, N) without channel dim."""
        from ptycho import params
        from ptycho.train_pinn import calculate_intensity_scale

        # Rank-3 tensor: shape (2, 4, 4) - no channel dimension
        np.random.seed(123)
        X_data = np.random.rand(2, 4, 4).astype(np.float32) + 0.1

        container = StubDataContainer(X_data)

        original_nphotons = params.get('nphotons')
        original_N = params.get('N')
        try:
            test_nphotons = 1e6
            test_N = 64
            params.set('nphotons', test_nphotons)
            params.set('N', test_N)

            # Expected: sum over axes (1, 2) for rank-3
            X_f64 = X_data.astype(np.float64)
            sum_intensity = np.sum(X_f64 ** 2, axis=(1, 2))  # shape (2,)
            batch_mean = float(np.mean(sum_intensity))
            expected_scale = float(np.sqrt(test_nphotons / batch_mean))

            actual_scale = calculate_intensity_scale(container)

            assert actual_scale == pytest.approx(expected_scale, rel=1e-6), (
                f"Rank-3 tensor scale mismatch: expected {expected_scale}, got {actual_scale}"
            )

        finally:
            params.set('nphotons', original_nphotons)
            params.set('N', original_N)

    def test_multi_channel_handling(self):
        """Verify multi-channel tensors (C > 1) are handled correctly."""
        from ptycho import params
        from ptycho.train_pinn import calculate_intensity_scale

        # Multi-channel tensor: shape (2, 4, 4, 4) - gridsize=2 means C=4
        np.random.seed(456)
        X_data = np.random.rand(2, 4, 4, 4).astype(np.float32) + 0.1

        container = StubDataContainer(X_data)

        original_nphotons = params.get('nphotons')
        original_N = params.get('N')
        try:
            test_nphotons = 1e6
            test_N = 64
            params.set('nphotons', test_nphotons)
            params.set('N', test_N)

            # Expected: sum over axes (1, 2, 3) for rank-4 with C=4
            X_f64 = X_data.astype(np.float64)
            sum_intensity = np.sum(X_f64 ** 2, axis=(1, 2, 3))  # shape (2,)
            batch_mean = float(np.mean(sum_intensity))
            expected_scale = float(np.sqrt(test_nphotons / batch_mean))

            actual_scale = calculate_intensity_scale(container)

            assert actual_scale == pytest.approx(expected_scale, rel=1e-6), (
                f"Multi-channel scale mismatch: expected {expected_scale}, got {actual_scale}"
            )

        finally:
            params.set('nphotons', original_nphotons)
            params.set('N', original_N)

    def test_lazy_container_does_not_materialize(self):
        """Verify _tensor_cache stays empty when _X_np is available.

        Per PINN-CHUNKED-001: calculate_intensity_scale() must prefer _X_np
        to avoid populating _tensor_cache, which triggers GPU allocation.
        """
        from ptycho import params
        from ptycho.train_pinn import calculate_intensity_scale

        # Create lazy container with _X_np and empty _tensor_cache
        np.random.seed(789)
        X_data = np.random.rand(2, 4, 4, 1).astype(np.float32) + 0.1

        container = LazyStubDataContainer(X_data)

        # Verify _tensor_cache starts empty
        assert len(container._tensor_cache) == 0, (
            f"_tensor_cache should start empty, got {len(container._tensor_cache)} entries"
        )

        original_nphotons = params.get('nphotons')
        original_N = params.get('N')
        try:
            test_nphotons = 1e6
            test_N = 64
            params.set('nphotons', test_nphotons)
            params.set('N', test_N)

            # Compute expected value using the spec formula
            X_f64 = X_data.astype(np.float64)
            sum_intensity = np.sum(X_f64 ** 2, axis=(1, 2, 3))  # shape (2,)
            batch_mean = float(np.mean(sum_intensity))
            expected_scale = float(np.sqrt(test_nphotons / batch_mean))

            # Call the function
            actual_scale = calculate_intensity_scale(container)

            # Verify the result matches the dataset-derived formula
            assert actual_scale == pytest.approx(expected_scale, rel=1e-6), (
                f"Lazy container scale mismatch: expected {expected_scale}, got {actual_scale}"
            )

            # CRITICAL: Verify _tensor_cache is STILL empty (no .X access)
            assert len(container._tensor_cache) == 0, (
                f"_tensor_cache should remain empty after calculate_intensity_scale(), "
                f"but has {len(container._tensor_cache)} entries: {list(container._tensor_cache.keys())}. "
                f"This indicates .X was accessed, which would trigger GPU allocation."
            )

        finally:
            params.set('nphotons', original_nphotons)
            params.set('N', original_N)

    def test_uses_dataset_stats(self):
        """Verify calculate_intensity_scale() prefers dataset_intensity_stats.

        Per Phase D4f: When dataset_intensity_stats is available (computed from
        raw diffraction BEFORE normalization), calculate_intensity_scale() must
        use it instead of falling back to the normalized _X_np data.

        This is critical because _X_np is already L2-normalized to (N/2)², so
        computing stats from it degenerates to the closed-form fallback, losing
        the true dataset-derived intensity scale.

        See: specs/spec-ptycho-core.md §Normalization Invariants
        """
        from ptycho import params
        from ptycho.train_pinn import calculate_intensity_scale

        # Create two distinct batch_mean values:
        # - raw_batch_mean: what we'd get from raw diffraction (large, realistic)
        # - normalized_batch_mean: what _X_np would give (L2 normalized to (N/2)²)

        test_nphotons = 1e6
        test_N = 64
        raw_batch_mean = 2996.0  # Realistic raw diffraction stats
        normalized_batch_mean = (test_N / 2) ** 2  # 1024.0 (L2 norm target)

        # Expected scales differ significantly
        expected_raw_scale = float(np.sqrt(test_nphotons / raw_batch_mean))  # ≈577.7
        expected_normalized_scale = float(np.sqrt(test_nphotons / normalized_batch_mean))  # ≈31.25

        # Create container with dataset_intensity_stats AND _X_np
        # The function should prefer dataset_intensity_stats
        class ContainerWithStats:
            def __init__(self):
                # Normalized data that would give closed-form result
                self._X_np = np.full((2, 4, 4, 1), np.sqrt(normalized_batch_mean / 16), dtype=np.float32)
                self._tensor_cache = {}
                # Raw diffraction stats (what we want to use)
                self.dataset_intensity_stats = {
                    'batch_mean_sum_intensity': raw_batch_mean,
                    'n_samples': 2,
                }

        container = ContainerWithStats()

        original_nphotons = params.get('nphotons')
        original_N = params.get('N')
        try:
            params.set('nphotons', test_nphotons)
            params.set('N', test_N)

            actual_scale = calculate_intensity_scale(container)

            # Should match raw-derived scale, NOT normalized scale
            assert actual_scale == pytest.approx(expected_raw_scale, rel=1e-6), (
                f"calculate_intensity_scale should use dataset_intensity_stats. "
                f"Got {actual_scale}, expected {expected_raw_scale} (raw), "
                f"not {expected_normalized_scale} (normalized fallback)"
            )

            # Verify _tensor_cache is still empty (no .X access needed)
            assert len(container._tensor_cache) == 0, (
                f"_tensor_cache should remain empty when using dataset_intensity_stats"
            )

        finally:
            params.set('nphotons', original_nphotons)
            params.set('N', original_N)

    def test_uses_dataset_stats_ignores_zero_mean(self):
        """Verify calculate_intensity_scale falls back when dataset stats are near zero."""
        from ptycho import params
        from ptycho.train_pinn import calculate_intensity_scale

        test_nphotons = 1e6
        test_N = 64

        # Container with near-zero dataset stats should fall through to _X_np
        class ContainerWithZeroStats:
            def __init__(self):
                np.random.seed(42)
                self._X_np = np.random.rand(2, 4, 4, 1).astype(np.float32) + 0.1
                self._tensor_cache = {}
                # Near-zero stats (should trigger fallback)
                self.dataset_intensity_stats = {
                    'batch_mean_sum_intensity': 1e-15,
                    'n_samples': 2,
                }

        container = ContainerWithZeroStats()

        # Expected: fall back to _X_np computation
        X_f64 = container._X_np.astype(np.float64)
        sum_intensity = np.sum(X_f64 ** 2, axis=(1, 2, 3))
        batch_mean_from_X = float(np.mean(sum_intensity))
        expected_scale = float(np.sqrt(test_nphotons / batch_mean_from_X))

        original_nphotons = params.get('nphotons')
        original_N = params.get('N')
        try:
            params.set('nphotons', test_nphotons)
            params.set('N', test_N)

            actual_scale = calculate_intensity_scale(container)

            # Should match _X_np-derived scale (fallback)
            assert actual_scale == pytest.approx(expected_scale, rel=1e-6), (
                f"With near-zero dataset stats, should fall back to _X_np. "
                f"Got {actual_scale}, expected {expected_scale}"
            )

        finally:
            params.set('nphotons', original_nphotons)
            params.set('N', original_N)
