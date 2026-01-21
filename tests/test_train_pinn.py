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
