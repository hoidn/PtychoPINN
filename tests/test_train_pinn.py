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
