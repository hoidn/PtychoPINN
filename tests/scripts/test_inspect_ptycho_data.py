"""Tests for scripts/inspect_ptycho_data.py NPZ loader.

Per Phase D4f.2: Verifies that load_ptycho_data preserves dataset_intensity_stats
for proper intensity_scale calculation. Tests cover:
1. NPZ files with stored stats keys (preferred path)
2. Fallback computation from X array when keys are missing
3. Stats dictionary structure and values

See: specs/spec-ptycho-core.md §Normalization Invariants
See: docs/findings.md PINN-CHUNKED-001
"""
import numpy as np
import pytest
import tempfile
import os


class TestInspectPtychoData:
    """Test suite for scripts/inspect_ptycho_data.py load_ptycho_data function."""

    def test_load_preserves_dataset_stats(self):
        """Verify load_ptycho_data preserves dataset_intensity_stats from NPZ.

        This test exercises both code paths:
        1. With stored stats keys (dataset_intensity_stats_batch_mean/_n_samples)
        2. Without stored keys (fallback to computing from X)

        Per Phase D4f.2: NPZ readers must retain dataset_intensity_stats so
        calculate_intensity_scale can use the spec-compliant dataset-derived
        formula instead of the closed-form fallback.
        """
        # Import inside test to avoid import errors during collection
        import sys
        # Add scripts directory to path if needed
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(scripts_dir))

        from inspect_ptycho_data import load_ptycho_data

        N = 32  # Small patch size for test speed
        B = 4   # Small batch

        np.random.seed(42)

        # Create minimal test data matching PtychoDataContainer.to_npz output
        X = np.random.rand(B, N, N, 1).astype(np.float32) * 0.3
        Y_I = np.random.rand(B, N, N, 1).astype(np.float32) * 0.5
        Y_phi = np.random.rand(B, N, N, 1).astype(np.float32) * 0.1
        norm_Y_I = np.array(1.0)
        YY_full = None  # Can be None
        coords_nominal = np.zeros((B, 1, 2, 1), dtype=np.float32)
        coords_true = np.zeros((B, 1, 2, 1), dtype=np.float32)
        nn_indices = np.zeros((B, 1), dtype=np.int32)
        global_offsets = np.zeros((B, 1, 2, 1), dtype=np.float32)
        local_offsets = np.zeros((B, 1, 2, 1), dtype=np.float32)
        probe = np.ones((N, N), dtype=np.complex64)

        # Define expected stats (simulate raw diffraction batch_mean)
        expected_batch_mean = 2996.0  # Realistic value
        expected_n_samples = B

        with tempfile.TemporaryDirectory() as tmpdir:
            # === Test 1: NPZ with stored stats keys (preferred path) ===
            npz_with_stats = os.path.join(tmpdir, 'data_with_stats.npz')
            np.savez(
                npz_with_stats,
                X=X,
                Y_I=Y_I,
                Y_phi=Y_phi,
                norm_Y_I=norm_Y_I,
                YY_full=YY_full,
                coords_nominal=coords_nominal,
                coords_true=coords_true,
                nn_indices=nn_indices,
                global_offsets=global_offsets,
                local_offsets=local_offsets,
                probe=probe,
                # Stored stats keys (from PtychoDataContainer.to_npz)
                dataset_intensity_stats_batch_mean=np.array(expected_batch_mean),
                dataset_intensity_stats_n_samples=np.array(expected_n_samples),
            )

            container_with_stats = load_ptycho_data(npz_with_stats)

            # Verify stats were loaded from stored keys
            assert hasattr(container_with_stats, 'dataset_intensity_stats'), \
                "Container should have dataset_intensity_stats attribute"
            assert container_with_stats.dataset_intensity_stats is not None, \
                "dataset_intensity_stats should not be None"

            stats = container_with_stats.dataset_intensity_stats
            assert 'batch_mean_sum_intensity' in stats
            assert 'n_samples' in stats

            # Verify values match stored keys
            assert abs(stats['batch_mean_sum_intensity'] - expected_batch_mean) < 0.01, \
                f"batch_mean mismatch: got {stats['batch_mean_sum_intensity']}, expected {expected_batch_mean}"
            assert stats['n_samples'] == expected_n_samples, \
                f"n_samples mismatch: got {stats['n_samples']}, expected {expected_n_samples}"

            # === Test 2: NPZ without stored stats keys (fallback path) ===
            npz_without_stats = os.path.join(tmpdir, 'data_without_stats.npz')
            np.savez(
                npz_without_stats,
                X=X,
                Y_I=Y_I,
                Y_phi=Y_phi,
                norm_Y_I=norm_Y_I,
                YY_full=YY_full,
                coords_nominal=coords_nominal,
                coords_true=coords_true,
                nn_indices=nn_indices,
                global_offsets=global_offsets,
                local_offsets=local_offsets,
                probe=probe,
                # No stored stats keys - should trigger fallback
            )

            container_without_stats = load_ptycho_data(npz_without_stats)

            # Verify stats were computed from X array
            assert container_without_stats.dataset_intensity_stats is not None, \
                "dataset_intensity_stats should be computed from X when keys missing"

            fallback_stats = container_without_stats.dataset_intensity_stats
            assert fallback_stats['n_samples'] == B, \
                f"Fallback n_samples should match batch size: got {fallback_stats['n_samples']}"

            # Verify fallback stats are computed from X
            # Expected: E_batch[Σ_xy |X|²]
            X_f64 = X.astype(np.float64)
            expected_fallback = float(np.mean(np.sum(X_f64 ** 2, axis=(1, 2, 3))))

            actual_fallback = fallback_stats['batch_mean_sum_intensity']
            assert abs(actual_fallback - expected_fallback) / expected_fallback < 0.001, \
                f"Fallback batch_mean mismatch: got {actual_fallback}, expected {expected_fallback}"

    def test_load_preserves_tensor_cache_empty(self):
        """Verify load_ptycho_data does not populate _tensor_cache.

        Per PINN-CHUNKED-001: Loading data should not trigger GPU tensor
        materialization. The _tensor_cache should remain empty after load.
        """
        import sys
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(scripts_dir))

        from inspect_ptycho_data import load_ptycho_data

        N = 32
        B = 2

        np.random.seed(123)
        X = np.random.rand(B, N, N, 1).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, 'test_data.npz')
            np.savez(
                npz_path,
                X=X,
                Y_I=np.random.rand(B, N, N, 1).astype(np.float32),
                Y_phi=np.random.rand(B, N, N, 1).astype(np.float32),
                norm_Y_I=np.array(1.0),
                YY_full=None,
                coords_nominal=np.zeros((B, 1, 2, 1), dtype=np.float32),
                coords_true=np.zeros((B, 1, 2, 1), dtype=np.float32),
                nn_indices=np.zeros((B, 1), dtype=np.int32),
                global_offsets=np.zeros((B, 1, 2, 1), dtype=np.float32),
                local_offsets=np.zeros((B, 1, 2, 1), dtype=np.float32),
                probe=np.ones((N, N), dtype=np.complex64),
            )

            container = load_ptycho_data(npz_path)

            # Verify _tensor_cache is empty (no GPU materialization during load)
            assert hasattr(container, '_tensor_cache'), \
                "Container should have _tensor_cache attribute"
            assert len(container._tensor_cache) == 0, \
                f"_tensor_cache should be empty after load, but has {len(container._tensor_cache)} entries"
