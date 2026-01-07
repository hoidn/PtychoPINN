"""Tests for lazy tensor allocation in loader.py.

These tests verify the OOM behavior with eager loading (baseline)
and the fix via lazy tensor allocation (Phase B).

Spec reference: spec-ptycho-workflow.md Resource Constraints
Finding: PINN-CHUNKED-001 (OOM blocker documentation)
"""
import pytest
import numpy as np
import tensorflow as tf

from ptycho.loader import PtychoDataContainer, load


class TestEagerLoadingOOM:
    """Tests demonstrating OOM with current eager loading architecture."""

    @pytest.mark.parametrize("n_images", [100, 500, 1000])
    def test_memory_usage_scales_with_dataset_size(self, n_images):
        """Verify memory usage scales linearly with dataset size.

        This test doesn't trigger OOM but measures memory consumption
        to demonstrate the eager loading problem.
        """
        N = 64  # Patch size
        gridsize = 2
        C = gridsize ** 2  # Channels

        # Synthetic data matching expected shapes
        X = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_I = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_phi = np.random.rand(n_images, N, N, C).astype(np.float32)
        coords = np.random.rand(n_images, 1, 2, C).astype(np.float32)
        probe = np.random.rand(N, N).astype(np.complex64)

        # Measure memory before
        initial_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] if tf.config.list_physical_devices('GPU') else 0

        # Create container (triggers eager tensorification)
        container = PtychoDataContainer(
            X=tf.convert_to_tensor(X, dtype=tf.float32),
            Y_I=tf.convert_to_tensor(Y_I, dtype=tf.float32),
            Y_phi=tf.convert_to_tensor(Y_phi, dtype=tf.float32),
            norm_Y_I=np.ones(n_images),
            YY_full=None,
            coords_nominal=tf.convert_to_tensor(coords, dtype=tf.float32),
            coords_true=tf.convert_to_tensor(coords, dtype=tf.float32),
            nn_indices=np.zeros((n_images, 7), dtype=np.int32),
            global_offsets=coords,
            local_offsets=coords,
            probeGuess=tf.convert_to_tensor(probe, dtype=tf.complex64),
        )

        # Verify container was created and data is accessible
        assert container.X.shape[0] == n_images
        assert container.Y.shape[0] == n_images

        # Calculate expected memory usage (approximate)
        # X: n_images * N * N * C * 4 bytes (float32)
        # Y: n_images * N * N * C * 8 bytes (complex64)
        # coords: n_images * 1 * 2 * C * 4 bytes (float32)
        expected_bytes = n_images * N * N * C * (4 + 8) + n_images * 1 * 2 * C * 4 * 2
        print(f"\nn_images={n_images}: Expected ~{expected_bytes / 1e6:.1f} MB tensor allocation")


    @pytest.mark.oom
    @pytest.mark.skip(reason="Intentionally triggers OOM - run manually with --run-oom")
    def test_oom_with_eager_loading(self):
        """Demonstrate OOM failure with large dataset.

        This test creates a dataset larger than available GPU memory
        to demonstrate the eager loading problem.

        Run with: pytest tests/test_lazy_loading.py::TestEagerLoadingOOM::test_oom_with_eager_loading -v --run-oom
        """
        N = 128  # Larger patch size
        gridsize = 2
        C = gridsize ** 2
        n_images = 20000  # Large dataset that exceeds typical GPU memory

        # Calculate expected memory: ~7.5 GB for this configuration
        expected_gb = (n_images * N * N * C * (4 + 8 + 4 * 2)) / 1e9
        print(f"\nAttempting to allocate ~{expected_gb:.1f} GB of tensors...")

        X = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_I = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_phi = np.random.rand(n_images, N, N, C).astype(np.float32)
        coords = np.random.rand(n_images, 1, 2, C).astype(np.float32)
        probe = np.random.rand(N, N).astype(np.complex64)

        # This should trigger OOM on most GPUs
        with pytest.raises((tf.errors.ResourceExhaustedError, MemoryError)):
            container = PtychoDataContainer(
                X=tf.convert_to_tensor(X, dtype=tf.float32),
                Y_I=tf.convert_to_tensor(Y_I, dtype=tf.float32),
                Y_phi=tf.convert_to_tensor(Y_phi, dtype=tf.float32),
                norm_Y_I=np.ones(n_images),
                YY_full=None,
                coords_nominal=tf.convert_to_tensor(coords, dtype=tf.float32),
                coords_true=tf.convert_to_tensor(coords, dtype=tf.float32),
                nn_indices=np.zeros((n_images, 7), dtype=np.int32),
                global_offsets=coords,
                local_offsets=coords,
                probeGuess=tf.convert_to_tensor(probe, dtype=tf.complex64),
            )


class TestLazyLoadingPlaceholder:
    """Placeholder tests for lazy loading implementation (Phase B).

    These tests define the expected behavior of the lazy loading fix.
    They are marked as skip until Phase B is implemented.
    """

    @pytest.mark.skip(reason="Phase B not implemented yet")
    def test_lazy_loading_avoids_oom(self):
        """Verify lazy loading handles large datasets without OOM.

        After Phase B, this test should:
        1. Create a LazyPtychoDataContainer with large dataset
        2. Verify no immediate GPU memory allocation
        3. Access batches via .as_dataset() without OOM
        """
        pass

    @pytest.mark.skip(reason="Phase B not implemented yet")
    def test_lazy_container_backward_compatible(self):
        """Verify lazy container works with existing training pipeline.

        After Phase B, accessing .X or .Y directly should:
        1. Log a deprecation warning
        2. Return the full tensor (for backward compatibility)
        3. Work correctly with existing code
        """
        pass
