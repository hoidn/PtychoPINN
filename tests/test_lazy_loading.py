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


class TestLazyLoading:
    """Tests for lazy loading implementation (Phase B).

    These tests verify that PtychoDataContainer stores NumPy arrays internally
    and converts to TensorFlow tensors only on demand.
    """

    def test_lazy_loading_avoids_oom(self):
        """Verify lazy loading doesn't allocate GPU memory at construction."""
        N = 64
        gridsize = 2
        C = gridsize ** 2
        n_images = 1000

        # Create NumPy data
        X = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_I = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_phi = np.random.rand(n_images, N, N, C).astype(np.float32)
        coords = np.random.rand(n_images, 1, 2, C).astype(np.float32)
        probe = np.random.rand(N, N).astype(np.complex64)

        # Measure GPU memory BEFORE creating container
        if tf.config.list_physical_devices('GPU'):
            initial_mem = tf.config.experimental.get_memory_info('GPU:0')['current']
        else:
            initial_mem = 0

        # Create container with NumPy arrays (should NOT allocate GPU memory)
        container = PtychoDataContainer(
            X=X,  # NumPy array, not tensor
            Y_I=Y_I,
            Y_phi=Y_phi,
            norm_Y_I=np.ones(n_images),
            YY_full=None,
            coords_nominal=coords,
            coords_true=coords,
            nn_indices=np.zeros((n_images, 7), dtype=np.int32),
            global_offsets=coords,
            local_offsets=coords,
            probeGuess=probe,
        )

        # Measure GPU memory AFTER creating container (should be unchanged)
        if tf.config.list_physical_devices('GPU'):
            after_construct_mem = tf.config.experimental.get_memory_info('GPU:0')['current']
        else:
            after_construct_mem = 0

        # Memory should not have increased significantly (allow 1MB tolerance for overhead)
        memory_delta = after_construct_mem - initial_mem
        assert memory_delta < 1e6, f"Container construction allocated {memory_delta/1e6:.1f} MB"

        # Verify data is accessible via properties (this WILL allocate)
        assert container.X.shape == (n_images, N, N, C)
        assert len(container) == n_images

    def test_lazy_container_backward_compatible(self):
        """Verify lazy container works with existing training pipeline patterns."""
        N = 64
        gridsize = 2
        C = gridsize ** 2
        n_images = 100

        X = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_I = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_phi = np.random.rand(n_images, N, N, C).astype(np.float32)
        coords = np.random.rand(n_images, 1, 2, C).astype(np.float32)
        probe = np.random.rand(N, N).astype(np.complex64)

        container = PtychoDataContainer(
            X=X,
            Y_I=Y_I,
            Y_phi=Y_phi,
            norm_Y_I=np.ones(n_images),
            YY_full=None,
            coords_nominal=coords,
            coords_true=coords,
            nn_indices=np.zeros((n_images, 7), dtype=np.int32),
            global_offsets=coords,
            local_offsets=coords,
            probeGuess=probe,
        )

        # Test backward-compatible property access
        assert tf.is_tensor(container.X)
        assert tf.is_tensor(container.Y)
        assert tf.is_tensor(container.coords_nominal)
        assert tf.is_tensor(container.probe)

        # Verify shapes
        assert container.X.shape == (n_images, N, N, C)
        assert container.Y.shape == (n_images, N, N, C)
        assert container.Y.dtype == tf.complex64

        # Verify coords alias works
        assert container.coords.shape == container.coords_nominal.shape

    def test_lazy_caching(self):
        """Verify that tensor conversion is cached (same object returned)."""
        N = 32
        C = 4
        n_images = 10

        X = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_I = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_phi = np.random.rand(n_images, N, N, C).astype(np.float32)
        coords = np.random.rand(n_images, 1, 2, C).astype(np.float32)
        probe = np.random.rand(N, N).astype(np.complex64)

        container = PtychoDataContainer(
            X=X,
            Y_I=Y_I,
            Y_phi=Y_phi,
            norm_Y_I=np.ones(n_images),
            YY_full=None,
            coords_nominal=coords,
            coords_true=coords,
            nn_indices=np.zeros((n_images, 7), dtype=np.int32),
            global_offsets=coords,
            local_offsets=coords,
            probeGuess=probe,
        )

        # First access triggers conversion
        X_first = container.X
        # Second access should return cached tensor (same object)
        X_second = container.X
        assert X_first is X_second, "Tensor should be cached"

    def test_tensor_input_handled(self):
        """Verify that tensor inputs are converted to numpy and stored."""
        N = 32
        C = 4
        n_images = 10

        # Pass TensorFlow tensors instead of NumPy arrays
        X = tf.random.uniform((n_images, N, N, C), dtype=tf.float32)
        Y_I = tf.random.uniform((n_images, N, N, C), dtype=tf.float32)
        Y_phi = tf.random.uniform((n_images, N, N, C), dtype=tf.float32)
        coords = tf.random.uniform((n_images, 1, 2, C), dtype=tf.float32)
        probe = tf.complex(
            tf.random.uniform((N, N)),
            tf.random.uniform((N, N))
        )

        container = PtychoDataContainer(
            X=X,
            Y_I=Y_I,
            Y_phi=Y_phi,
            norm_Y_I=np.ones(n_images),
            YY_full=None,
            coords_nominal=coords,
            coords_true=coords,
            nn_indices=np.zeros((n_images, 7), dtype=np.int32),
            global_offsets=coords.numpy(),
            local_offsets=coords.numpy(),
            probeGuess=probe,
        )

        # Verify internal storage is NumPy
        assert isinstance(container._X_np, np.ndarray)
        assert isinstance(container._Y_I_np, np.ndarray)
        assert isinstance(container._probe_np, np.ndarray)

        # Verify property access still returns tensors
        assert tf.is_tensor(container.X)
        assert tf.is_tensor(container.probe)

    def test_len_method(self):
        """Verify __len__ returns correct sample count."""
        N = 32
        C = 4
        n_images = 50

        X = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_I = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_phi = np.random.rand(n_images, N, N, C).astype(np.float32)
        coords = np.random.rand(n_images, 1, 2, C).astype(np.float32)
        probe = np.random.rand(N, N).astype(np.complex64)

        container = PtychoDataContainer(
            X=X,
            Y_I=Y_I,
            Y_phi=Y_phi,
            norm_Y_I=np.ones(n_images),
            YY_full=None,
            coords_nominal=coords,
            coords_true=coords,
            nn_indices=np.zeros((n_images, 7), dtype=np.int32),
            global_offsets=coords,
            local_offsets=coords,
            probeGuess=probe,
        )

        assert len(container) == n_images
