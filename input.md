Mode: Implementation
Focus: FEAT-LAZY-LOADING-001 — Implement Lazy Tensor Allocation in loader.py
Branch: feature/torchapi-newprompt-2
Mapped tests: pytest tests/test_lazy_loading.py::test_oom_with_eager_loading -v (target test — to be created this loop)
Artifacts: plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T210000Z/

## Summary

Create a test that reproduces the OOM failure caused by eager `tf.convert_to_tensor()` calls in `loader.py`. This establishes the baseline failure mode that Phase B will fix.

## Context

**Problem:** `PtychoDataContainer` and the `load()` function immediately convert all input arrays to TensorFlow tensors via `tf.convert_to_tensor()`. This forces the entire dataset into GPU memory at once, causing OOM errors for large datasets (e.g., fly64 dense study with 20k+ images).

**Root Cause Location:** `ptycho/loader.py:309-311,325`:
```python
X = tf.convert_to_tensor(X_full_split, dtype=tf.float32)
coords_nominal = tf.convert_to_tensor(coords_nominal, dtype=tf.float32)
coords_true = tf.convert_to_tensor(coords_true, dtype=tf.float32)
# ...
Y = tf.convert_to_tensor(Y_split, dtype=tf.complex64)
```

**Finding:** PINN-CHUNKED-001 documents that chunked inference is blocked by this eager loading architecture.

## Do Now

### A1: Create OOM Reproduction Test

1. **Create test file** `tests/test_lazy_loading.py`:

```python
"""Tests for lazy tensor allocation in loader.py.

These tests verify the OOM behavior with eager loading (baseline)
and the fix via lazy tensor allocation (Phase B).
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
```

2. **Add pytest marker** for OOM tests in `conftest.py` (if not already present):

```python
# Add to tests/conftest.py (append to existing markers if any)
def pytest_addoption(parser):
    parser.addoption(
        "--run-oom",
        action="store_true",
        default=False,
        help="Run OOM tests that intentionally exhaust memory",
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "oom: mark test as OOM-triggering")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-oom"):
        skip_oom = pytest.mark.skip(reason="Need --run-oom option to run")
        for item in items:
            if "oom" in item.keywords:
                item.add_marker(skip_oom)
```

### A2: Run the Memory Scaling Test

```bash
# Run memory scaling test (safe, doesn't trigger OOM)
pytest tests/test_lazy_loading.py::TestEagerLoadingOOM::test_memory_usage_scales_with_dataset_size -v 2>&1 | tee plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T210000Z/pytest_memory_scaling.log
```

### A3: Verify Test Registry

```bash
# Verify test collection works
pytest --collect-only tests/test_lazy_loading.py 2>&1 | tee plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T210000Z/pytest_collect.log
```

## How-To Map

```bash
# Artifacts directory
ARTIFACTS=plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T210000Z

# Run memory scaling tests (safe)
pytest tests/test_lazy_loading.py::TestEagerLoadingOOM::test_memory_usage_scales_with_dataset_size -v

# Verify collection
pytest --collect-only tests/test_lazy_loading.py

# Optional: Run OOM test (WARNING: may crash/freeze system)
# pytest tests/test_lazy_loading.py::TestEagerLoadingOOM::test_oom_with_eager_loading -v --run-oom
```

## Pitfalls To Avoid

1. **DO NOT** run `test_oom_with_eager_loading` without explicit user consent — it WILL exhaust GPU/system memory
2. **DO NOT** modify `loader.py` in this phase — we are only creating the reproduction test
3. **DO** ensure the test imports work correctly with the existing module structure
4. **DO** use synthetic data that matches the real data shapes from `loader.py:126-138`
5. **DO** check if `conftest.py` already has pytest options/markers before adding duplicates
6. **Environment Freeze:** Do not install packages or modify environment

## If Blocked

1. If test imports fail: Check `ptycho/loader.py` imports are accessible from `tests/`
2. If GPU not available: Tests should fall back gracefully to CPU memory measurement
3. If conftest conflict: Merge with existing markers rather than overwriting
4. Record blocker in `$ARTIFACTS/blocked_<timestamp>.md` with error signature

## Findings Applied

- **PINN-CHUNKED-001:** This initiative directly addresses the OOM blocker documented there
- **DATA-001:** Test data should match NPZ data contract shapes (though synthetic)
- **CONFIG-001:** Not directly relevant for this test phase (no params.cfg usage)

## Pointers

- Implementation plan: `plans/active/FEAT-LAZY-LOADING-001/implementation.md`
- Eager loading code: `ptycho/loader.py:309-311,325` (tf.convert_to_tensor calls)
- Container class: `ptycho/loader.py:97-158` (PtychoDataContainer)
- PINN-CHUNKED-001 finding: `docs/findings.md` (OOM blocker documentation)
- fix_plan: `docs/fix_plan.md` (FEAT-LAZY-LOADING-001 entry)

## Exit Criteria

1. `tests/test_lazy_loading.py` created with `TestEagerLoadingOOM` class
2. `test_memory_usage_scales_with_dataset_size` runs and prints memory estimates
3. `test_oom_with_eager_loading` exists (skipped by default)
4. pytest collection shows new tests without errors
5. Ledger updated with results

## Next Up

- Phase B: Implement lazy container architecture (modify `PtychoDataContainer.__init__` to store NumPy, add `.as_dataset()` method)
