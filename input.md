Mode: Implementation
Focus: FEAT-LAZY-LOADING-001 — Phase B: Implement Lazy Tensor Allocation
Branch: feature/torchapi-newprompt-2
Mapped tests: pytest tests/test_lazy_loading.py -v
Artifacts: plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T220000Z/

## Summary

Implement lazy tensor allocation in `PtychoDataContainer` so data is stored as NumPy arrays and converted to TensorFlow tensors only on demand. This fixes the OOM blocker (PINN-CHUNKED-001).

## Context

**Phase A Complete:** Tests created and passing (3/3 memory scaling tests, OOM test skipped by default).

**Problem:** `PtychoDataContainer` eagerly converts all data to TensorFlow GPU tensors at construction time via `tf.convert_to_tensor()`. For large datasets (20k+ images), this exhausts GPU memory before training begins.

**Solution:** Store NumPy arrays internally; provide lazy conversion via:
1. **Property-based access** (`.X`, `.Y`, etc.) converts to tensor on first access and caches
2. **`as_tf_dataset(batch_size)`** returns a `tf.data.Dataset` for memory-efficient batched training

## Do Now

### B1: Modify PtychoDataContainer for Lazy Storage

**File:** `ptycho/loader.py`

Add internal storage and lazy property access:

```python
# In PtychoDataContainer.__init__, REPLACE the current assignments with:
# Store as numpy, convert lazily
self._X_np = X.numpy() if tf.is_tensor(X) else X
self._Y_I_np = Y_I.numpy() if tf.is_tensor(Y_I) else Y_I
self._Y_phi_np = Y_phi.numpy() if tf.is_tensor(Y_phi) else Y_phi
self._coords_nominal_np = coords_nominal.numpy() if tf.is_tensor(coords_nominal) else coords_nominal
self._coords_true_np = coords_true.numpy() if tf.is_tensor(coords_true) else coords_true
self._probe_np = probeGuess.numpy() if tf.is_tensor(probeGuess) else probeGuess

# Lazy cache for tensorified data
self._tensor_cache = {}

# These remain as-is (NumPy only attributes)
self.norm_Y_I = norm_Y_I
self.YY_full = YY_full
self.nn_indices = nn_indices
self.global_offsets = global_offsets
self.local_offsets = local_offsets
```

### B2: Add Lazy Property Accessors

Add property getters that convert to tensor on first access:

```python
@property
def X(self):
    """Diffraction patterns — tf.float32, shape (B, N, N, C).

    WARNING: Accessing this property loads the full tensor into GPU memory.
    For large datasets, use as_tf_dataset() instead.
    """
    if 'X' not in self._tensor_cache:
        self._tensor_cache['X'] = tf.convert_to_tensor(self._X_np, dtype=tf.float32)
    return self._tensor_cache['X']

@property
def Y_I(self):
    """Ground truth amplitude — tf.float32, shape (B, N, N, C)."""
    if 'Y_I' not in self._tensor_cache:
        self._tensor_cache['Y_I'] = tf.convert_to_tensor(self._Y_I_np, dtype=tf.float32)
    return self._tensor_cache['Y_I']

@property
def Y_phi(self):
    """Ground truth phase — tf.float32, shape (B, N, N, C)."""
    if 'Y_phi' not in self._tensor_cache:
        self._tensor_cache['Y_phi'] = tf.convert_to_tensor(self._Y_phi_np, dtype=tf.float32)
    return self._tensor_cache['Y_phi']

@property
def Y(self):
    """Combined complex ground truth — tf.complex64, shape (B, N, N, C)."""
    if 'Y' not in self._tensor_cache:
        from .tf_helper import combine_complex
        self._tensor_cache['Y'] = combine_complex(self.Y_I, self.Y_phi)
    return self._tensor_cache['Y']

@property
def coords_nominal(self):
    """Scan coordinates (channel format) — tf.float32, shape (B, 1, 2, C)."""
    if 'coords_nominal' not in self._tensor_cache:
        self._tensor_cache['coords_nominal'] = tf.convert_to_tensor(
            self._coords_nominal_np, dtype=tf.float32
        )
    return self._tensor_cache['coords_nominal']

@property
def coords(self):
    """Alias for coords_nominal (backward compatibility)."""
    return self.coords_nominal

@property
def coords_true(self):
    """True scan coordinates — tf.float32, shape (B, 1, 2, C)."""
    if 'coords_true' not in self._tensor_cache:
        self._tensor_cache['coords_true'] = tf.convert_to_tensor(
            self._coords_true_np, dtype=tf.float32
        )
    return self._tensor_cache['coords_true']

@property
def probe(self):
    """Probe function — tf.complex64, shape (N, N)."""
    if 'probe' not in self._tensor_cache:
        self._tensor_cache['probe'] = tf.convert_to_tensor(
            self._probe_np, dtype=tf.complex64
        )
    return self._tensor_cache['probe']
```

### B3: Add as_tf_dataset Method

Add streaming dataset method for memory-efficient access:

```python
def as_tf_dataset(self, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    """Create a tf.data.Dataset for memory-efficient batched access.

    This is the preferred method for large datasets as it streams data
    in batches rather than loading everything into GPU memory.

    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset (default True)

    Returns:
        tf.data.Dataset yielding (inputs, outputs) tuples compatible
        with model.fit()
    """
    from . import params as p
    from . import tf_helper as hh

    n_samples = len(self._X_np)
    intensity_scale = p.get('intensity_scale')

    def generator():
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            # Convert batch to tensors
            X_batch = tf.convert_to_tensor(
                self._X_np[batch_idx], dtype=tf.float32
            )
            coords_batch = tf.convert_to_tensor(
                self._coords_nominal_np[batch_idx], dtype=tf.float32
            )
            Y_I_batch = tf.convert_to_tensor(
                self._Y_I_np[batch_idx], dtype=tf.float32
            )

            # Prepare inputs: [X * intensity_scale, coords]
            inputs = [X_batch * intensity_scale, coords_batch]

            # Prepare outputs: [centered_Y_I[:,:,:,:1], X*s, (X*s)^2]
            Y_I_centered = hh.center_channels(Y_I_batch, coords_batch)[:, :, :, :1]
            X_scaled = intensity_scale * X_batch
            outputs = [Y_I_centered, X_scaled, X_scaled ** 2]

            yield inputs, outputs

    # Define output signature for tf.data.Dataset
    N = self._X_np.shape[1]
    C = self._X_np.shape[3]

    output_signature = (
        (
            tf.TensorSpec(shape=(None, N, N, C), dtype=tf.float32),  # X
            tf.TensorSpec(shape=(None, 1, 2, C), dtype=tf.float32),  # coords
        ),
        (
            tf.TensorSpec(shape=(None, N, N, 1), dtype=tf.float32),  # Y_I centered
            tf.TensorSpec(shape=(None, N, N, C), dtype=tf.float32),  # X*s
            tf.TensorSpec(shape=(None, N, N, C), dtype=tf.float32),  # (X*s)^2
        )
    )

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

def __len__(self):
    """Return number of samples in the container."""
    return len(self._X_np)
```

### B4: Update load() Function

Modify the `load()` function to pass NumPy arrays (not tensors) to PtychoDataContainer:

At lines 309-311 and 325, the current code does:
```python
X = tf.convert_to_tensor(X_full_split, dtype=tf.float32)
coords_nominal = tf.convert_to_tensor(coords_nominal, dtype=tf.float32)
coords_true = tf.convert_to_tensor(coords_true, dtype=tf.float32)
...
Y = tf.convert_to_tensor(Y_split, dtype=tf.complex64)
```

**Change to:** Simply pass the NumPy arrays directly without conversion. The PtychoDataContainer will handle conversion lazily.

```python
# Remove eager tensorification - pass NumPy arrays directly
# The container will convert lazily on property access
X = X_full_split.astype(np.float32)
coords_nominal = coords_nominal.astype(np.float32)
coords_true = coords_true.astype(np.float32)
...
Y_I = np.abs(Y_split).astype(np.float32)
Y_phi = np.angle(Y_split).astype(np.float32)
```

### B5: Update Tests

Update `tests/test_lazy_loading.py` to verify lazy loading works:

```python
# Replace the skip on test_lazy_loading_avoids_oom with actual test:
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
```

### B6: Run Tests and Verify

```bash
# Run all lazy loading tests
pytest tests/test_lazy_loading.py -v 2>&1 | tee plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T220000Z/pytest_phase_b.log

# Run existing model factory tests to ensure no regressions
pytest tests/test_model_factory.py -v 2>&1 | tee -a plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T220000Z/pytest_phase_b.log
```

## How-To Map

```bash
ARTIFACTS=plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T220000Z

# After implementation, run tests
pytest tests/test_lazy_loading.py -v

# Run regression tests
pytest tests/test_model_factory.py -v

# Verify test collection
pytest --collect-only tests/test_lazy_loading.py
```

## Pitfalls To Avoid

1. **DO** ensure `.numpy()` calls are conditional on `tf.is_tensor()` to handle both tensor and array inputs
2. **DO NOT** remove the combine_complex import — it's still needed for the Y property
3. **DO** preserve the `@debug` decorator on `__init__` if it exists
4. **DO** ensure `coords` property is an alias to `coords_nominal` for backward compatibility
5. **DO** test that `to_npz()` still works — it should use `self._X_np` etc. directly now
6. **DO NOT** change the `__repr__` method signature — update it to use properties or private attrs
7. **Environment Freeze:** Do not install packages or modify environment

## If Blocked

1. If existing tests fail due to type mismatches: Check that both NumPy array and TF tensor inputs are handled in `__init__`
2. If `as_tf_dataset` generator fails: Ensure `intensity_scale` is set in params.cfg before calling
3. If `center_channels` fails in generator: Import `tf_helper as hh` inside the function to avoid circular import
4. Record blocker in `$ARTIFACTS/blocked_<timestamp>.md` with error signature

## Findings Applied

- **PINN-CHUNKED-001:** This is the direct fix for the OOM blocker
- **CONFIG-001:** `intensity_scale` must be set before using `as_tf_dataset()`
- **DATA-001:** Container preserves NPZ data contract shapes

## Pointers

- Implementation plan: `plans/active/FEAT-LAZY-LOADING-001/implementation.md`
- Container class: `ptycho/loader.py:97-158`
- Eager loading lines to fix: `ptycho/loader.py:309-311,325`
- Phase A tests: `tests/test_lazy_loading.py`
- PINN-CHUNKED-001 finding: `docs/findings.md`

## Exit Criteria

1. `PtychoDataContainer` stores NumPy arrays internally (not TensorFlow tensors)
2. Properties (`.X`, `.Y`, `.coords`) convert to tensors on first access with caching
3. `as_tf_dataset(batch_size)` method returns a `tf.data.Dataset`
4. `test_lazy_loading_avoids_oom` passes (verifies no GPU allocation at construction)
5. `test_lazy_container_backward_compatible` passes (verifies property access works)
6. Existing tests (`test_model_factory.py`) continue to pass

## Next Up

- Phase C: Update training pipeline (`train_pinn.py`, `model.py`) to optionally use `as_tf_dataset()` for large datasets
