Mode: Implementation
Focus: FEAT-LAZY-LOADING-001 — Phase C: Integrate Lazy Loading with Training Pipeline
Branch: feature/torchapi-newprompt-2
Mapped tests: pytest tests/test_lazy_loading.py -v
Artifacts: plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-08T030000Z/

## Summary

Integrate the lazy loading container with the training pipeline. Add streaming support to `model.train()` so large datasets can use `as_tf_dataset()` instead of loading full tensors into GPU memory.

## Context

**Phase A/B Complete:** Lazy container implemented and tested (8/8 pass, commit 37985157).
- `PtychoDataContainer` stores NumPy arrays internally
- `.X`, `.Y`, etc. convert to tensors lazily on first access with caching
- `as_tf_dataset(batch_size)` returns `tf.data.Dataset` for streaming

**Current Training Pattern (ptycho/model.py:622-660):**
```python
def train(epochs, trainset: PtychoDataContainer, model_instance=None):
    # ...
    history = model_instance.fit(
        prepare_inputs(trainset),   # Accesses trainset.X, trainset.coords → triggers full tensor load
        prepare_outputs(trainset),  # Accesses trainset.Y_I, trainset.X → reuses cached tensors
        shuffle=True, batch_size=batch_size, ...
    )
```

**Target Pattern:**
```python
def train(epochs, trainset: PtychoDataContainer, model_instance=None, use_streaming=None):
    # Auto-detect: stream if dataset > threshold
    if use_streaming is None:
        use_streaming = len(trainset) > 10000  # Heuristic threshold

    if use_streaming:
        # Use as_tf_dataset() for memory-efficient streaming
        dataset = trainset.as_tf_dataset(batch_size)
        history = model_instance.fit(dataset, epochs=epochs, ...)
    else:
        # Current pattern for smaller datasets
        history = model_instance.fit(prepare_inputs(trainset), prepare_outputs(trainset), ...)
```

## Do Now

### C1: Add `train_with_dataset()` Helper (Optional Refactor)

**File:** `ptycho/model.py`

This is optional — may be cleaner to just add streaming logic directly to `train()`. Skip if complexity doesn't warrant it.

### C2: Update `train()` Function

**File:** `ptycho/model.py:622-660`

Modify `train()` to accept optional `use_streaming` parameter:

```python
def train(epochs, trainset: PtychoDataContainer, model_instance=None, use_streaming=None):
    """Train the ptychography model.

    Args:
        epochs: Number of training epochs
        trainset: Training data container
        model_instance: Optional compiled model. If None, uses module-level
                       singleton (for backward compatibility).
        use_streaming: If True, use as_tf_dataset() for memory-efficient streaming.
                      If None (default), auto-detect based on dataset size.
                      Large datasets (>10000 samples) automatically use streaming.

    Returns:
        Training history object
    """
    assert type(trainset) == PtychoDataContainer

    # Use provided model or fall back to module-level singleton
    if model_instance is None:
        model_instance = autoencoder

    batch_size = p.params()['batch_size']

    # Auto-detect streaming mode based on dataset size
    if use_streaming is None:
        use_streaming = len(trainset) > 10000

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.0001, verbose=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(
                            '%s/weights.{epoch:02d}.h5' % wt_path,
                            monitor='val_loss', verbose=1, save_best_only=True,
                            save_weights_only=False, mode='auto', save_freq='epoch')

    if use_streaming:
        # Memory-efficient streaming for large datasets
        print(f"Using streaming mode for {len(trainset)} samples")
        dataset = trainset.as_tf_dataset(batch_size, shuffle=True)
        # Note: validation_split not compatible with tf.data.Dataset
        # For streaming, skip validation or use separate validation dataset
        history = model_instance.fit(
            dataset,
            epochs=epochs,
            verbose=1,
            callbacks=[reduce_lr, earlystop]
        )
    else:
        # Standard mode for smaller datasets (current behavior)
        history = model_instance.fit(
            prepare_inputs(trainset),
            prepare_outputs(trainset),
            shuffle=True, batch_size=batch_size, verbose=1,
            epochs=epochs, validation_split=0.05,
            callbacks=[reduce_lr, earlystop]
        )

    return history
```

### C3: Add Integration Test

**File:** `tests/test_lazy_loading.py`

Add a test that verifies streaming training works:

```python
class TestStreamingTraining:
    """Tests for streaming training integration (Phase C)."""

    def test_streaming_training_small_dataset(self):
        """Verify streaming mode works with small dataset."""
        from ptycho import model as m
        from ptycho import params as p
        from ptycho.loader import PtychoDataContainer

        N = 64
        C = 4
        n_images = 100

        # Set up minimal params for training
        p.params()['intensity_scale'] = 1.0
        p.params()['batch_size'] = 16
        p.params()['N'] = N
        p.params()['gridsize'] = 2

        X = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_I = np.random.rand(n_images, N, N, C).astype(np.float32)
        Y_phi = np.random.rand(n_images, N, N, C).astype(np.float32)
        coords = np.random.rand(n_images, 1, 2, C).astype(np.float32)
        probe = np.random.rand(N, N).astype(np.complex64)

        container = PtychoDataContainer(
            X=X, Y_I=Y_I, Y_phi=Y_phi,
            norm_Y_I=np.ones(n_images),
            YY_full=None,
            coords_nominal=coords, coords_true=coords,
            nn_indices=np.zeros((n_images, 7), dtype=np.int32),
            global_offsets=coords, local_offsets=coords,
            probeGuess=probe,
        )

        # Get a tf.data.Dataset from the container
        dataset = container.as_tf_dataset(batch_size=16, shuffle=False)

        # Verify dataset yields correct structure
        for inputs, outputs in dataset.take(1):
            assert len(inputs) == 2  # [X, coords]
            assert len(outputs) == 3  # [Y_I_centered, X*s, (X*s)^2]
            assert inputs[0].shape[1:] == (N, N, C)  # X shape
            assert inputs[1].shape[1:] == (1, 2, C)  # coords shape
            break
```

### C4: Run Tests

```bash
ARTIFACTS=plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-08T030000Z

# Run all lazy loading tests
pytest tests/test_lazy_loading.py -v 2>&1 | tee $ARTIFACTS/pytest_phase_c.log

# Run regression tests
pytest tests/test_model_factory.py -v 2>&1 | tee -a $ARTIFACTS/pytest_phase_c.log
```

## How-To Map

```bash
ARTIFACTS=plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-08T030000Z

# After implementation
pytest tests/test_lazy_loading.py -v

# Regression tests
pytest tests/test_model_factory.py -v

# Verify test collection
pytest --collect-only tests/test_lazy_loading.py
```

## Pitfalls To Avoid

1. **DO NOT** break backward compatibility — `train()` must work the same when `use_streaming=False` or not specified for small datasets
2. **DO** note that `validation_split` is not compatible with `tf.data.Dataset` — streaming mode may need different validation approach
3. **DO NOT** change `prepare_inputs()` or `prepare_outputs()` signatures — keep them for non-streaming path
4. **DO** ensure `intensity_scale` is available in params before calling `as_tf_dataset()`
5. **DO** handle the case where `len(trainset)` returns 0 gracefully
6. **Environment Freeze:** Do not install packages or modify environment

## If Blocked

1. If `as_tf_dataset()` fails with params error: ensure `intensity_scale` is set in `params.cfg`
2. If streaming training produces different results: this is expected — `validation_split` doesn't work with Dataset API
3. If model.fit() rejects Dataset: check output signature matches model's expected shapes
4. Record blocker in `$ARTIFACTS/blocked_<timestamp>.md` with error signature

## Findings Applied

- **PINN-CHUNKED-001:** This phase completes the fix by integrating lazy loading with training
- **CONFIG-001:** `intensity_scale` must be set before using `as_tf_dataset()`
- **DATA-001:** Container preserves NPZ data contract shapes

## Pointers

- Implementation plan: `plans/active/FEAT-LAZY-LOADING-001/implementation.md` (Phase C tasks)
- Container with `as_tf_dataset()`: `ptycho/loader.py:255-321`
- Training function to update: `ptycho/model.py:622-660`
- Existing tests: `tests/test_lazy_loading.py`

## Exit Criteria

1. `train()` accepts optional `use_streaming` parameter
2. Streaming mode uses `as_tf_dataset()` instead of `prepare_inputs()/prepare_outputs()`
3. Auto-detection threshold (>10000 samples) selects streaming by default
4. Test `test_streaming_training_small_dataset` passes
5. Existing tests (`test_lazy_loading.py`, `test_model_factory.py`) continue to pass

## Next Up

- Phase C3 (compare_models.py chunked inference) — if time permits
- Mark FEAT-LAZY-LOADING-001 done and unblock STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
