Mode: Implementation
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — G-scaled: Integration test for lazy container in compare_models
Branch: feature/torchapi-newprompt-2
Mapped tests: pytest tests/test_lazy_loading.py::TestCompareModelsChunking -v
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T210000Z/

## Summary

Run integration test to verify lazy loading integration with compare_models.py inference path. The unit test (G1-G3) verified the container API; now verify real inference works.

## Context

**Prior Work Verified:**
- `test_container_numpy_slicing_for_chunked_inference` PASSED (commit db8f15bd)
- Container exposes `_X_np`, `_coords_nominal_np` for chunked access
- NumPy slicing doesn't populate tensor cache
- Backward-compatible `.X` access works

**Lazy Loading Benefit for compare_models.py:**
1. `create_ptycho_data_container()` now uses lazy container
2. Data stored as NumPy initially
3. `.X`/`.coords_nominal` convert to tensors on first access (cached)
4. `--pinn-chunk-size` flag creates per-chunk containers (already existed)

**Current compare_models chunking strategy:**
- Line 1093-1122: Slices `RawData` per chunk, creates new container per chunk
- This works but creates many containers; lazy loading ensures each doesn't OOM at init

## Do Now

### G4: Add Integration Test for Lazy Container in Inference

**File:** `tests/test_lazy_loading.py`

Add test to `TestCompareModelsChunking`:

```python
def test_lazy_container_inference_integration(self):
    """Verify lazy container works for inference without OOM at container creation.

    This tests the integration path: create_ptycho_data_container() returns
    a lazy container that stores NumPy internally and converts to tensors
    on property access. This is the same path compare_models.py uses.
    """
    from ptycho.workflows.components import create_ptycho_data_container
    from ptycho.params import TrainingConfig
    import tensorflow as tf
    import numpy as np

    # Create minimal config
    config = TrainingConfig(
        n_images=100,
        n=64,
        gridsize=1,
        scan_stepsize=20.0,
    )

    # Create synthetic raw data (mimics what compare_models loads from NPZ)
    N = 64
    n_images = 100
    raw_data = type('RawData', (), {
        'X': np.random.rand(n_images, N, N, 1).astype(np.float32),
        'Y_I': np.random.rand(n_images, N, N, 1).astype(np.float32),
        'Y_phi': np.random.rand(n_images, N, N, 1).astype(np.float32),
        'norm_Y_I': np.ones(n_images, dtype=np.float32),
        'YY_full': None,
        'coords_nominal': np.random.rand(n_images, 1, 2, 1).astype(np.float32),
        'coords_true': np.random.rand(n_images, 1, 2, 1).astype(np.float32),
        'nn_indices': np.zeros((n_images, 7), dtype=np.int32),
        'global_offsets': np.random.rand(n_images, 1, 2, 1).astype(np.float32),
        'local_offsets': np.random.rand(n_images, 1, 2, 1).astype(np.float32),
        'probeGuess': np.random.rand(N, N).astype(np.complex64),
    })()

    # Create container - should NOT OOM because of lazy loading
    container = create_ptycho_data_container(raw_data, config)

    # Verify lazy storage
    assert hasattr(container, '_X_np'), "Container should use lazy storage"
    assert container._tensor_cache == {}, "Tensor cache should be empty initially"

    # Access .X triggers lazy conversion
    X_tensor = container.X
    assert isinstance(X_tensor, tf.Tensor)
    assert X_tensor.shape == (n_images, N, N, 1)

    # Verify caching worked
    assert 'X' in container._tensor_cache

    # Verify coords_nominal also works (needed for model.predict([X, coords]))
    coords = container.coords_nominal
    assert isinstance(coords, tf.Tensor)
    assert coords.shape == (n_images, 1, 2, 1)
```

### G5: Run Tests

```bash
ARTIFACTS=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T210000Z
mkdir -p "$ARTIFACTS"

# Run integration test
pytest tests/test_lazy_loading.py::TestCompareModelsChunking -v 2>&1 | tee "$ARTIFACTS/pytest_integration.log"

# Run full lazy loading suite
pytest tests/test_lazy_loading.py -v 2>&1 | tee "$ARTIFACTS/pytest_lazy_loading.log"

# Regression check
pytest tests/test_model_factory.py -v 2>&1 | tee "$ARTIFACTS/pytest_model_factory.log"
```

### G6: Update Summary

Prepend to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T210000Z/summary.md`:

```markdown
### Turn Summary
Added integration test verifying lazy container works in the inference path (create_ptycho_data_container returns lazy container).
Test confirms: NumPy storage, empty tensor cache initially, lazy conversion on property access, caching works.
All tests pass; lazy loading is confirmed integrated with compare_models workflow components.
Next: Consider marking G-scaled verification complete if integration test passes.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T210000Z/ (pytest_integration.log)
```

## How-To Map

```bash
ARTIFACTS=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T210000Z
mkdir -p "$ARTIFACTS"

# Run tests
pytest tests/test_lazy_loading.py::TestCompareModelsChunking -v
pytest tests/test_lazy_loading.py -v
pytest tests/test_model_factory.py -v
```

## Pitfalls To Avoid

1. **DO NOT** modify compare_models.py — this tests the existing integration
2. **DO** verify `_tensor_cache` is empty before first access
3. **DO NOT** skip the `create_ptycho_data_container` import — that's the real integration path
4. **DO** use `TrainingConfig` with valid defaults (gridsize=1 for simplicity)
5. **Environment Freeze:** Do not install packages

## If Blocked

1. If `create_ptycho_data_container` not found: check import path `ptycho.workflows.components`
2. If TrainingConfig errors: check required fields in `ptycho/params.py`
3. Record blocker in `$ARTIFACTS/blocked_<timestamp>.md`

## Findings Applied

- **PINN-CHUNKED-001:** This test verifies the resolution works in real workflow
- **DATA-001:** Uses same data structures as compare_models
- **CONFIG-001:** Uses TrainingConfig properly

## Pointers

- Lazy container: `ptycho/loader.py:97-321`
- Workflow components: `ptycho/workflows/components.py`
- Prior G-scaled test: `tests/test_lazy_loading.py::TestCompareModelsChunking::test_container_numpy_slicing_for_chunked_inference`

## Exit Criteria

1. New test `test_lazy_container_inference_integration` passes
2. All existing lazy loading tests (13/14) continue to pass (1 OOM skip)
3. Model factory regression (3/3) continues to pass
4. Evidence archived in `$ARTIFACTS`

## Next Up

- If G4-G6 pass: Consider STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 G-scaled verification complete
- Remaining blockers: BASELINE-CHUNKED-001/002 (separate from PINN chunking)
