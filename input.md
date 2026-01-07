Mode: Implementation
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — G-scaled: Verify lazy loading enables chunked PINN inference
Branch: feature/torchapi-newprompt-2
Mapped tests: pytest tests/test_lazy_loading.py -v
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T200000Z/

## Summary

Verify that FEAT-LAZY-LOADING-001 (just completed) enables chunked PINN inference for compare_models.py. Run a minimal scaled test to prove OOM is resolved.

## Context

**PINN-CHUNKED-001 RESOLVED:** FEAT-LAZY-LOADING-001 Phase C complete (commit bd8d9480). `PtychoDataContainer` now:
- Stores data as NumPy arrays internally (`_X_np`, `_Y_I_np`, etc.)
- Provides lazy tensor conversion via properties
- Offers `as_tf_dataset(batch_size)` for streaming

**Prior Blocker:** `scripts/compare_models.py` PINN inference failed with OOM because `create_ptycho_data_container()` eagerly converted ALL groups to GPU tensors in `PtychoDataContainer.__init__`.

**Expected Fix:** Container now stores NumPy; direct access to `._X_np` enables chunk-wise inference without full GPU allocation.

## Do Now

### G1: Verify Lazy Loading Enables Chunked Access

**Task:** Add a minimal test to verify `compare_models.py` can slice container data without OOM.

**File:** `tests/test_lazy_loading.py`

Add to `TestStreamingTraining` or new `TestCompareModelsChunking` class:

```python
def test_container_numpy_slicing_for_chunked_inference(self):
    """Verify container supports NumPy slicing for chunked PINN inference."""
    from ptycho.loader import PtychoDataContainer
    import numpy as np

    # Create container with moderate size
    N = 64
    C = 4
    n_images = 500  # Moderate size

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

    # Verify NumPy arrays are accessible for chunked slicing
    assert hasattr(container, '_X_np'), "Container should have _X_np for chunked access"
    assert container._X_np.shape == (n_images, N, N, C)

    # Slice chunks without triggering full tensor conversion
    chunk_size = 100
    for i in range(0, n_images, chunk_size):
        chunk_X = container._X_np[i:i+chunk_size]
        chunk_coords = container._coords_nominal_np[i:i+chunk_size]
        assert chunk_X.shape[0] <= chunk_size
        assert chunk_coords.shape[0] <= chunk_size

    # Full .X access should still work (backward compat)
    X_tensor = container.X
    assert X_tensor.shape == (n_images, N, N, C)
```

### G2: Run Verification

```bash
ARTIFACTS=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T200000Z
mkdir -p "$ARTIFACTS"

# Run lazy loading tests including new chunking test
pytest tests/test_lazy_loading.py -v 2>&1 | tee "$ARTIFACTS/pytest_lazy_loading.log"

# Verify regression
pytest tests/test_model_factory.py -v 2>&1 | tee "$ARTIFACTS/pytest_model_factory.log"
```

### G3: Document Evidence

If tests pass, update `docs/findings.md` PINN-CHUNKED-001 to note that chunked inference is now possible via `_X_np` slicing.

## How-To Map

```bash
ARTIFACTS=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T200000Z
mkdir -p "$ARTIFACTS"

# Run tests
pytest tests/test_lazy_loading.py -v
pytest tests/test_model_factory.py -v
```

## Pitfalls To Avoid

1. **DO NOT** modify `compare_models.py` yet — just prove the container API supports chunking
2. **DO** verify `_X_np` attribute exists on container (not all paths set it)
3. **DO NOT** trigger full `.X` access in the chunking loop — that defeats the purpose
4. **DO** check that `_coords_nominal_np` exists (needed for offset alignment in compare_models)
5. **Environment Freeze:** Do not install packages

## If Blocked

1. If `_X_np` doesn't exist: check if container was created with tensor inputs (backward compat path)
2. If `_coords_nominal_np` missing: need to add it to lazy container (minor update to loader.py)
3. Record blocker in `$ARTIFACTS/blocked_<timestamp>.md`

## Findings Applied

- **PINN-CHUNKED-001:** This test verifies the resolution path
- **DATA-001:** Container preserves NPZ data contract shapes
- **CONFIG-001:** Not applicable (no params.cfg interaction in this test)

## Pointers

- Lazy container: `ptycho/loader.py:97-321`
- Phase C evidence: `plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-08T030000Z/pytest_phase_c.log`
- PINN-CHUNKED-001 finding: `docs/findings.md` (search "PINN-CHUNKED-001")

## Exit Criteria

1. New test `test_container_numpy_slicing_for_chunked_inference` passes
2. All existing lazy loading tests (12/13) continue to pass
3. Model factory regression (3/3) continues to pass
4. Evidence archived in `$ARTIFACTS`

## Next Up

- G-scaled full pipeline: Run compare_models with chunked PINN inference on small dataset
- Update compare_models.py to use `_X_np` slicing instead of full container
