Mode: Implementation
Focus: REFACTOR-MODEL-SINGLETON-001 — Phase A (XLA Trace Cache Fix)
Selector: tests/test_model_factory.py::test_multi_n_model_creation

## Overview

The `dose_response_study.py` crashes when creating models with different `N` values in a single process. The error occurs in `projective_warp_xla_jit` because the `@tf.function(jit_compile=True)` decorator traces the function with the first-seen shapes (N=128 → padded_size=156), and those traces persist even after `tf.keras.backend.clear_session()`. When a subsequent model uses N=64 (padded_size=78), the XLA graph expects 24336 values but receives 389376.

**Error signature:**
```
InvalidArgumentError: Input to reshape is a tensor with 389376 values, but the requested shape has 24336
  389376 = 78 × 78 × 64  (padded_size=78 for N=64, batch=64)
  24336  = 156 × 156     (padded_size=156 for N=128 — stale XLA trace)
```

**Evidence:** `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T163900Z/red/dose_response_reproduce.log`

## Contracts

- **MODULE-SINGLETON-001** (docs/findings.md): Module-level singletons capture shapes at import time; factory functions must create fresh models.
- **docs/specs/spec-ptycho-runtime.md:15**: Non-XLA translation runs MUST respect `USE_XLA_TRANSLATE` toggle.
- **docs/specs/spec-ptycho-core.md**: Model Sizes (N, gridsize, offset) must be configurable at runtime.

## Root Cause Analysis

1. `projective_warp_xla_jit` at ptycho/projective_warp_xla.py:202-211 is decorated with `@tf.function(jit_compile=True)`
2. This causes TF to trace and XLA-compile the function with the first-seen input shapes
3. `tf.keras.backend.clear_session()` clears Keras models but does NOT clear TF function traces
4. `create_model_with_gridsize()` at ptycho/model.py sets `use_xla_translate=False` in params.cfg
5. BUT: `Translation` layer instances are created during model building and call `should_use_xla()` at layer instantiation time
6. If params.cfg hasn't been updated yet (or if `should_use_xla()` is cached), the XLA path is still taken

## Phase A Fix Strategy

**Approach:** Force non-XLA mode for multi-config scenarios by ensuring the XLA toggle is properly propagated during model construction.

**Key insight:** The `Translation` layer at tf_helper.py:817-850 accepts `use_xla` in `__init__`. The model factory must explicitly pass `use_xla=False` when creating layers, not rely on global config.

## Tasks

### A0: Test-First Gate (RED)
**File:** `tests/test_model_factory.py`
**Function:** `test_multi_n_model_creation`

Create a test that reproduces the shape mismatch bug:
1. Create a model with N=128, gridsize=2
2. Run a forward pass to trigger XLA tracing
3. Clear session
4. Create a model with N=64, gridsize=2
5. Run a forward pass — this should NOT crash

The test should initially FAIL (RED) demonstrating the bug.

### A1: Fix Translation Layer XLA Toggle
**File:** `ptycho/model.py::create_model_with_gridsize`
**Function:** Ensure Translation layers are created with explicit `use_xla=False`

The fix involves:
1. Pass `use_xla=False` explicitly to all Translation layer instantiations within `create_model_with_gridsize()`
2. OR: Set `os.environ['USE_XLA_TRANSLATE'] = '0'` at the start of the factory function
3. Verify by checking `_reassemble_patches_position_real` at tf_helper.py:879 which creates `Translation(jitter_stddev=0.0, use_xla=should_use_xla())`

### A2: Verify Test Passes (GREEN)
Run the test again after the fix — it should now PASS.

```bash
pytest tests/test_model_factory.py::test_multi_n_model_creation -vv
```

## Pitfalls To Avoid

1. **DO NOT** rely on `tf.keras.backend.clear_session()` alone — it doesn't clear XLA traces
2. **DO NOT** modify `projective_warp_xla_jit` decorator — changing shared infra is out of scope
3. **DO NOT** add try/except around XLA calls — fix the toggle propagation instead
4. **DO** use `USE_XLA_TRANSLATE=0` environment variable OR explicit `use_xla=False` parameter
5. **DO** verify the fix works with the actual `dose_response_study.py` after tests pass

## Selector

```bash
pytest tests/test_model_factory.py::test_multi_n_model_creation -vv
```

Expected: Initially FAIL (RED), then PASS (GREEN) after fix.

## Artifacts

- Reports: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T173000Z/`
- Test log: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T173000Z/green/pytest_model_factory.log`

## Findings Applied

- **MODULE-SINGLETON-001**: Factory functions must create fresh models; this task ensures XLA traces don't persist across model creations.
- **CONFIG-001**: `update_legacy_dict()` must run before legacy modules; the fix ensures XLA toggle is set before Translation layers are instantiated.
- **ANTIPATTERN-001**: Import-time side effects; the factory approach avoids import-time model creation.

## Pointers

- Implementation plan: `plans/active/REFACTOR-MODEL-SINGLETON-001/implementation.md` (Phase A checklist)
- Translation layer: `ptycho/tf_helper.py:817-850`
- XLA toggle: `ptycho/tf_helper.py:154-175` (`should_use_xla()`)
- Model factory: `ptycho/model.py::create_model_with_gridsize`
- Blocker log: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-06T163900Z/red/dose_response_reproduce.log`

## Next Up (Optional)

If A0-A2 complete successfully:
- A3: Run `dose_response_study.py` end-to-end to verify the full workflow
- Move to Phase B (Module Variable Inventory) per implementation.md
