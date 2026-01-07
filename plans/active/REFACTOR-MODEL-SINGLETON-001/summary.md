# REFACTOR-MODEL-SINGLETON-001 Summary

**Status:** done
**Created:** 2026-01-06
**Completed:** 2026-01-07
**Owner:** Ralph
**Spec Owner:** docs/specs/spec-ptycho-core.md

## Problem

The `ptycho/model.py` module executed extensive model construction at **module import time** (lines ~140-600), capturing `params.cfg` values into module-level variables. This prevented dynamic reconfiguration of model parameters (like `N` or `gridsize`) within a single process, causing shape mismatches when `create_model_with_gridsize()` was called with different parameters.

**Error Manifestation (fixed):**
```
InvalidArgumentError: Input to reshape is a tensor with 389376 values,
but the requested shape has 24336
  389376 = 78 * 78 * 64  (padded_size=78 for N=64, batch=64)
  24336  = 156 * 156     (padded_size=156 for N=128)
```

## Solution

Three-phase refactoring completed:
- **Phase A:** XLA workaround (env vars + eager execution) to stabilize multi-N workflows
- **Phase B:** Lazy loading via `__getattr__` - models created on-demand, not at import
- **Phase C:** Remove Phase A workarounds - lazy loading makes them unnecessary
- **Phase D:** Consumer updates (train_pinn, model_manager) - already complete

## Phase Completion Summary

### Phase A (2026-01-07) - XLA Workaround
- Added `USE_XLA_TRANSLATE=0` and `TF_XLA_FLAGS=--tf_xla_auto_jit=0` env vars
- Added `tf.config.run_functions_eagerly(True)`
- Test: `test_multi_n_model_creation` PASSED
- Commit: 3e877cde

### Phase B (2026-01-07) - Lazy Loading
- Implemented `_lazy_cache`, `_model_construction_done`, `_build_module_level_models()`
- Added `__getattr__` at module level for backward-compatible lazy singleton access
- Emits `DeprecationWarning` when legacy singletons accessed
- Test: `test_import_no_side_effects` PASSED
- Commit: 0206ff42

### Phase C (2026-01-07) - XLA Re-enablement
- Spike test verified lazy loading fixes multi-N XLA bug
- Removed all XLA workarounds from `dose_response_study.py` and `tests/test_model_factory.py`
- Updated `docs/findings.md` MODULE-SINGLETON-001 to "Resolved"
- Tests: All 3 PASSED (`test_multi_n_model_creation`, `test_import_no_side_effects`, `test_multi_n_with_xla_enabled`)
- Commit: 347ce7d6

### Phase D (Complete) - Consumer Updates
- D1: `train_pinn.train()` accepts `model_instance` argument, uses `create_compiled_model()` when None
- D2: `workflows/components.py` calls `train_pinn.train_eval()` which uses factories
- D3: `model_manager.py` calls `create_model_with_gridsize(gridsize, N)` explicitly
- D5: DeprecationWarning in `__getattr__` (from Phase B)
- D4: Pending verification - `dose_response_study.py` full run deferred to STUDY-SYNTH-DOSE-COMPARISON-001

## Exit Criteria

| Criterion | Status |
|-----------|--------|
| Importing `ptycho.model` does not instantiate Keras models or tf.Variables | ✅ Verified |
| `tests/test_model_factory.py::test_multi_n_model_creation` passes | ✅ PASSED |
| `tests/test_model_factory.py::test_import_no_side_effects` passes | ✅ PASSED |
| `dose_response_study.py` runs successfully with varying N/gridsize | ⚠️ Pending (STUDY-SYNTH-DOSE-COMPARISON-001) |

## Key Files

| File | Changes |
|------|---------|
| `ptycho/model.py` | Lazy loading via `__getattr__` (lines 867-890) |
| `ptycho/train_pinn.py` | `train()` accepts `model_instance`, uses factory (line 70) |
| `ptycho/model_manager.py` | Explicit `create_model_with_gridsize(gridsize, N)` (line 176) |
| `tests/test_model_factory.py` | Multi-N and import side-effect tests |
| `docs/findings.md` | MODULE-SINGLETON-001 marked Resolved |

## Artifacts

- `reports/2026-01-07T005113Z/pytest_model_factory.log` - Phase A test
- `reports/2026-01-07T040000Z/pytest_phase_b.log` - Phase B test
- `reports/2026-01-07T050000Z/pytest_phase_c_spike*.log` - Phase C spike
- `reports/2026-01-07T060000Z/pytest_phase_c_final.log` - Phase C final

## Next Steps

REFACTOR-MODEL-SINGLETON-001 is complete. STUDY-SYNTH-DOSE-COMPARISON-001 is now unblocked and should be the next focus to verify D4 (dose_response_study.py runs with varying N/gridsize).
