# Implementation Plan: Remove Module-Level Singletons (REFACTOR-MODEL-SINGLETON-001)

## Initiative
- **ID:** REFACTOR-MODEL-SINGLETON-001
- **Title:** Remove Global State Pollution in ptycho/model.py
- **Owner:** Ralph
- **Spec Owner:** specs/spec-ptycho-core.md
- **Status:** in_progress

## Problem Statement

The `ptycho/model.py` module executes extensive model construction at **module import time** (lines ~140-600), capturing `params.cfg` values into module-level variables. This prevents dynamic reconfiguration of model parameters (like `N` or `gridsize`) within a single process, causing shape mismatches when `create_model_with_gridsize()` is called with different parameters.

### Error Manifestation

```
InvalidArgumentError: Input to reshape is a tensor with 389376 values,
but the requested shape has 24336
  389376 = 78 * 78 * 64  (padded_size=78 for N=64, batch=64)
  24336  = 156 * 156     (padded_size=156 for N=128)
```

### Root Cause

1. **Module-Level Singletons:** Variables like `tprobe`, `initial_probe_guess`, `probe_illumination`, and model layers are instantiated at import time, baking in `N` and `gridsize` from global config.

2. **Stale Closure Captures:** Helper functions and layers capture global variables during definition or instantiation.

3. **XLA Trace Caching:** `@tf.function(jit_compile=True)` traces persist at Python level, expecting original shapes.

4. **Non-XLA Broadcasting Bug:** The fallback path in `translate_core` has fragile `tf.repeat` logic that produces empty tensors when batch dimensions don't align.

## Goals

1. **Eliminate Import-Time Side Effects:** Refactor `ptycho/model.py` to stop creating Keras models or tf.Variables at module import time.

2. **Enable Dynamic Reconfiguration:** Ensure model architecture (`N`, `gridsize`) respects runtime arguments passed to factories.

3. **Stabilize Non-XLA Execution:** Fix the broadcasting bug in `translate_core`.

## Phases Overview

- **Phase A — Immediate Stabilization:** Fix the non-XLA translation path in `tf_helper.py`.
- **Phase B — Core Refactor:** Move model construction into pure factory functions and sequester global state.
- **Phase C — Migration & Cleanup:** Update workflows and `model_manager` to use factories.

## Exit Criteria

1. `dose_response_study.py` runs successfully with varying `N` and `gridsize` in a single process.
2. Importing `ptycho.model` does not instantiate Keras models or create tf.Variables (verified by test).
3. `tests/test_tf_helper_broadcasting.py` passes.
4. Test registry synchronized: `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` updated.

## Compliance Matrix (Mandatory)

- [ ] **Spec Constraint:** `specs/spec-ptycho-core.md` (Model Sizes — N, gridsize, offset must be configurable)
- [ ] **Finding/Policy ID:** `MODULE-SINGLETON-001` (Global state capture at import time)
- [ ] **Finding/Policy ID:** `ANTIPATTERN-001` (Import-time side effects)
- [ ] **Finding/Policy ID:** `TF-NON-XLA-SHAPE-001` (Non-XLA translation path bug)
- [ ] **Finding/Policy ID:** `CONFIG-001` (Legacy dict updates must happen before usage)

## Sequencing

**Phase A must be complete (A0-A2 passing) before starting Phase B. Phase C depends on Phase B completion.**

## Rollback Trigger

If any existing test in `tests/test_model*.py` fails after Phase B changes, revert and reassess before proceeding.

## Context Priming

- **Primary docs:** `docs/DEVELOPER_GUIDE.md` (§2.1 Anti-Patterns), `ptycho/model.py`, `ptycho/tf_helper.py`
- **Required findings:** `MODULE-SINGLETON-001`, `TF-NON-XLA-SHAPE-001`
- **Data dependencies:** None (code refactor only)

---

## Phase A — Immediate Stabilization

**Objective:** Fix the immediate crash in non-XLA mode and create regression tests.
**Status:** ✅ COMPLETE (2026-01-07)

### Checklist

- [x] **A0: Environment Variable Fix:** Set `USE_XLA_TRANSLATE=0` before ptycho imports to prevent XLA trace caching.
  - Applied to `scripts/studies/dose_response_study.py`
  - Also added `TF_XLA_FLAGS=--tf_xla_auto_jit=0` to disable TensorFlow's XLA JIT

- [x] **A1: Eager Execution Workaround:** Added `tf.config.run_functions_eagerly(True)` to avoid Keras 3.x XLA graph compilation issues with dynamic batch dimensions in the non-XLA translation path.
  - Applied to both test and dose_response_study.py

- [x] **A2:** Create `tests/test_model_factory.py` with `test_multi_n_model_creation` asserting that models with N=128 and N=64 can be created and run forward passes in the same process without XLA shape conflicts.
  - Test PASSED (1 passed, 8.41s)
  - Artifacts: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T005113Z/pytest_model_factory.log`

### Implementation Notes

The original plan called for fixing `translate_core` broadcasting, but the root cause analysis revealed:
1. The XLA trace caching bug affects module-level code at import time
2. Keras 3.x uses XLA JIT for graph compilation by default
3. The non-XLA path's `tf.repeat` with dynamic `repeat_factor` is incompatible with XLA

The fix uses environment variables and eager execution to avoid XLA entirely for multi-N workflows:
- `USE_XLA_TRANSLATE=0` - prevents XLA translation path in tf_helper.py
- `TF_XLA_FLAGS=--tf_xla_auto_jit=0` - disables TensorFlow's auto JIT
- `tf.config.run_functions_eagerly(True)` - forces eager execution in Keras

### Dependency Analysis

- **Touched Modules:** `scripts/studies/dose_response_study.py`, `tests/test_model_factory.py`
- **Circular Import Risks:** None
- **State Migration:** None

---

## Phase B — Core Refactor

**Objective:** Move model construction into factories and sever global ties.
**Status:** ✅ COMPLETE (2026-01-07)

### Module-Level Variable Inventory (B0)

The following variables in `ptycho/model.py` were refactored:

| Variable | Original Line | Original Behavior | New Behavior |
|----------|--------------|------------------|--------------|
| `initial_probe_guess` | 162-165 | tf.Variable at import | Lazy via `_get_initial_probe_guess()` |
| `probe_illumination` | 233 | Singleton layer at import | Lazy via `_get_probe_illumination()` |
| `log_scale` | 240-243 | tf.Variable at import | Lazy via `_get_log_scale()` |
| Model singletons | 554-570 | Models at import | Lazy via `__getattr__` + `_build_module_level_models()` |

### Checklist

- [x] **B0: Audit Inventory:** Documented all module-level variables (see table above).

- [x] **B1:** Updated `ProbeIllumination` and `IntensityScaler` to use lazy getters.
  - `ProbeIllumination` now calls `_get_initial_probe_guess()` in backward-compat path
  - `IntensityScaler`/`IntensityScaler_inv` now call `_get_log_scale()`

- [x] **B2:** Verified `ExtractPatchesPositionLayer` accepts `N` and `gridsize` in `__init__`.

- [x] **B3:** `create_compiled_model(gridsize, N, ...)` remains the primary public API; verified functional.

- [x] **B4: Backward Compatibility (Lazy Init):** Implemented:
  - `_lazy_cache = {}` and `_model_construction_done = False` guards
  - `_build_module_level_models()` containing all model construction
  - `__getattr__` with DeprecationWarning for `autoencoder`, `diffraction_to_obj`, `autoencoder_no_nll`

- [x] **B5: Test:** Added `test_import_no_side_effects` to verify clean import. 2/2 tests PASSED.

### Implementation Notes

Key changes to `ptycho/model.py`:
1. Added lazy getter functions: `_get_initial_probe_guess()`, `_get_probe_illumination()`, `_get_log_scale()`
2. Wrapped model construction in `_build_module_level_models()` (lines 498-609)
3. Added `__getattr__` at module bottom for backward-compatible singleton access (lines 867-890)
4. Removed all module-level side effects (tf.Variable creation, model instantiation)

### Dependency Analysis

- **Touched Modules:** `ptycho/model.py`, `tests/test_model_factory.py`
- **Circular Import Risks:** None observed
- **Test Results:** `pytest tests/test_model_factory.py -vv` → 2 passed, 11.85s

### Notes & Risks

- **Risk:** Existing scripts using `from ptycho.model import autoencoder` will trigger DeprecationWarning
- **Mitigation:** Lazy loading preserves functionality while signaling migration need

---

## Phase C — XLA Re-enablement

**Objective:** Remove Phase A XLA workarounds now that lazy loading prevents import-time model creation.

**Rationale:** Phase A added XLA workarounds (`USE_XLA_TRANSLATE=0`, `TF_XLA_FLAGS`, eager execution) to fix multi-N crashes caused by import-time model construction. With Phase B complete, models are only created on-demand with correct N values, so XLA can be safely re-enabled for performance.

**Status:** C-SPIKE ✅ PASSED (2026-01-07) — Ready for C1-C4 implementation

### Spike Test Results (C-SPIKE)

The XLA re-enablement spike test verified the hypothesis that lazy loading fixes the multi-N XLA bug:

- **Test:** `tests/test_model_factory.py::TestXLAReenablement::test_multi_n_with_xla_enabled`
- **Result:** PASSED
- **Key Evidence:**
  - `Lazy loading verified: no models at import` — Phase B works
  - `XLA service ... initialized for platform CUDA` — XLA active (not disabled)
  - `Compiled cluster using XLA!` — XLA compilation occurred
  - Forward pass N=128 succeeded, output shapes correct
  - Forward pass N=64 succeeded, output shapes correct (previously this would crash!)
- **Artifacts:** `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/`
  - `pytest_phase_c_spike.log`
  - `pytest_phase_c_spike_verbose.log`

**Conclusion:** Lazy loading (Phase B) is sufficient to fix the XLA shape mismatch bug. Phase A workarounds can be removed.

### Checklist

- [x] **C-SPIKE:** Create and run XLA spike test to verify hypothesis.
  - Added `TestXLAReenablement::test_multi_n_with_xla_enabled` to `tests/test_model_factory.py`
  - Test runs in subprocess with XLA enabled (no env var workarounds)
  - PASSED — confirms lazy loading fixes the multi-N bug

- [ ] **C1:** Remove XLA workarounds from `scripts/studies/dose_response_study.py`:
  - Delete `os.environ['USE_XLA_TRANSLATE'] = '0'`
  - Delete `os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'`
  - Delete `tf.config.run_functions_eagerly(True)`

- [ ] **C2:** Update `tests/test_model_factory.py`:
  - Keep `test_multi_n_model_creation` but remove XLA workarounds (test should pass with XLA enabled)
  - Add `test_multi_n_with_xla` explicitly testing XLA mode works for multi-N

- [ ] **C3:** Verify XLA performance is restored:
  - Run `dose_response_study.py` without workarounds
  - Compare wall-clock time with/without `jit_compile=True`

- [ ] **C4:** Update `docs/findings.md` to note that MODULE-SINGLETON-001 is fully resolved (not just worked around).

### Exit Criteria

- `dose_response_study.py` runs without any XLA env vars or eager execution hacks
- `test_multi_n_model_creation` passes with XLA enabled
- No regression in model training/inference correctness

### Dependency Analysis

- **Depends on:** Phase B complete (lazy loading implemented)
- **Touched Files:** `scripts/studies/dose_response_study.py`, `tests/test_model_factory.py`, `docs/findings.md`

---

## Phase D — Migration & Cleanup

**Objective:** Update consumers to use the new factory API.

### Checklist

- [x] **D1:** Update `ptycho/train_pinn.py` to accept `model_instance` argument in `train()` and `train_eval()`.
  - Done: Line 70 accepts `model_instance=None`, lines 85-86 use `create_compiled_model()` when None

- [x] **D2:** Update `ptycho/workflows/components.py` (`train_cdi_model`, `run_cdi_example`) to call `create_compiled_model` using values from `TrainingConfig` instead of importing singletons.
  - Done: Line 736 calls `train_pinn.train_eval()` which internally uses factories

- [x] **D3:** Update `ptycho/model_manager.py` to explicitly pass `N`/`gridsize` to factories during loading, removing reliance on implicit global state during the load process.
  - Done: Line 176 calls `create_model_with_gridsize(gridsize, N)` explicitly

- [ ] **D4:** Verify `dose_response_study.py` runs successfully with varying `N` and `gridsize`.
  - Deferred to STUDY-SYNTH-DOSE-COMPARISON-001 (main goal of that initiative)

- [x] **D5:** Add `DeprecationWarning` message to lazy-loader (already in B4) with migration instructions.
  - Done: Lines 867-890 in ptycho/model.py emit DeprecationWarning

### Notes & Risks

- **Risk:** `model_manager` serialization/deserialization relies on global state restoration
- **Mitigation:** D3 explicitly addresses this by passing params to factory

---

## Work Completed in Previous Session (2026-01-06)

The following changes were applied but require verification/integration:

### ptycho/model.py
- `ProbeIllumination.__init__` now accepts `initial_probe` and `N` parameters
- `ProbeIllumination` generates `_probe_mask` in `__init__` (efficient, not in `call`)
- `create_model_with_gridsize` creates fresh `ProbeIllumination` with correct probe
- `create_model_with_gridsize` sets `use_xla_translate=False` and clears session

### ptycho/custom_layers.py
- `ExtractPatchesPositionLayer` accepts `N` and `gridsize` parameters
- `ReassemblePatchesLayer` accepts `padded_size`, `N`, and `gridsize` parameters

### ptycho/tf_helper.py
- `extract_patches_position` accepts `N` and `gridsize` parameters
- `reassemble_patches` accepts `N` and `gridsize` parameters
- `mk_norm` accepts `N` and `gridsize` parameters

### Known Issue
- Non-XLA path still fails with broadcasting bug (Phase A target)
- Module-level variable `probe` was shadowed by output tensor (fixed: renamed to `probe_tensor`)

---

## Artifacts Index

- Reports root: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/`
- Session log: `2026-01-06T.../`

## References

- Issue discovered in: `scripts/studies/dose_response_study.py`
- Related spec: `specs/spec-ptycho-core.md`
- Related guide: `docs/GRIDSIZE_N_GROUPS_GUIDE.md`
