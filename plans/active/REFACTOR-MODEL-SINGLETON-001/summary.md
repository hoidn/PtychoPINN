# REFACTOR-MODEL-SINGLETON-001 Summary

**Status:** in_progress
**Created:** 2026-01-06
**Owner:** Ralph
**Spec Owner:** docs/specs/spec-ptycho-core.md

## Problem

The `ptycho/model.py` module executes extensive model construction at **module import time** (lines ~140-600), capturing `params.cfg` values into module-level variables. This prevents dynamic reconfiguration of model parameters (like `N` or `gridsize`) within a single process, causing shape mismatches when `create_model_with_gridsize()` is called with different parameters.

**Error Manifestation:**
```
InvalidArgumentError: Input to reshape is a tensor with 389376 values,
but the requested shape has 24336
  389376 = 78 * 78 * 64  (padded_size=78 for N=64, batch=64)
  24336  = 156 * 156     (padded_size=156 for N=128)
```

**Root Causes:**
1. Module-level singletons (`tprobe`, `initial_probe_guess`, `probe_illumination`, etc.)
2. Stale closure captures in helper functions
3. XLA trace caching
4. Non-XLA broadcasting bug in `translate_core`

## Solution

Three-phase refactoring:
- **Phase A:** Fix non-XLA translation path (immediate stabilization)
- **Phase B:** Move model construction into factory functions
- **Phase C:** Update consumers to use factory API

See `implementation.md` for full details.

## Current Phase

**Phase A — Immediate Stabilization** (in progress)

Blockers:
- Non-XLA `translate_core` broadcasting bug needs fix before testing Phase B changes

## Turn Log

### 2026-01-06 — Plan Revision and Restart

Previous "completed" status was premature. The simple factory function approach (`create_compiled_model`) was added but does not solve the underlying global state pollution. The shape mismatch error persists when running `dose_response_study.py` with varying `N` and `gridsize`.

**Work completed in previous session:**
- `ProbeIllumination.__init__` now accepts `initial_probe` and `N` parameters
- `ProbeIllumination` generates `_probe_mask` in `__init__` (efficient, not in `call`)
- `create_model_with_gridsize` creates fresh `ProbeIllumination` with correct probe
- `create_model_with_gridsize` sets `use_xla_translate=False` and clears session
- `ExtractPatchesPositionLayer` accepts `N` and `gridsize` parameters
- `ReassemblePatchesLayer` accepts `padded_size`, `N`, and `gridsize` parameters
- `extract_patches_position`, `reassemble_patches`, `mk_norm` accept N/gridsize parameters

**Known Issue:**
- Non-XLA path still fails with broadcasting bug (Phase A target)
- Module-level variable `probe` was shadowed by output tensor (fixed: renamed to `probe_tensor`)

Revised implementation plan written to `implementation.md`.

### 2026-01-06 — Initial Analysis (earlier)

- Identified root cause: module-level model creation at import time
- Created initial simple factory function approach
- Marked complete prematurely

## Key Files

| File | Status |
|------|--------|
| `ptycho/model.py` | Partially refactored |
| `ptycho/custom_layers.py` | Updated (N/gridsize params) |
| `ptycho/tf_helper.py` | Partially updated, needs Phase A fix |

## Exit Criteria

1. [ ] `dose_response_study.py` runs successfully with varying `N` and `gridsize`
2. [ ] Importing `ptycho.model` does not instantiate Keras models or tf.Variables
3. [ ] `tests/test_tf_helper_broadcasting.py` passes
4. [ ] Test registry synchronized

## Blocking Issues

**TF-NON-XLA-SHAPE-001:** Non-XLA translation path has broadcasting bug in `translate_core` (lines ~779-800). Must be fixed before Phase B can be validated.
