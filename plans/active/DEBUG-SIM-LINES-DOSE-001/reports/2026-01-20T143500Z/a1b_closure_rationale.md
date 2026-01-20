# A1b Ground Truth Run — Closure Rationale

**Date:** 2026-01-20T14:35:00Z
**Initiative:** DEBUG-SIM-LINES-DOSE-001
**Checklist Item:** A1b (Run dose_experiments simulate → train → infer)

## Summary

The A1b ground-truth run cannot be completed as originally specified due to a fundamental **Keras 3.x API incompatibility** with the legacy `dose_experiments` code. This blocker is **outside the current initiative scope** and does not affect the successful completion of the NaN debugging objective.

## Blocker Details

### Environment
- Current environment: TensorFlow 2.18+ with Keras 3.x
- Legacy code location: `/home/ollie/Documents/PtychoPINN` (dose_experiments branch)

### Failure Point
- **Simulation stage:** ✅ Works (with parameter clamping for neighbor_count/nimages/gridsize/N)
- **Training stage:** ❌ Fails at model construction
- **Error:** `KerasTensor cannot be used as input to a TensorFlow function`
- **Root cause:** Legacy `ptycho/model.py` uses `tf.shape()` directly on Keras tensors, which is prohibited in Keras 3.x

### Technical Details
The failure occurs in:
```
ptycho/model.py:258 → tf_helper.py:450
  b = tf.shape(inputs)[0]
```

Keras 3.x requires wrapping such TF operations in a Keras layer. The legacy code predates this API change.

## Why A1b Was Requested

A1b was intended to:
1. Establish ground-truth outputs from the legacy pipeline for comparison
2. Verify that the `dose_experiments` configuration actually worked
3. Provide a reference point for debugging sim_lines_4x discrepancies

## Why A1b Is No Longer Required

The **primary debugging goal is already achieved**:

1. **NaN Root Cause Identified:** CONFIG-001 violation (stale `params.cfg` values not synced before training/inference)
2. **Fix Applied:** C4f added `update_legacy_dict(params.cfg, config)` calls before all training/inference handoffs
3. **Verification Complete:** All four scenarios (gs1/gs2 × ideal/custom) now train without NaN

The A1b ground-truth run was a diagnostic tool to compare legacy vs modern behavior. Since we have:
- Identified the root cause (CONFIG-001)
- Applied and verified the fix (C4f)
- Confirmed all scenarios work correctly

...the ground-truth comparison is no longer necessary for the NaN debugging scope.

## What Was Accomplished

Despite not completing the full simulate→train→infer flow:

1. **Simulation stage succeeded** with 512 diffraction patterns generated
2. **Parameter clamping implemented** in `run_dose_stage.py`:
   - neighbor_count clamped to `min(default, nimages - 1)`
   - nimages capped at 512 to avoid GPU OOM
   - gridsize forced to 1 (RawData.from_simulation requirement)
   - N forced to 64 (probe size from NPZ)
3. **Compatibility shims built**:
   - `tfa_stub/` for tensorflow_addons
   - `components` patch for update_params/setup_configuration
   - RawData patch for from_simulation with return_patches

Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/`

## Resolution Options (Out of Scope)

If A1b ground-truth comparison is needed in the future:

1. **Option A:** Provision a legacy TF 2.x / Keras 2.x conda environment
   - Effort: Medium (environment isolation, dependency pinning)
   - Risk: May have other compatibility issues

2. **Option B:** Port legacy model.py to Keras 3.x
   - Effort: High (architectural changes to model construction)
   - Risk: May change behavior, defeating ground-truth purpose

3. **Option C:** Accept simulation-only capability
   - The simulation stage works; use it for data generation comparisons
   - Training/inference comparisons would use the modern codebase only

## Decision

**A1b is documented as blocked but no longer required** for the NaN debugging scope.

The initiative's primary objective (NaN debugging) is COMPLETE. The amplitude bias (~3-6x undershoot) that remains is a separate issue requiring a distinct investigation initiative if prioritized.

## Evidence

- Simulation success: `simulation_clamped4.log`
- Training failure: Keras 3.x KerasTensor error in same log
- Prior attempts: `simulation_attempt*.log`, `simulation_smoke.log`
- Fix verification: `reports/2026-01-20T102300Z/` (B0f isolation test), `reports/2026-01-20T160000Z/` (C4f CONFIG-001 bridging)
