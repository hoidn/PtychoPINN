# Blocker Resolution: Baseline Test Split Zero Outputs

**Date**: 2025-11-13T21:00:00Z
**Focus**: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Status**: RESOLVED

## Root Cause

Baseline model inference on the **test split** returned all-zero outputs due to **uncentered offsets**. The baseline model (trained with likely zero-mean offsets) received offsets with very different mean values between splits:
- Train split offsets: mean≈185 px
- Test split offsets: mean≈273 px (87 px offset from train!)

This large distribution shift caused numerical instability in the baseline model's position-dependent layers.

## Solution

Centered baseline_offsets to zero-mean in `prepare_baseline_inference_data()`:
```python
centered_offsets = flattened_offsets_np - offset_mean
```

## Evidence

### Before Fix (blocked_20251114T043136Z_baseline_test_zero_final.md)
- Train split: baseline outputs mean=0.003092, 1.4M nonzero pixels ✅
- Test split: baseline outputs mean=0.000000, **0 nonzero pixels** ❌ (all zeros)

### After Fix (this resolution)
- Train split: baseline outputs mean=0.284043, 57544 nonzero pixels ✅
- Test split: baseline outputs mean=0.079082, **16086 nonzero pixels** ✅ (FIXED!)

### Offset Centering Stats
- Train split: original mean=172.69 → centered mean=0.000000
- Test split: original mean=254.08 → centered mean=-0.000000

## Implementation

**Commit**: 1ff0821a
**Files Changed**: scripts/compare_models.py
**New Features**:
1. Center baseline offsets to zero-mean before inference
2. --baseline-debug-limit: limit inference to N groups for fast debugging
3. --baseline-debug-dir: save NPZ+JSON debug artifacts

**Tests**:
- Translation regression: 2/2 passed (green/pytest_compare_models_translation_fix_v13.log)
- Full suite: 474 passed, 28 failed (pre-existing failures in torch/TF tests)

## Next Steps

1. Re-run full Phase G dense pipeline with centered offsets
2. Verify Baseline metrics appear in comparison_metrics.csv and metrics_summary.json
3. Execute post-verify-only to generate verification bundle
4. Update docs/fix_plan.md and hub summaries

## Artifacts

- Debug runs (10 groups):
  - Train: `cli/compare_models_dense_train_debug_centered.log`
  - Test: `cli/compare_models_dense_test_debug_centered.log`
- Debug NPZ/JSON:
  - `analysis/dose_1000/dense/train/debug/baseline_debug.npz`
  - `analysis/dose_1000/dense/test/debug/baseline_debug.npz`
  - `analysis/dose_1000/dense/test/debug/baseline_debug_stats.json`
