# Blocker: Baseline Test Split Outputs All Zeros

**Date**: 2025-11-14T04:31:36Z
**Focus**: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Severity**: Critical (blocks Phase G completion)

## Issue

Baseline model inference succeeds on the **train** split but returns **all-zero outputs** on the **test** split, despite receiving valid inputs. This prevents metrics computation and blocks the entire Phase G dense study pipeline.

## Evidence

### Train Split (SUCCESS)
```
File: analysis/dose_1000/dense/train/logs/logs/debug.log:424-535
DIAGNOSTIC baseline_input stats: mean=0.112964, max=6.064422, nonzero_count=17447360/333447168
DIAGNOSTIC baseline_output stats: mean=0.003092, max=1.295571, nonzero_count=1387120/333447168
```

### Test Split (FAILURE)
```
File: analysis/dose_1000/dense/test/logs/logs/debug.log (just captured)
DIAGNOSTIC baseline_input stats: mean=0.112671, max=5.362341, nonzero=17793971
DIAGNOSTIC baseline_output stats: mean=0.000000, max=0.000000, nonzero_count=0/341835776
```

### RuntimeError Triggered
```python
RuntimeError: Baseline model inference failed: outputs are all zeros (mean=0.000000, nonzero=0/341835776).
This indicates a TensorFlow/model runtime issue. Inputs were valid (mean=0.112671, nonzero=17793971).
Investigation required before Phase G can proceed.
```

## Analysis

1. **Inputs are valid**: Both splits have similar input statistics (mean~0.112, max~5-6, millions of nonzero values)
2. **Model loads successfully**: No errors during `load_inference_bundle()` for either split
3. **Train split works**: Baseline produces non-zero outputs (mean=0.003092, 1.4M nonzero pixels) on train data
4. **Test split fails**: Baseline produces all zeros despite healthy inputs
5. **Not a compare_models.py issue**: Diagnostic instrumentation confirms the model.predict() call returns zeros

## Root Cause Hypothesis

This is a **TensorFlow/model runtime issue**, not a data preprocessing or stitching problem. Possible causes:

1. **Numerical underflow in model layers** specific to test data characteristics (different batch sizes, data distribution)
2. **XLA compilation differences** between train/test execution paths
3. **Model state corruption** or uninitialized weights affecting only certain data patterns
4. **Test data batch size** (652 vs 636 batches) triggering edge case in model architecture

## Immediate Action Required

This blocker is **beyond the scope of scripts/compare_models.py fixes**. Investigation requires:

1. **Baseline model architecture review**: Check for numerical instability in model layers
2. **TensorFlow runtime debugging**: Enable TF debug mode to trace tensor values through inference
3. **Data characteristic analysis**: Compare train vs test data distributions to find trigger
4. **Model weight inspection**: Verify baseline model weights are properly loaded and not corrupt

## Decision Point

**Options**:
1. **Debug baseline model** (time-consuming, requires TF/model expertise)
2. **Proceed with PINN vs PtyChi only** (skip Baseline rows in Phase G metrics)
3. **Retrain baseline model** from scratch to rule out weight corruption

**Recommendation**: Escalate to supervisor for decision on whether to proceed with partial metrics (PINN vs PtyChi) or invest time in baseline debugging.

## Commands That Failed

```bash
# Test split comparison (triggers RuntimeError)
PYTHONPATH="$PWD" python scripts/compare_models.py \
  --pinn_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/baseline/gs1 \
  --test_data plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_c/dose_1000/patched_test.npz \
  --output_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test \
  --ms-ssim-sigma 1.0 \
  --tike_recon_path plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_f/dose_1000/dense/test/ptychi_reconstruction.npz \
  --register-ptychi-only
```

## Status

- **Translation regression tests**: GREEN (2/2 passed in 6.17s)
- **Train split compare_models**: SUCCESS (non-zero Baseline outputs, metrics computed)
- **Test split compare_models**: BLOCKED (all-zero Baseline outputs, RuntimeError)
- **Phase G pipeline**: BLOCKED (cannot proceed without valid Baseline metrics for both splits)
- **Supervisor decision**: PENDING

## Artifacts

- Translation tests: `green/pytest_compare_models_translation_fix_v11.log`
- Train split metrics: `analysis/dose_1000/dense/train/comparison_metrics.csv` (Baseline rows present)
- Test split failure log: Captured in command output above
- Diagnostic code: `scripts/compare_models.py:1053-1071, 1111-1118`
