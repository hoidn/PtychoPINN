# Blocker: Baseline Model Returns All Zeros on Dense Test Split

**Date:** 2025-11-13T20:03:45Z  
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Severity:** Critical - Blocks Phase G completion

## Problem Statement

The Baseline model inference returns all-zero predictions on the dense **test** split despite receiving valid non-zero inputs, while the train split produces valid non-zero outputs. This is a TensorFlow/model runtime issue that cannot be resolved in `scripts/compare_models.py`.

## Evidence

### Train Split (✅ Works)
```
2025-11-13 19:59:59 - INFO - DIAGNOSTIC baseline_input stats: mean=0.112964, max=6.064422, nonzero_count=17447360/333447168
2025-11-13 20:00:16 - INFO - DIAGNOSTIC baseline_output stats: mean=0.003092, max=1.295571, nonzero_count=1387120/333447168
```
- Input: Valid (mean=0.113, 17.4M nonzero pixels)  
- Output: Valid (mean=0.003, 1.4M nonzero pixels)  
- **Result:** comparison_metrics.csv contains Baseline rows with canonical ID

### Test Split (❌ Fails)
```
2025-11-13 20:03:45 - INFO - DIAGNOSTIC baseline_output stats: mean=0.000000, max=0.000000, nonzero_count=0/341835776
2025-11-13 20:03:45 - ERROR - CRITICAL: Baseline model returned all-zero predictions!
RuntimeError: Baseline model inference failed: outputs are all zeros (mean=0.000000, nonzero=0/341835776). This indicates a TensorFlow/model runtime issue. Inputs were valid (mean=0.112671, nonzero=17793971). Investigation required before Phase G can proceed.
```
- Input: Valid (mean=0.113, 17.8M nonzero pixels)  
- Output: **All zeros** (mean=0.0, max=0.0, 0 nonzero pixels)  
- **Result:** RuntimeError halts execution before metrics can be saved

## Instrumentation Applied (Commit e830a5be)

1. **Zero-output assertion**: Added RuntimeError guard that halts execution if `baseline_output_mean == 0.0 or baseline_output_nonzero == 0`, providing diagnostic stats in the error message
2. **Canonical model IDs**: Fixed algorithm naming to emit `Baseline` and `PtyChi` instead of `"Pty-chi (pty-chi)"` per METRICS-NAMING-001

## Root Cause Analysis

The issue is **NOT** in compare_models.py. The instrumentation proves:
- Data preparation is correct (valid inputs reach the model)
- The Baseline model architecture is loaded successfully
- The model inference executes without TensorFlow errors
- Outputs are structurally correct (right shape/dtype) but numerically zero

Possible causes (all beyond compare_models.py scope):
1. **Model weights issue**: Test-specific numerical instability or underflow
2. **XLA compilation**: Different compilation paths for train vs test batch sizes  
3. **Data characteristics**: Test data triggers edge case in model internals
4. **TensorFlow runtime**: Graph optimization differences between runs

## Impact

- `analysis/metrics_summary.json` lacks Baseline rows for test split
- `report_phase_g_dense_metrics.py` fails with "Required models missing for delta computation: Baseline"
- `analysis/metrics_delta_highlights_preview.txt` never materializes
- `run_phase_g_dense.py --post-verify-only` aborts under PREVIEW-PHASE-001
- `analysis/verification_report.json` stuck at 0/10

## Next Steps

Per the Brief, the instrumentation task is complete. The zero-output issue requires investigation of:
1. Baseline model architecture and weights
2. TensorFlow graph compilation/optimization for test vs train
3. Test data numerical characteristics that might trigger underflow
4. Potential batch-size or memory-related differences

**Decision Point:** Proceed with PINN vs PtyChi only, OR debug Baseline model internals, OR retrain Baseline model.

## Artifacts

- Train split log (successful): `cli/compare_models_dense_train_instrumented.log`
- Train split CSV (canonical IDs): `analysis/dose_1000/dense/train/comparison_metrics.csv`
- Test split log (RuntimeError): `cli/compare_models_dense_test_instrumented.log`
- Instrumentation commit: e830a5be

