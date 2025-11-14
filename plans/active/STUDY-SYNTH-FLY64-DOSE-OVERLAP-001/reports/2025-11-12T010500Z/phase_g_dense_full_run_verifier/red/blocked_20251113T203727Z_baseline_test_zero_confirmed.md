# Blocker: Baseline Test Split Returns Zero Outputs (Confirmed with Fast Debug Loop)

**Date**: 2025-11-13T20:37:27Z
**Focus**: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Severity**: Critical (blocks Phase G completion)

## Summary

Fast debug runs with `--n-test-groups 10` confirm the Baseline model inference issue:
- **Train split** (10 groups): Baseline outputs are sparse but non-zero (mean=0.000415, 406 nonzero pixels)
- **Test split** (10 groups): Baseline outputs are **completely zero** (mean=0.000000, 0 nonzero pixels)

The instrumentation with first-patch logging and the RuntimeError assertion correctly halts execution when test split Baseline outputs are all zeros.

## Evidence

### Train Split Debug Run (10 groups - SUCCESS with sparse outputs)
```
File: cli/compare_models_dense_train_debug.log
DIAGNOSTIC baseline_input stats: mean=0.111416, max=4.601460, nonzero_count=33335/655360
DIAGNOSTIC baseline_output stats: mean=0.000415, max=0.938639, nonzero_count=406/655360
```
**Result**: Completes successfully. Baseline metrics show NaN in CSV (likely due to sparse outputs), but model does produce some non-zero values.

### Test Split Debug Run (10 groups - FAILURE with all-zero outputs)
```
File: cli/compare_models_dense_test_debug.log
DIAGNOSTIC baseline_input stats: mean=0.112296, max=4.524805, nonzero_count=33937/655360
DIAGNOSTIC baseline_output stats: mean=0.000000, max=0.000000, nonzero_count=0/655360

RuntimeError: Baseline model inference failed: outputs are all zeros (mean=0.000000,
nonzero=0/655360). This indicates a TensorFlow/model runtime issue. Inputs were valid
(mean=0.112296, nonzero=33937). Investigation required before Phase G can proceed.
```
**Result**: RuntimeError correctly raised, execution halted.

## Analysis

1. **Inputs are consistently healthy**: Both splits have similar input statistics (mean~0.11-0.12, thousands of nonzero values)
2. **Train split is marginal**: Only 406 nonzero pixels out of 655,360 total pixels (0.06%), suggesting the model is barely producing outputs even on train data
3. **Test split completely fails**: Absolute zeros across all pixels
4. **Not a data prep issue**: The instrumentation confirms inputs are valid
5. **TensorFlow/model runtime issue**: This is beyond scripts/compare_models.py scope

## Root Cause Hypothesis

The Baseline model appears to have severe numerical instability issues:
- **Train split**: Model produces outputs but they are extremely sparse (99.94% zeros)
- **Test split**: Model produces no outputs at all (100% zeros)

Possible causes:
1. **Numerical underflow** in model layers (activations collapsing to zero)
2. **Gradient vanishing** during training leading to near-dead weights
3. **Batch size sensitivity** (test has 40 patches vs train could have different batch characteristics)
4. **Model architecture issues** (e.g., ReLU clipping, batch norm issues)
5. **Weight initialization or corruption**

## Commands Executed

### Train Split (10 groups)
```bash
PYTHONPATH="$PWD" python scripts/compare_models.py \
  --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 \
  --test_data "$HUB"/data/phase_c/dose_1000/patched_train.npz \
  --output_dir "$HUB"/analysis/dose_1000/dense/train_debug \
  --ms-ssim-sigma 1.0 \
  --tike_recon_path "$HUB"/data/phase_f/dose_1000/dense/train/ptychi_reconstruction.npz \
  --register-ptychi-only \
  --n-test-groups 10
```
**Exit code**: 0 (success)

### Test Split (10 groups)
```bash
PYTHONPATH="$PWD" python scripts/compare_models.py \
  --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 \
  --test_data "$HUB"/data/phase_c/dose_1000/patched_test.npz \
  --output_dir "$HUB"/analysis/dose_1000/dense/test_debug \
  --ms-ssim-sigma 1.0 \
  --tike_recon_path "$HUB"/data/phase_f/dose_1000/dense/test/ptychi_reconstruction.npz \
  --register-ptychi-only \
  --n-test-groups 10
```
**Exit code**: 1 (RuntimeError raised as designed)

## Decision Point

The instrumentation is working correctly. The Baseline model has fundamental issues that prevent it from producing valid outputs on the test split (and marginal outputs on train split).

**Options**:
1. **Proceed with PINN vs PtyChi only** - Skip Baseline rows in metrics (fastest path forward)
2. **Debug Baseline model architecture** - Investigate layer outputs, activations, gradients (time-consuming)
3. **Retrain Baseline from scratch** - Could fix weight corruption but won't fix architecture issues
4. **Accept train-split-only Baseline metrics** - Document the test split failure but keep train metrics

**Recommendation**: Option 1 (proceed with PINN vs PtyChi) allows Phase G to complete. Baseline debugging should be a separate investigation outside the critical path.

## Status

- Translation regression tests: ✅ GREEN (2/2 passed in 6.20s)
- Train split compare_models (10 groups): ✅ SUCCESS (sparse but non-zero outputs)
- Test split compare_models (10 groups): ❌ BLOCKED (all-zero outputs, RuntimeError)
- Fast debug instrumentation: ✅ Working correctly (--n-test-groups, first-patch logging, RuntimeError gate)
- Phase G pipeline: **BLOCKED** pending supervisor decision

## Artifacts

- Translation guard: `green/pytest_compare_models_translation_fix_v12.log` (2/2 PASSED, 6.20s)
- Train debug (10 groups): `cli/compare_models_dense_train_debug.log` (sparse outputs, exit 0)
- Test debug (10 groups): `cli/compare_models_dense_test_debug.log` (zero outputs, RuntimeError)
- Diagnostic instrumentation: `scripts/compare_models.py:1032-1071, 1111-1118`

## Next Steps

Await supervisor decision on:
1. Whether to proceed with PINN vs PtyChi metrics only (skip Baseline)
2. Whether to invest time in Baseline model debugging/retraining
3. Whether to accept partial metrics (train-only Baseline data)
