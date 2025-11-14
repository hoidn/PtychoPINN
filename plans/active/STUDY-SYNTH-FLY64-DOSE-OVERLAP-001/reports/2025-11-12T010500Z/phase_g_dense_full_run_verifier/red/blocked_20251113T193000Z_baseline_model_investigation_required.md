# Blocker: Baseline Model Returns Zero Predictions on Test Split

**Timestamp:** 2025-11-13T19:30:00Z
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Severity:** Critical - Blocks Phase G completion
**Root Cause:** TensorFlow/Baseline Model Runtime Issue

## Executive Summary

Translation regression guards remain GREEN (2 passed, 6.26s). However, Phase G cannot proceed because the Baseline model produces **all-zero predictions on the test split** while train split works correctly. Diagnostic instrumentation confirms this is a **TensorFlow/model runtime issue**, not a compare_models.py data handling bug.

## Evidence

### Translation Guards: GREEN ✓
```
tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_batched_predictions PASSED
tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split PASSED
2 passed in 6.26s
```
Source: `$HUB/green/pytest_compare_models_translation_fix_v9.log`

### Baseline Inference: Train Split Works, Test Split Fails

**Train split** (SUCCESS):
```
DIAGNOSTIC baseline_input stats: mean=0.112964, max=6.064422, nonzero_count=17447360/333447168
DIAGNOSTIC baseline_output stats: mean=0.003092, max=1.295571, nonzero_count=1387120/333447168
```

**Test split** (FAILURE):
```
DIAGNOSTIC baseline_input stats: mean=0.112671, max=5.362341, nonzero_count=17793971/341835776
DIAGNOSTIC baseline_output stats: mean=0.000000, max=0.000000, nonzero_count=0/341835776
ERROR - CRITICAL: Baseline model returned all-zero predictions!
```

Sources:
- Train: `$HUB/cli/compare_models_dense_train_rerun.log:605,722`
- Test: `$HUB/cli/compare_models_dense_test_rerun.log:606,726-727`

## Analysis

The diagnostic logging added to `scripts/compare_models.py` (lines 1041-1068) successfully proves:

1. **Input validation**: Both train and test splits receive valid, non-zero diffraction data
2. **Model execution**: `baseline_model.predict([baseline_input, baseline_offsets])` completes without exceptions for both splits
3. **Output divergence**: Train produces valid predictions (mean=0.003092, 1.4M nonzero values), test returns all zeros

**Conclusion**: This is **NOT** a compare_models.py bug. The model itself behaves differently on test data.

## Root Cause Hypotheses

1. **Numerical underflow**: Test data characteristics (different coordinate distributions, batch sizes, or diffraction pattern statistics) may trigger underflow in the Baseline model's internal computations
2. **Model weight/config issue**: Baseline model may have trained with train-specific assumptions that fail on test data
3. **XLA compilation differences**: JIT compilation may produce different behavior for test vs train batch shapes
4. **Probe/object interaction failure**: Baseline model's internal ptychographic forward model may fail for test split's coordinate geometry

## Impact

- ❌ Phase G metrics/reporting fails (Baseline rows missing from `metrics_summary.json`)
- ❌ Cannot generate SSIM grid, verification logs, highlights, or preview
- ❌ `analysis/verification_report.json` remains 0/10
- ❌ Study publication blocked

## Required Investigation (Outside compare_models.py scope)

### Immediate Actions
1. **Inspect baseline model architecture**:
   ```bash
   ls -lha $HUB/data/phase_e/dose_1000/baseline/gs1/
   ```

2. **Compare train vs test NPZ characteristics**:
   ```python
   import numpy as np
   for split in ['train', 'test']:
       data = np.load(f'$HUB/data/phase_c/dose_1000/patched_{split}.npz', allow_pickle=True)
       print(f"{split}: shape={data['diffraction'].shape}, mean={data['diffraction'].mean()}, max={data['diffraction'].max()}")
   ```

3. **Check baseline training logs** for warnings about test data:
   ```bash
   grep -i "warn\|error\|test" $HUB/data/phase_e/dose_1000/baseline/gs1/train.log
   ```

4. **Try disabling XLA** for baseline inference to isolate compilation effects:
   ```python
   # In compare_models.py baseline inference section
   with tf.config.experimental.disable_jit():
       baseline_predictions = baseline_model.predict(...)
   ```

5. **Increase numerical precision** in baseline model forward pass (if model supports float64)

### Medium-term Solutions
- Retrain baseline model with:
  - Different numerical stability settings
  - Test data included in validation monitoring
  - Explicit dtype=float64 for critical operations
- Investigate whether this is a known TensorFlow/Keras issue with certain batch sizes
- Consider alternative baseline model architecture with better numerical stability

## Recommendations

**DO NOT** spend more time instrumenting `compare_models.py` - the diagnostic logging already proves the model is at fault.

**Next Focus**: Create a dedicated investigation task (e.g., `FIX-BASELINE-MODEL-ZERO-PREDICTIONS`) to debug the Baseline model's TensorFlow runtime behavior, or consider:
1. Proceeding with Phase G using only PINN vs PtyChi comparisons (document Baseline failure)
2. Retraining baseline model from scratch with different settings
3. Switching to a different baseline architecture

## Commands to Reproduce

Train (works):
```bash
cd /home/ollie/Documents/PtychoPINN
python scripts/compare_models.py \
  --pinn_dir $HUB/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir $HUB/data/phase_e/dose_1000/baseline/gs1 \
  --test_data $HUB/data/phase_c/dose_1000/patched_train.npz \
  --output_dir $HUB/analysis/dose_1000/dense/train \
  --ms-ssim-sigma 1.0 --register-ptychi-only
```

Test (fails with all zeros):
```bash
python scripts/compare_models.py \
  --pinn_dir $HUB/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir $HUB/data/phase_e/dose_1000/baseline/gs1 \
  --test_data $HUB/data/phase_c/dose_1000/patched_test.npz \
  --output_dir $HUB/analysis/dose_1000/dense/test \
  --ms-ssim-sigma 1.0 --register-ptychi-only
```

## Status

**BLOCKED** - Requires investigation outside compare_models.py scope.
**Assignee**: Requires decision on whether to:
- Debug baseline model internals
- Proceed without Baseline metrics
- Retrain baseline model
