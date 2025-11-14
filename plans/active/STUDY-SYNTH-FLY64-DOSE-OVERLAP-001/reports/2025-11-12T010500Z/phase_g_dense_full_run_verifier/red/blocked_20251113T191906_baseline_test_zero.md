# Blocker: Baseline test split returns all-zero predictions

**Timestamp:** 2025-11-13T19:18:30Z  
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Blocking:** Phase G counted dense rerun

## Problem

The baseline model returns **all-zero predictions** for the test split, while the train split produces non-zero values.

## Evidence

### Train split (SUCCESS):
```
2025-11-13 19:14:34,317 - INFO - DIAGNOSTIC baseline_input stats: mean=0.112964, max=6.064422, nonzero_count=17447360/333447168
2025-11-13 19:14:51,413 - INFO - DIAGNOSTIC baseline_output stats: mean=0.003092, max=1.295571, nonzero_count=1387120/333447168
```
Source: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_rerun.log:605,722`

### Test split (FAILURE):
```
2025-11-13 19:17:38,462 - INFO - DIAGNOSTIC baseline_input stats: mean=0.112671, max=5.362341, nonzero_count=17793971/341835776
2025-11-13 19:17:55,801 - INFO - DIAGNOSTIC baseline_output stats: mean=0.000000, max=0.000000, nonzero_count=0/341835776
2025-11-13 19:17:55,802 - ERROR - CRITICAL: Baseline model returned all-zero predictions! Check model weights and input data.
```
Source: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_test_rerun.log:606,726-727`

## Impact

- Phase G metrics/reporting will fail because Baseline rows remain empty for test split
- `report_phase_g_dense_metrics.py` will continue to error with "Required models missing"
- Verification bundle (SSIM/highlights/preview) cannot be generated without valid Baseline metrics
- Cannot proceed with counted Phase G rerun until this is resolved

## Next Actions

This is a TensorFlow/XLA runtime or model-specific issue that affects test data differently than train data. Possible causes:
1. Different data characteristics between train/test splits causing numerical underflow
2. Model weights or configuration issue specific to test batch sizes
3. XLA compilation behavior differing between splits
4. Baseline model's probe handling for test split coordinate distribution

Recommend:
1. Investigate baseline model weights/architecture for numerical stability
2. Compare train vs test NPZ coordinate distributions and batch characteristics
3. Check for any test-specific preprocessing or normalization differences
4. Consider disabling XLA for baseline inference to isolate compilation effects

## Commands to reproduce

Train (works):
```bash
python scripts/compare_models.py \
  --pinn_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/baseline/gs1 \
  --test_data plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_c/dose_1000/patched_train.npz \
  --output_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train \
  --ms-ssim-sigma 1.0 --register-ptychi-only
```

Test (fails - all zeros):
```bash
python scripts/compare_models.py \
  --pinn_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/baseline/gs1 \
  --test_data plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_c/dose_1000/patched_test.npz \
  --output_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test \
  --ms-ssim-sigma 1.0 --register-ptychi-only
```
