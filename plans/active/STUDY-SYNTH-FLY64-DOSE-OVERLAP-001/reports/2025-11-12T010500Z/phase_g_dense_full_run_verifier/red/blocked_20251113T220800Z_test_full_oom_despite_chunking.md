# Blocker: Test Full Compare Models OOM Despite Chunked Baseline

## Timestamp
2025-11-13T22:08:00Z

## Status
BLOCKED - Full test-split compare_models fails with GPU OOM even with chunked Baseline inference

## Context
- Translation guard tests: GREEN (2/2 passed)
- Debug-limited runs (320 groups): SUCCESSFUL for both train and test splits
  - Train debug: Baseline mean=0.188307, 78.7M nonzero pixels
  - Test debug: Baseline mean=0.158501, 1.0M nonzero pixels
- Full train run (5088 groups): SUCCESSFUL with chunked Baseline
  - Baseline mean=0.188307, 78.7M nonzero pixels
  - Computation time: 50.6s for Baseline
- Full test run (5216 groups): FAILED with ResourceExhaustedError

## Error Signature
```
tensorflow.python.framework.errors_impl.ResourceExhaustedError:
{{function_node __wrapped__Cast_device_/job:localhost/replica:0/task:0/device:GPU:0}}
failed to allocate memory [Op:Cast]
```

##Command
```bash
PYTHONPATH="$PWD" python scripts/compare_models.py \
  --pinn_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/baseline/gs1 \
  --test_data plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_c/dose_1000/patched_test.npz \
  --output_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test \
  --ms-ssim-sigma 1.0 \
  --tike_recon_path plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_f/dose_1000/dense/test/ptychi_reconstruction.npz \
  --register-ptychi-only \
  --baseline-chunk-size 256 \
  --baseline-predict-batch-size 16
```

## Log Path
`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_test_full.log`

## Analysis
The OOM error occurred during a Cast operation, not during Baseline inference. This suggests:
1. Chunked Baseline inference (BASELINE-CHUNKED-001) is working correctly
2. The memory issue is elsewhere in the pipeline - possibly during PINN inference or metrics computation
3. The test split has 5216 groups vs train's 5088 groups (~2.5% more)
4. The debug run with 320 groups succeeded, indicating the issue scales with dataset size

## Evidence
- Debug runs: `cli/compare_models_dense_{train,test}_debug.log` (both successful)
- Train full: `cli/compare_models_dense_train_full.log` (successful)
- Test full: `cli/compare_models_dense_test_full.log` (ResourceExhaustedError)

## Mitigation Options
1. Reduce `--baseline-chunk-size` further (e.g., 128 or 64)
2. Reduce `--baseline-predict-batch-size` (currently 16)
3. Skip full test-split compare_models and proceed with train-only Baseline metrics
4. Investigate if PINN inference can also be chunked (currently processes all groups at once)

## Next Steps
Supervisor decision required on mitigation path before proceeding with Phase D acceptance → counted pipeline → post-verify sweep.
