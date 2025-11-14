# Test split Baseline inference OOM blocker

**Status**: BLOCKED (TensorFlow GPU memory exhaustion)
**Created**: 2025-11-16T01:00:00Z
**Loop**: Ralph implementation loop (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001)

## Summary
Dense-test Baseline inference failed with `ResourceExhaustedError: failed to allocate memory [Op:Cast]` during the full compare_models run (no `--n-test-groups` limit). Train split succeeded with healthy Baseline outputs (`mean=0.188307, nonzero_count=78705296/333447168`), but test split OOM'd before completing Baseline reconstruction, leaving blank Baseline metric rows in `comparison_metrics.csv`.

## Evidence

### Train split: SUCCESS
**Log**: `cli/compare_models_dense_train_full.log`
**Metrics CSV**: `analysis/dose_1000/dense/train/comparison_metrics.csv`
```csv
Baseline,mae,0.880613,1.161542,
Baseline,mse,1.157216,1.982887,
Baseline,psnr,47.496658,45.157825,
Baseline,ssim,0.004859,0.284683,
Baseline,ms_ssim,0.025108,0.271055,
Baseline,frc50,1.000000,1.000000,
```

**Debug diagnostics** (`analysis/dose_1000/dense/train/logs/logs/debug.log:533`):
```
2025-11-13 21:23:21,279 - INFO - DIAGNOSTIC baseline_output stats: mean=0.188307, max=1.673462, nonzero_count=78705296/333447168
```

### Test split: OOM FAILURE
**Log**: `cli/compare_models_dense_test_full.log` (tail shows ResourceExhaustedError)
**Metrics CSV**: `analysis/dose_1000/dense/test/comparison_metrics.csv`
```csv
Baseline,mae,,,
Baseline,mse,,,
Baseline,psnr,,,
Baseline,ssim,,,
Baseline,ms_ssim,,,
Baseline,frc50,,,
```

**Debug log** (`analysis/dose_1000/dense/test/logs/logs/debug.log:520-560`):
```
2025-11-13 21:22:50,357 - ERROR - ^
[... repeated ERROR lines ...]
2025-11-13 21:22:50,361 - ERROR - tensorflow.python.framework.errors_impl.ResourceExhaustedError: {{function_node __wrapped__Cast_device_/job:localhost/replica:0/task:0/device:GPU:0}} failed to allocate memory [Op:Cast]
```

**No DIAGNOSTIC baseline_output line** — inference never completed.

## Root cause
TensorFlow GPU memory exhaustion during dense-test Baseline inference. The test split container has 5088 groups (same as train), but the OOM occurs during inference, not data loading. Possible factors:
- TF graph compilation/optimization state from prior train run
- Fragmented GPU memory
- Baseline model memory footprint when processing full dense-test dataset

## Impact on pipeline
Per the Brief's success criteria and PREVIEW-PHASE-001 / TEST-CLI-001:
1. **Blocked**: Cannot proceed with Phase D acceptance selectors → counted `run_phase_g_dense.py --clobber` → metrics helpers → `--post-verify-only` until both train AND test splits have valid Baseline rows
2. **Missing artifacts**: `analysis/metrics_summary.json` will lack complete Baseline entries, `report_phase_g_dense_metrics.py` will fail with "Required models missing for delta computation: Baseline", and SSIM/verification/preview/inventory cannot be generated
3. **Verification stuck at 0/10**: `verification_report.json` remains incomplete

## Mitigation options

### Option 1: Batched inference (reduce GPU memory footprint)
Modify `scripts/compare_models.py` Baseline inference to process groups in smaller batches (e.g., 100-500 groups at a time) with explicit `tf.keras.backend.clear_session()` between batches. Requires code changes to `scripts/compare_models.py`.

**Pros**: Should resolve OOM without changing model or data
**Cons**: Implementation effort; need to ensure stitching/accumulation logic is correct
**Estimated effort**: 1-2 loops

### Option 2: Skip Baseline for test split, proceed with PINN vs PtyChi only
Accept train-only Baseline data, update `report_phase_g_dense_metrics.py` to handle missing Baseline gracefully, and generate partial metrics/verification bundle.

**Pros**: Unblocks Phase G pipeline immediately
**Cons**: Incomplete comparison data; undermines study goal of 3-model comparison
**Estimated effort**: <1 loop (conditional logic in metrics reporter)

### Option 3: Reduce test split size for dense view
Regenerate Phase C containers with smaller `n_images` for test split (e.g., 64 or 128 instead of full dataset), rerun Phase E/F/G.

**Pros**: Guaranteed to fit in GPU memory
**Cons**: Requires full pipeline rerun (Phase C onward); changes study parameters
**Estimated effort**: 3-4 loops (Phase C → Phase G)

### Option 4: GPU memory optimization (TF config)
Set TensorFlow to use memory growth or explicit memory limits, clear session between models, ensure XLA is not over-allocating.

**Pros**: Minimal code changes
**Cons**: May not be sufficient; TF memory fragmentation issues can persist
**Estimated effort**: 1 loop (configuration + retry)

## Recommendation
**Option 1 (batched Baseline inference)** is the most robust solution that preserves the full dataset and 3-model comparison. If supervisor prioritizes speed over completeness, **Option 2** (skip Baseline for test, proceed with PINN/PtyChi) can unblock the pipeline immediately but should be marked as a limitation in the study results.

## Required decision
Supervisor must choose mitigation path before next Ralph loop can proceed with:
- Phase D acceptance selectors (`pytest tests/study/test_dose_overlap_overlap.py::...`)
- Counted `run_phase_g_dense.py --clobber`
- Metrics reporters (`report_phase_g_dense_metrics.py`, `analyze_dense_metrics.py`)
- Fully parameterized `--post-verify-only`

## Artifacts
- Train split evidence: `green/pytest_compare_models_translation_fix_v14.log` (2/2 PASSED), `analysis/dose_1000/dense/train/comparison_metrics.csv` (Baseline rows present), `analysis/dose_1000/dense/train/logs/logs/debug.log:533` (non-zero diagnostics)
- Test split evidence: `analysis/dose_1000/dense/test/comparison_metrics.csv` (blank Baseline rows), `analysis/dose_1000/dense/test/logs/logs/debug.log:520-560` (OOM error), `cli/compare_models_dense_test_full.log` (tail shows ResourceExhaustedError)
