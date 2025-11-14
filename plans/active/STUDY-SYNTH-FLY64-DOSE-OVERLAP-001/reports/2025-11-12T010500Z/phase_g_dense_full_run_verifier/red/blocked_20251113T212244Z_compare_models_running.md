# Long-running compare_models jobs in progress

**Status**: RUNNING (blocked per Ralph §0 — long-running job policy)
**Created**: 2025-11-13T21:22:44Z
**Loop**: Ralph implementation loop (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001)

## Summary
Both compare_models jobs for dense train/test splits were successfully launched following the offset-centering fix (commit 1ff0821a). Translation regression tests passed (2/2 GREEN). The jobs are actively running but will not complete within this loop's execution window, triggering the Ralph §0 long-running job policy requiring status documentation and escalation.

## Running jobs

### Train split (PID 2610948)
**Command**:
```bash
python -m scripts.compare_models \
  --pinn_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/baseline/gs1 \
  --test_data plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_c/dose_1000/patched_train.npz \
  --output_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train \
  --frc-sigma 0.5 \
  --phase-align-method plane
```

**Log path**: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_full.log`
**PID**: 2610948
**CPU**: 108% (active GPU inference)
**Memory**: 4.5 GB
**Progress**: Successfully loaded model, generated 5088 groups, running PINN/Baseline inference

### Test split (PID 2611189)
**Command**:
```bash
python -m scripts.compare_models \
  --pinn_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/baseline/gs1 \
  --test_data plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_c/dose_1000/patched_test.npz \
  --output_dir plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test \
  --frc-sigma 0.5 \
  --phase-align-method plane
```

**Log path**: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_test_full.log`
**PID**: 2611189
**CPU**: 108% (active GPU inference)
**Memory**: 4.3 GB
**Progress**: Successfully loaded model, generated groups, running PINN/Baseline inference

## Expected completion signal
- Both jobs will complete when they finish writing:
  - `analysis/dose_1000/dense/{split}/comparison_metrics.csv` (with Baseline/PINN/PtyChi rows)
  - `analysis/dose_1000/dense/{split}/logs/logs/debug.log` (tail shows DIAGNOSTIC baseline_output stats)
  - NPZ outputs and visualizations

- Success criteria (from Brief):
  - Baseline rows present in `comparison_metrics.csv` for both splits
  - `debug.log` tail shows `DIAGNOSTIC baseline_output stats:` with **non-zero mean** and **nonzero_count > 0** (not `mean=0.000000 ... nonzero_count=0`)

## Return condition
Next Ralph loop should:
1. Check if both jobs completed successfully (exit code 0)
2. Verify Baseline rows exist in `comparison_metrics.csv` for train AND test
3. Tail `debug.log` for each split and confirm non-zero Baseline stats
4. If either split shows zero Baseline outputs, create new blocker under `red/blocked_<timestamp>_baseline_zero_<split>.md` per Brief instructions
5. If both splits show healthy Baseline outputs, proceed with Phase D acceptance tests → counted `run_phase_g_dense.py --clobber` → metrics helpers → `--post-verify-only` until verification 10/10

## Artifacts
- `green/pytest_compare_models_translation_fix_v14.log` (2/2 PASSED, 6.19s)
- `cli/compare_models_dense_train_full.log` (in progress)
- `cli/compare_models_dense_test_full.log` (in progress)
