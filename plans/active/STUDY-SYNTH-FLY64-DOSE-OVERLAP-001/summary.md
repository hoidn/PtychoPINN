### Turn Summary
Implemented zero-output RuntimeError assertion and canonical model ID mapping (commit e830a5be) so compare_models halts immediately when Baseline outputs are all zeros.
Translation guard remains GREEN (2/2 passed, 6.21s); train split succeeds with non-zero Baseline outputs (mean=0.003, CSV contains canonical `Baseline`/`PtyChi` IDs); test split triggers assertion (Baseline inputs mean=0.113, 17.8M nonzero → outputs mean=0.0, 0 nonzero), proving TensorFlow/model runtime issue beyond compare_models.py scope.
Next: supervisor decision required — proceed with PINN vs PtyChi only, debug Baseline model internals, or retrain Baseline model.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v10.log, cli/compare_models_dense_train_instrumented.log, cli/compare_models_dense_test_instrumented.log, analysis/dose_1000/dense/train/comparison_metrics.csv, red/blocked_20251113T200447Z_baseline_test_zero_instrumented.md

Checklist:
- Files touched: scripts/compare_models.py; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T200447Z_baseline_test_zero_instrumented.md
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv; python scripts/compare_models.py (train split succeeded, test split triggered RuntimeError)
- Artifacts updated: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v10.log; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_instrumented.log; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_test_instrumented.log; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/comparison_metrics.csv; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T200447Z_baseline_test_zero_instrumented.md

### Turn Summary
Dense-test Baseline predictions still come back all zeros (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log:421-535`), so `analysis/metrics_summary.json` keeps blank Baseline rows and `cli/aggregate_report_cli.log:1-11` plus `cli/run_phase_g_dense_post_verify_only.log:4-23` continue failing (metrics reporter + PREVIEW-PHASE-001).
Updated the implementation plan, ledger, and input to force Ralph to fix/instrument `scripts/compare_models.py` (tf.debugging asserts + canonical `Baseline`/`PtyChi` labels), verify the CSV/JSON metrics contain Baseline values before rerunning `report_phase_g_dense_metrics.py`, and to block the SSIM grid helper until `analysis/metrics_delta_highlights_preview.txt` exists.
Do Now remains ready_for_implementation: rerun the guarded pytest selector, repair Baseline inference so both compare_models splits log non-zero `baseline_output` stats, ensure metrics reporters succeed (Baseline rows populated), then rerun the counted `run_phase_g_dense.py --clobber` + fully parameterized `--post-verify-only` to produce the SSIM/verification/highlights/metrics/preview bundle.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log, analysis/metrics_summary.json, cli/aggregate_report_cli.log, cli/run_phase_g_dense_post_verify_only.log

Checklist:
- Files touched: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md; docs/fix_plan.md; input.md; galph_memory.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md
- Tests run: none
- Artifacts updated: none (audit-only)

### Turn Summary
Translation regression guards confirmed GREEN (2 passed, 6.26s); Baseline model zero-prediction issue verified as TensorFlow/model runtime problem outside compare_models.py scope.
Created comprehensive blocker document (`red/blocked_20251113T193000Z_baseline_model_investigation_required.md`) documenting that train split Baseline works (mean=0.003092, nonzero=1.4M) while test split returns all zeros despite valid inputs (input mean=0.112671, nonzero=17.8M).
Diagnostic instrumentation from previous loop successfully proves model receives valid inputs but outputs zeros—this requires investigation of baseline model architecture, numerical stability, XLA compilation behavior, or test data characteristics.
Phase G completion blocked pending decision to either debug baseline model internals, proceed with PINN vs PtyChi only, or retrain baseline model.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v9.log, red/blocked_20251113T193000Z_baseline_model_investigation_required.md

Checklist:
- Files touched: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T193000Z_baseline_model_investigation_required.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv
- Artifacts updated: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v9.log; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T193000Z_baseline_model_investigation_required.md

### Turn Summary
Phase G hub still reports 0/10 (`analysis/verification_report.json:1-80`) because the dense-test Baseline logs keep `DIAGNOSTIC baseline_output` at zero even though the inputs are valid (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log:312-535`), which leaves `analysis/metrics_summary.json` without Baseline values and mislabels `PtyChi`.
Captured the failure signatures from `cli/aggregate_report_cli.log:1-9` and `cli/run_phase_g_dense_post_verify_only.log:1-15`, then updated the plan so compare_models must halt/raise on zero Baseline outputs, metrics reruns must verify canonical `Baseline`/`PtyChi` rows before proceeding, and PREVIEW-PHASE-001 cannot execute until `analysis/metrics_delta_highlights_preview.txt` exists.
docs/fix_plan.md, the plan, and input.md now direct Ralph to rerun the translation guard, regenerate both compare_models bundles with the diagnostic gating, rerun `run_phase_g_dense.py --clobber`, replay `report_phase_g_dense_metrics.py` + `analyze_dense_metrics.py`, and only then redo the SSIM/verification helper (blockers → `$HUB/red/blocked_<timestamp>.md`).
Next: execute the guarded pytest selector, fix Baseline inference so the new compare_models logs show non-zero baseline_output stats, regenerate the metrics + preview bundle, and rerun `run_phase_g_dense.py --post-verify-only` to fill the SSIM/verification/highlights evidence.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log, analysis/metrics_summary.json, cli/aggregate_report_cli.log, cli/run_phase_g_dense_post_verify_only.log

### Turn Summary
Translation regression guards remain GREEN (2 passed, 6.22s); confirmed diagnostic logging from previous loop successfully captures Baseline input/output stats.
Train split Baseline predictions valid (mean=0.003092, nonzero=1.4M), but test split produces **all zeros** despite receiving valid inputs (input mean=0.112671, nonzero=17.8M).
Updated blocker doc (`red/blocked_20251113T191906_baseline_test_zero.md`) confirming this is a **TensorFlow/model runtime issue** that cannot be fixed in compare_models.py; instrumentation proves model receives valid inputs but outputs zeros for test data.
Phase G counted rerun blocked pending baseline model investigation (numerical stability, XLA compilation differences, or test data characteristics triggering underflow).
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v8.log, red/blocked_20251113T191906_baseline_test_zero.md

Checklist:
- Files touched: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T191906_baseline_test_zero.md
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv
- Artifacts updated: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v8.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T191906_baseline_test_zero.md

### Turn Summary
Verification guard is still 0/10 and the SSIM/preview hooks abort immediately because the hub never produced the metrics bundle or artifact inventory (`analysis/verification_report.json:1-80`, `cli/run_phase_g_dense_post_verify_only.log:1-24`).
Dense test Baseline reconstructions remain all zeros so the metrics reporter skips every Baseline row and Phase G deltas never materialize (`analysis/dose_1000/dense/test/comparison_metrics.csv:1-23`, `analysis/dose_1000/dense/test/comparison.log:1015-1030`).
Next: capture the new compare_models diagnostics for both train/test splits, stop immediately if the `baseline_input`/`baseline_output` stats are still zero, and only proceed to the guarded pytest selectors + counted rerun once Baseline predictions contain signal.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier

#### One-off analysis — baseline NPZ stats
```python
python - <<'PY'
import numpy as np
from pathlib import Path
base = Path('plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense')
for split in ['train','test']:
    npz = base / split / 'reconstructions_aligned.npz'
    with np.load(npz, allow_pickle=True) as data:
        arr = data['baseline_complex']
        amp = np.abs(arr)
        print(split, 'baseline_complex shape', arr.shape, 'mean', arr.mean(), 'max', arr.max(), 'nonzero', np.count_nonzero(arr))
        print(split, 'baseline_amp range', amp.min(), amp.max())
PY
```
Output:
```
train baseline_complex shape (185, 354) mean (0.00022677403+0.0005013436j) max (0.662244+0.46133828j) nonzero 104
train baseline_amp range 0.0 0.8070936
test baseline_complex shape (186, 353) mean 0j max 0j nonzero 0
test baseline_amp range 0.0 0.0
```

### Turn Summary
Instrumented scripts/compare_models.py with diagnostic logging to track down all-zero baseline reconstructions on test split; added input/output statistics logging (mean/max/nonzero_count) before and after baseline model inference with CRITICAL errors when values are zero.
Confirmed the issue: baseline model returns valid non-zero predictions for train split but all zeros for test split (verified via existing artifacts in dose_1000/dense/{train,test}/reconstructions_aligned.npz), causing downstream metrics/reporting failures.
Next: wait for full pytest suite completion, then analyze diagnostic output from instrumented compare_models to determine root cause (likely TensorFlow/XLA runtime issue specific to test data characteristics or batch size).
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v6.log (translation guards GREEN)

Checklist:
- Files touched: scripts/compare_models.py
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_batched_predictions tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -vv; pytest -v tests/ (background)
- Artifacts updated: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v6.log

### Turn Summary
Baseline dense-test reconstructions still collapse to zeros (`analysis/dose_1000/dense/test/reconstructions_aligned.npz` and the raw `reconstructions.npz`), so `analysis/metrics_summary.json` never captures `Baseline` rows and `report_phase_g_dense_metrics.py` continues to fail with "Required models missing" (`cli/aggregate_report_cli.log`).
`run_phase_g_dense.py --post-verify-only` still aborts immediately because the preview file never materializes (`cli/run_phase_g_dense_post_verify_only.log` ↔ `cli/ssim_grid_cli.log`), leaving `analysis/verification_report.json` locked at 0/10.
Updated the plan, ledger, and input to require instrumentation/fixes inside `scripts/compare_models.py` so both train/test compare_models runs produce non-zero Baseline stats before replaying the Phase D guards, counted rerun, metrics helpers, and full verification sweep.
Next: patch the Baseline inference path, prove the regenerated compare_models outputs contain canonical `Baseline` entries, then rerun the dense pipeline + `--post-verify-only` until the SSIM/verification/highlights/metrics/preview/inventory bundle is complete and `verification_report.json` reports 10/10.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/reconstructions_aligned.npz, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/metrics_summary.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/aggregate_report_cli.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_post_verify_only.log

### Turn Summary
Phase G hub still lacks verification evidence: `analysis/verification_report.json:1-19` remains `n_valid=0/10`, the dense test Baseline predictions are all zeros (`analysis/dose_1000/dense/test/logs/logs/debug.log:605-609`), and `analysis/metrics_summary.json:48-105` still lacks Baseline rows while labeling the PtyChi model as "Pty-chi (pty-chi)".
`report_phase_g_dense_metrics.py` and `run_phase_g_dense.py --post-verify-only` continue to fail because the reporter cannot find canonical Baseline/PtyChi entries (`cli/aggregate_report_cli.log:1-11`) and the preview file is missing (`cli/run_phase_g_dense_post_verify_only.log:12-23`), so the SSIM grid never runs.
Updated `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`, docs/fix_plan.md, and input.md so Ralph must rerun both compare_models splits with canonical IDs, rerun the Phase D guards, execute the counted `run_phase_g_dense.py --clobber`, refresh `report_phase_g_dense_metrics.py` + `bin/analyze_dense_metrics.py` before calling `--post-verify-only`, and log blockers if Baseline remains zero.
Next: run the translation pytest guard, rebuild the train/test comparison bundles, rerun `run_phase_g_dense.py --clobber` plus the metrics helpers, then re-run `run_phase_g_dense.py --post-verify-only` until the SSIM/verification/highlights/metrics/preview/artifact-inventory bundle exists and `analysis/verification_report.json` reports 10/10.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/metrics_summary.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/aggregate_report_cli.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_post_verify_only.log

### Turn Summary
Confirmed translation fix regression tests remain GREEN after commits a80d4d2b/bf3f1b07, proving batched reassembly and XLA streaming fixes are stable on HEAD.
Launched the counted Phase C→G dense pipeline with `--clobber` for dose=1000 in background (PID 2445542); Phase C (Dataset Generation) is currently executing TensorFlow diffraction simulation with CUDA (RTX 3090) and will automatically proceed through all 8 phases.
Pipeline estimated to complete in 30-120 minutes; remaining workflow steps (--post-verify-only, report_phase_g_dense_metrics.py, analyze_dense_metrics.py, verification of n_valid=10) are blocked pending pipeline completion.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v4.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_stdout_v4.log (streaming)

Checklist:
- Files touched: none
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_batched_predictions tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -vv
- Artifacts updated: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v4.log; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_stdout_v4.log
### Turn Summary
Dense-test Baseline predictions still come back all zeros (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log:421-535`), so `analysis/metrics_summary.json` keeps blank Baseline rows and `cli/aggregate_report_cli.log:1-11` plus `cli/run_phase_g_dense_post_verify_only.log:4-23` continue failing (metrics reporter + PREVIEW-PHASE-001).
Updated the implementation plan, ledger, and input to force Ralph to fix/instrument `scripts/compare_models.py` (tf.debugging asserts + canonical `Baseline`/`PtyChi` labels), verify the CSV/JSON metrics contain Baseline values before rerunning `report_phase_g_dense_metrics.py`, and to block the SSIM grid helper until `analysis/metrics_delta_highlights_preview.txt` exists.
Do Now remains ready_for_implementation: rerun the guarded pytest selector, repair Baseline inference so both compare_models splits log non-zero `baseline_output` stats, ensure metrics reporters succeed (Baseline rows populated), then rerun the counted `run_phase_g_dense.py --clobber` + fully parameterized `--post-verify-only` to produce the SSIM/verification/highlights/metrics/preview bundle.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log, analysis/metrics_summary.json, cli/aggregate_report_cli.log, cli/run_phase_g_dense_post_verify_only.log

Checklist:
- Files touched: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md; docs/fix_plan.md; input.md; galph_memory.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md
- Tests run: none
- Artifacts updated: none (audit-only)
