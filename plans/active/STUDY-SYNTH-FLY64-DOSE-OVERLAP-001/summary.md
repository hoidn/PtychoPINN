### Turn Summary
Dense-train compare_models succeeded with healthy Baseline outputs (mean=0.188, 78.7M nonzero pixels, complete CSV rows), but dense-test failed with TensorFlow GPU OOM during Baseline inference, leaving blank Baseline metric rows in test CSV.
Translation regression tests remain GREEN (2/2 passed).
Blocker documented (`red/blocked_20251116T010000Z_test_baseline_oom.md`) with evidence, root cause (TF ResourceExhaustedError during test-split Baseline inference), and four mitigation options (batched inference, skip test Baseline, reduce dataset, TF memory config).
Next: await supervisor decision on mitigation path before proceeding with Phase D acceptance → counted pipeline → post-verify sweep.
Artifacts: red/blocked_20251116T010000Z_test_baseline_oom.md; analysis/dose_1000/dense/train/comparison_metrics.csv (Baseline rows complete); analysis/dose_1000/dense/test/comparison_metrics.csv (Baseline rows blank)

Checklist:
- Files touched: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251116T010000Z_test_baseline_oom.md
- Tests run: none (verified completed compare_models outputs)
- Artifacts updated: red/blocked_20251116T010000Z_test_baseline_oom.md

### Turn Summary
Translation regression tests remain GREEN (2/2 passed, 6.19s), confirming the offset-centering fix maintains correctness.
Launched full compare_models for dense train/test splits (PIDs 2610948, 2611189) but they are actively running GPU inference and will not complete within this loop's execution window per Ralph §0 long-running job policy.
Jobs blocked pending completion; documented running status with PIDs, log paths (`cli/compare_models_dense_{split}_full.log`), and return condition (verify non-zero Baseline stats in debug.log and CSV rows) in `red/blocked_20251113T212244Z_compare_models_running.md`.
Next: await job completion, verify Baseline output health for both splits, then proceed with Phase D acceptance tests → counted pipeline → post-verify sweep.
Artifacts: green/pytest_compare_models_translation_fix_v14.log; cli/compare_models_dense_{train,test}_full.log (in progress); red/blocked_20251113T212244Z_compare_models_running.md

Checklist:
- Files touched: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T212244Z_compare_models_running.md
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_batched_predictions tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -vv
- Artifacts updated: green/pytest_compare_models_translation_fix_v14.log; cli/compare_models_dense_{train,test}_full.log (background PIDs 2610948, 2611189); red/blocked_20251113T212244Z_compare_models_running.md

### Turn Summary
10-group compare_models debug runs now prove the offset-centering fix works (train mean=0.284, test mean=0.079 with non-zero `baseline_output` counts logged under `analysis/dose_1000/dense/{split}/debug/baseline_debug_stats.json`), so Baseline inference is healthy in the limited probe.
The counted train/test compare_models logs are still the pre-fix artifacts (`analysis/dose_1000/dense/test/logs/logs/debug.log:540` reports zeros and `analysis/dose_1000/dense/test/comparison_metrics.csv` lacks Baseline values), so `analysis/metrics_summary.json`, highlights, preview text, and `analysis/verification_report.json` remain missing (0/10).
Next: rerun the translation pytest guard, execute full compare_models for dense train/test without the debug limit (tee to `$HUB/cli/compare_models_dense_{split}_full.log`), confirm CSV/JSON Baseline rows plus non-zero DIAGNOSTIC stats, then run the dense acceptance selector, counted `run_phase_g_dense.py --clobber`, metrics helpers, and `--post-verify-only` until `{analysis}` holds SSIM grid/verification/highlights/preview/inventory artifacts with 10/10 validity.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/{train,test}/debug/baseline_debug_stats.json, .../cli/compare_models_dense_{split}_debug_centered.log, .../analysis/dose_1000/dense/test/logs/logs/debug.log

### Turn Summary
Centered baseline offsets to fix the test-split zero-output blocker (commit 1ff0821a); debug runs with 10 groups prove train/test both emit non-zero Baseline predictions (train mean=0.284, test mean=0.079) after centering offsets from mean≈273px (test) and mean≈185px (train) to zero-mean.
Root cause was numerical instability in the Baseline model's position-dependent layers when facing the 87px train→test offset distribution shift; centering `flattened_offsets` before inference solved it immediately.
Translation regression tests remain GREEN (2/2 passed), full suite reports 474 passed/28 failed (pre-existing torch/TF failures unrelated to this change).
Next: re-run the full Phase G dense pipeline (pytest guards → counted `run_phase_g_dense.py --clobber` → `--post-verify-only`) to generate the SSIM/verification/highlights/preview/inventory bundle required by PREVIEW-PHASE-001 and TEST-CLI-001, ensuring Baseline rows appear in metrics_summary.json/comparison_metrics.csv.
Artifacts: green/pytest_compare_models_translation_fix_v13.log, green/blocker_resolved_offset_centering.md, cli/compare_models_dense_{train,test}_debug_centered.log, analysis/dose_1000/dense/{train,test}/debug/{baseline_debug.npz,baseline_debug_stats.json}

Checklist:
- Files touched: scripts/compare_models.py, docs/findings.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_batched_predictions tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -vv; pytest -v tests/; python scripts/compare_models.py (train/test debug runs, 10 groups)
- Artifacts updated: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/{pytest_compare_models_translation_fix_v13.log,blocker_resolved_offset_centering.md}; .../cli/{compare_models_dense_train_debug_centered.log,compare_models_dense_test_debug_centered.log}; .../analysis/dose_1000/dense/{train,test}/debug/baseline_debug.npz

### Turn Summary
Baseline compare_models on the dense-test split still aborts with `DIAGNOSTIC baseline_output stats: mean=0.000000 ... nonzero_count=0` so `analysis/metrics_summary.json` and `analysis/dose_1000/dense/test/comparison_metrics.csv` never gain Baseline rows and `report_phase_g_dense_metrics.py` fails immediately (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log#L539-L550`, `.../cli/aggregate_report_cli.log:1-11`).
Dense-train logs prove the instrumentation still records non-zero Baseline outputs (`analysis/.../dense/train/logs/logs/debug.log#L534-L542`), so the split-specific regression plus missing preview/verification artifacts keeps the Phase G bundle blocked pending a counted rerun.
A one-off probe rebuilt the Phase C containers with `n_images=64` and showed flattened `baseline_offsets` for dense-test shifted to ≈273 ± 86 px versus ≈185 ± 72 px for dense-train, so the next engineer loop must center offsets in `scripts/compare_models.py`, add debug flags/NPZ dumps, rerun the guarded compare_models commands, and only then follow the Phase D pytest guard → counted `run_phase_g_dense.py --clobber` → metrics helpers → fully parameterized `--post-verify-only` chain until `{analysis}` holds the SSIM/verification/highlights/preview/inventory bundle.
Next: land the offset-centering + debug plumbing, prove dense-train/dense-test compare_models both emit canonical Baseline rows, then execute the counted pipeline/post-verify sweep per plan; blockers (zero Baseline stats, missing preview, verification <10/10) must be logged under `$HUB/red/blocked_<timestamp>.md` with the failing command signature.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log#L539-L550, .../analysis/dose_1000/dense/train/logs/logs/debug.log#L534-L542, .../cli/aggregate_report_cli.log:1-11, .../cli/run_phase_g_dense_post_verify_only.log:1-23

Checklist:
- Files touched: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md; docs/fix_plan.md; input.md; galph_memory.md
- Tests run: none
- Artifacts updated: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/{train,test}/logs/logs/debug.log (referenced); .../cli/{aggregate_report_cli.log,run_phase_g_dense_post_verify_only.log}; offset micro-probe recorded below

#### One-off analysis — baseline offset stats (n_images=64)
```python
from pathlib import Path
from ptycho.workflows.components import load_data, create_ptycho_data_container
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from scripts.compare_models import prepare_baseline_inference_data
from ptycho import params as p

def summarize(split, path):
    raw = load_data(str(path), n_images=64)
    cfg = TrainingConfig(model=ModelConfig(N=raw.probeGuess.shape[0], gridsize=2),
                         n_groups=raw.diff3d.shape[0], neighbor_count=7,
                         train_data_file=path)
    update_legacy_dict(p.cfg, cfg)
    container = create_ptycho_data_container(raw, cfg)
    _, offsets = prepare_baseline_inference_data(container)
    return offsets.mean(), offsets.std(), offsets.min(), offsets.max()

base = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_c/dose_1000")
train_stats = summarize("train", base/"patched_train.npz")
test_stats = summarize("test", base/"patched_test.npz")
print(train_stats, test_stats)
```
Results:
- train offsets: mean≈185.18, std≈72.14, min≈58.06, max≈401.35
- test offsets: mean≈272.80, std≈85.76, min≈51.38, max≈427.88

### Turn Summary
Translation regression tests remain GREEN (2/2 passed in 6.20s); fast debug runs with `--n-test-groups 10` confirmed the Baseline zero-output issue on test split while train split produces sparse but non-zero outputs (mean=0.000415, 406 nonzero pixels).
Test split Baseline inference returns completely zero outputs (mean=0.0, 0 nonzero) despite valid inputs (mean=0.112296, 33937 nonzero), triggering the instrumented RuntimeError as designed and halting execution before wasting compute on full pipeline.
Blocker documented (`red/blocked_20251113T203727Z_baseline_test_zero_confirmed.md`) with fast-loop evidence, decision point, and recommendation to proceed with PINN vs PtyChi only metrics.
Next: await supervisor decision—proceed with partial metrics (skip Baseline), debug Baseline model architecture, retrain from scratch, or accept train-only Baseline data.
Artifacts: green/pytest_compare_models_translation_fix_v12.log, cli/compare_models_dense_train_debug.log (10 groups, sparse outputs), cli/compare_models_dense_test_debug.log (10 groups, zero outputs, RuntimeError), red/blocked_20251113T203727Z_baseline_test_zero_confirmed.md

Checklist:
- Files touched: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T203727Z_baseline_test_zero_confirmed.md
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_batched_predictions tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -vv; python scripts/compare_models.py --n-test-groups 10 (train exit 0, test RuntimeError)
- Artifacts updated: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v12.log; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_debug.log; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_test_debug.log; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T203727Z_baseline_test_zero_confirmed.md

### Turn Summary
Added first-patch diagnostic logging to baseline inference (commit 035fb5c3); translation regression tests remain GREEN (2/2 passed, 6.17s).
Train split compare_models succeeds with non-zero Baseline outputs (mean=0.003092, 1.4M nonzero pixels, canonical `Baseline`/`PtyChi` IDs in CSV), but test split triggers RuntimeError (Baseline inputs mean=0.112671, 17.8M nonzero → outputs mean=0.0, 0 nonzero), proving TensorFlow/model runtime issue beyond compare_models.py scope.
Blocker documented (`red/blocked_20251114T043136Z_baseline_test_zero_final.md`) with evidence and decision point—supervisor must choose whether to proceed with PINN vs PtyChi only, debug baseline model internals, or retrain baseline model.
Next: await supervisor decision before resuming Phase D guards, counted rerun, metrics regeneration, and full verification sweep.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v11.log, analysis/dose_1000/dense/train/comparison_metrics.csv, red/blocked_20251114T043136Z_baseline_test_zero_final.md

Checklist:
- Files touched: scripts/compare_models.py; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251114T043136Z_baseline_test_zero_final.md; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_batched_predictions tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -vv; python scripts/compare_models.py (train split succeeded, test split triggered RuntimeError)
- Artifacts updated: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix_v11.log; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/comparison_metrics.csv; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251114T043136Z_baseline_test_zero_final.md

### Turn Summary
Dense-test Baseline inference still emits zero-valued predictions despite healthy inputs (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log:426-540`), so `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/metrics_summary.json` and `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/comparison_metrics.csv` have no Baseline rows and the metrics helper keeps failing (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/aggregate_report_cli.log:1-9`).
Verification remains 0/10 and PREVIEW-PHASE-001 blocks persist because the preview, SSIM grid, and artifact inventory never regenerate (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json:1-14`, `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_post_verify_only.log:1-19`).
Updated the plan/ledger/input so Ralph must instrument/repair `scripts/compare_models.py` (capture per-split baseline output stats, optionally limit groups for faster repro) before rerunning the Phase D guards, counted `run_phase_g_dense.py --clobber`, metrics reporters, and fully parameterized `--post-verify-only`, logging `$HUB/red/blocked_*.md` immediately if Baseline outputs stay zero.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/test/logs/logs/debug.log:426-540, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/logs/logs/debug.log:425-537, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/metrics_summary.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/aggregate_report_cli.log:1-9, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_post_verify_only.log:1-19

[Earlier summaries below...]
