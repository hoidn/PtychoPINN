Brief:
Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier"`, then chase the RED pytest evidence in `plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/reports/pytest_translation_fix.log` by instrumenting `_reassemble_position_batched` / `ReassemblePatchesLayer` so the batched path always feeds `Translation` matching canvas + offset shapes (add assertions/logs if it helps).
Once the fix is in, rerun `pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv | tee "$HUB"/green/pytest_compare_models_translation_fix.log` and keep the selector GREEN before proceeding.
Reproduce the train/test CLI commands verbatim (`scripts/compare_models.py ... --split train` / `--split test`) capturing logs under `$HUB/cli/phase_g_dense_translation_fix_{split}.log`; stash blockers under `$HUB/red/blocked_<timestamp>*.md` if any Translation stack trace remains.
Only hand back to STUDY-SYNTH after both CLI runs exit 0, `analysis/dose_1000/dense/{train,test}/comparison_metrics.csv` refresh, and `analysis/blocker.log` is either removed or rewritten with a success note.

Summary: plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/summary.md
Plan: plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md
Selector: pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv
