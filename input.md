Brief:
Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`, then rerun `scripts/compare_models.py` twice (train/test NPZs, no `--split` flag) so the Translation failure is freshly captured under `$HUB/cli/phase_g_dense_translation_fix_{train,test}.log` (blockers → `$HUB/red/blocked_<timestamp>.md`).
Update `ptycho/custom_layers.ReassemblePatchesLayer` to log when batching engages and call `hh.mk_reassemble_position_batched_real` only when `total_patches > batch_size`, then refactor `_reassemble_position_batched` to derive `padded_size` from kwargs/`params`, switch to a single `tf.image.resize_with_crop_or_pad` per batch, and add `tf.debugging.assert_equal` shape/dtype guards plus padded-size/crop counters in the log.
Keep `pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv | tee "$HUB"/green/pytest_compare_models_translation_fix.log` GREEN after the refactor and rerun the train/test compare_models commands to regenerate `analysis/dose_1000/dense/{train,test}` metrics bundles.
Finish by clearing/rewriting `$HUB/analysis/blocker.log` and `{analysis}/verification_report.json`, and update the fixation plan + summary with MS-SSIM/MAE deltas and command/log references once the hub reaches 10/10 validity.

Summary: plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/summary.md
Plan: plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md
Selector: pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv
