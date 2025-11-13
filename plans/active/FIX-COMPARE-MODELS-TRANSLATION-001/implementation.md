# Initiative: FIX-COMPARE-MODELS-TRANSLATION-001

Initiative Header
- ID: FIX-COMPARE-MODELS-TRANSLATION-001
- Title: Dense Phase G translation guard
- Owner/Date: Ralph / 2025-11-14 (Galph handoff)
- Status: in_progress (Critical — Tier 2 dwell mitigation)
- Working Plan: this file
- Linked Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 (blocked until this fix lands)
- Reports Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`

> Treat this plan as the evidence/control center for the translation fix. Once the tests + CLI reproductions succeed, hand the hub back to the parent initiative to resume the counted Phase G rerun.

<plan_update version="1.0">
  <trigger>`analysis/blocker.log` and `analysis/dose_1000/dense/train/comparison.log:520-540` prove the counted Phase G compare_models call still explodes inside the PINN `Translation` layer (`inputs=['tf.Tensor(shape=(32, 138, 138, 1)...', 'tf.Tensor(shape=(128, 2)...']`), so Ralph needs a dedicated focus to stream/batch the reassembly path before rerunning the pipeline.</trigger>
  <focus_id>FIX-COMPARE-MODELS-TRANSLATION-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (POLICY-001 / ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/specs/spec-ptychopinn.md, docs/specs/spec-ptycho-runtime.md, docs/specs/spec-ptycho-workflow.md, docs/specs/spec-ptycho-interfaces.md, docs/specs/spec-ptycho-tracing.md, docs/DEVELOPER_GUIDE.md, docs/architecture.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, specs/ptychodus_api_spec.md, specs/overlap_metrics.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,summary.md}, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/{summary.md,analysis/blocker.log,analysis/dose_1000/dense/train/comparison.log}</documents_read>
  <current_plan_path>plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md</current_plan_path>
  <proposed_changes>Capture the failure evidence, authorize targeted edits to `ptycho/custom_layers.py` + `ptycho/tf_helper.py`, spell out the new regression coverage, and define a reproducible Do Now (guarded pytest selectors + direct `scripts/compare_models.py` invocations) so Ralph can unblock STUDY-SYNTH quickly.</proposed_changes>
  <impacts>Phase G deliverables (verification, highlights, metrics, preview) remain blocked until this fix lands; touching custom layers/tf_helper is usually forbidden, so we must document the scope carefully.</impacts>
  <ledger_updates>Added FIX-COMPARE-MODELS-TRANSLATION-001 to docs/fix_plan.md, rewrote input.md, and prepended the new Turn Summary to `plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/summary.md`.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.1">
  <trigger>`plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/reports/pytest_translation_fix.log` shows `test_pinn_reconstruction_reassembles_full_train_split` still failing with `InvalidArgumentError: required broadcastable shapes` inside `tf_helper._reassemble_position_batched` (canvas and batch_result are mismatched when `ReassemblePatchesLayer` runs with B=159, C=4).</trigger>
  <focus_id>FIX-COMPARE-MODELS-TRANSLATION-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (REASSEMBLY-BATCH-001 / ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / DATA-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/DEVELOPER_GUIDE.md, docs/architecture.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, specs/ptychodus_api_spec.md, specs/overlap_metrics.md, docs/fix_plan.md, plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/{implementation.md,summary.md,reports/pytest_translation_fix.log}, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{summary.md,implementation.md}, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/comparison.log, input.md</documents_read>
  <current_plan_path>plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md</current_plan_path>
  <proposed_changes>Keep the batching-scope authorization but note the regression test already exists and is RED; refine the checklist/Do Now so Ralph focuses on instrumenting `_reassemble_position_batched`, hardening `ReassemblePatchesLayer`, and rerunning both the targeted pytest selector and the train/test CLI reproductions (logs → `$HUB/cli/phase_g_dense_translation_fix_{split}.log`).</proposed_changes>
  <impacts>Until the batched reassembly guard works, the counted Phase G rerun and all downstream reports stay blocked, so STUDY-SYNTH cannot exit Tier 2 dwell.</impacts>
  <ledger_updates>Updated docs/fix_plan.md “Latest Attempt” entry, rewrote input.md, refreshed this implementation plan, and prepended the Turn Summary.</ledger_updates>
  <status>approved</status>
</plan_update>

## Goal & Success Criteria
- Restore `scripts/compare_models.py` so it can reconstruct the entire dense train/test splits (5 088 patches each) without the Translation shape mismatch.
- Deliver regression coverage (`pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split}`) proving the fix.
- Produce fresh CLI logs + metrics for both splits (`analysis/dose_1000/dense/{train,test}`) with zero `Translation` errors and updated `comparison_metrics.csv`.
- Remove or overwrite `analysis/blocker.log`; notify STUDY-SYNTH focus that the counted Phase G rerun can proceed.

## Constraints & Risks
- Core translation helpers live in `ptycho/custom_layers.py` and `ptycho/tf_helper.py`. Edits are authorized **only** for the batching/shape guard described here; preserve existing semantics for sparse view / smaller datasets.
- PyTorch (POLICY-001) remains mandatory even if the fix is TensorFlow-centric.
- CLI commands must continue to read/write under the existing hub; large artifacts stay in place to avoid losing provenance.
- Tests must stay within the runtime guidance from `docs/TESTING_GUIDE.md` (≤40 s for the new regression).

## Approach Overview
1. **Reproduce & log** the failure with the exact compare_models commands used by the pipeline (train + test splits, using the gs2 checkpoints + Phase F reconstructions).
2. **Batch/stream the translation path** inside `ReassemblePatchesLayer` (and helper functions) so `Translation` always sees matching `(batch, 2)` offset tensors. Prefer reusing `_reassemble_position_batched`/`mk_reassemble_position_batched_real` rather than reinventing chunking logic.
3. **Add regression coverage** that feeds a synthetic >5 k-patch tensor (or a reduced but shape-equivalent sample) through the updated layer to guard against future shape regressions.
4. **Validate & document** by rerunning the guarded pytest selectors plus the CLI compare_models commands for both splits, archiving logs under the hub, and recording the outcome in summary + blocker notes.

## Checklist

### Phase A — Reproduce & isolate
- [ ] Export `AUTHORITATIVE_CMDS_DOC` and `HUB` per Do Now.
- [ ] Run the train-split command:  
      `PYTHONPATH="$PWD" python scripts/compare_models.py --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 --test_data "$HUB"/data/phase_c/dose_1000/patched_train.npz --output_dir "$HUB"/analysis/dose_1000/dense/train --ms-ssim-sigma 1.0 --tike_recon_path "$HUB"/data/phase_f/dose_1000/dense/train/ptychi_reconstruction.npz --register-ptychi-only --split train |& tee "$HUB"/cli/phase_g_dense_translation_fix_train.log`
- [ ] Repeat for `--split test` (`--output_dir "$HUB"/analysis/dose_1000/dense/test"`, log to `..._test.log`).
- [ ] If either command still fails, capture the minimal stack + exit code in `$HUB/red/blocked_<timestamp>.md`.

### Phase B — Harden the translation path
- [ ] Update `ptycho/custom_layers.ReassemblePatchesLayer` to accept a configurable batch size and invoke `hh.mk_reassemble_position_batched_real` (falling back to the existing path for small patch counts).
- [ ] Where necessary, adjust helpers in `ptycho/tf_helper.py` (`mk_reassemble_position_real`, `_reassemble_position_batched`, `reassemble_patches`) so patch/offset tensors stay aligned when chunked; add shape assertions with actionable error messages.
- [ ] Document the new behavior (module docstring or short comment) so future contributors understand when batching engages.

### Phase C — Regression coverage & validation
- [ ] Ensure `tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split}` both pass (update the existing fixtures/test bodies only if required for the new batching behavior).
- [ ] Run `pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv | tee "$HUB"/green/pytest_compare_models_translation_fix.log`.
- [ ] Re-run the train + test `scripts/compare_models.py` commands; verify exit code 0 and refreshed metrics/plots under `analysis/dose_1000/dense/{train,test}`.
- [ ] Remove or overwrite `analysis/blocker.log`, update `analysis/verification_report.json` summary snippet, and notify STUDY-SYNTH focus that the counted rerun may resume.

## Do Now (hand-off to Ralph)
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier"`
2. Reproduce the failure for both splits using the commands listed in Phase A; keep the logs (`$HUB/cli/phase_g_dense_translation_fix_{train,test}.log`) and add a `$HUB/red/blocked_<timestamp>.md` note if the ValueError still appears.
3. Modify `ptycho/custom_layers.py` (and supporting helpers in `ptycho/tf_helper.py`) so `ReassemblePatchesLayer` streams patches in chunks (via `mk_reassemble_position_batched_real` / `_reassemble_position_batched`) whenever `patch_count > batch_size`, guaranteeing `Translation` sees matching shapes. Add shape assertions and brief docstrings explaining the batching behavior.
4. Keep the ≥5 k patch regression test (`tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split`) RED/ GREEN-focused—update fixtures or assertions only if the batching changes require it, then ensure it passes alongside `test_pinn_reconstruction_reassembles_batched_predictions`.
5. Run `pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv | tee "$HUB"/green/pytest_compare_models_translation_fix.log`.
6. Rerun `scripts/compare_models.py` for train and test splits (same args as Phase A, with updated logs). Success criteria: exit 0, refreshed `comparison_metrics.csv`, and no Translation errors; stash logs under `$HUB/cli/phase_g_dense_translation_fix_{train,test}.log`.
7. Update `analysis/blocker.log` / `{analysis}` summaries to reflect the fix and ping STUDY-SYNTH so the counted Phase G rerun can resume.

## Test Strategy
- Unit/functional: `pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv`
- CLI validation: the two `scripts/compare_models.py` commands listed above (train/test splits) with logs captured under `$HUB/cli/`.

Keep failure evidence under `$HUB/red/blocked_<timestamp>.md` (command, exit code, minimal stack) per TEST-CLI-001 whenever a step fails.
