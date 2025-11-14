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

<plan_update version="1.2">
  <trigger>The emergency reverts (`git revert da91e466` / `git revert 087a9238`) restored the pre-batching `_reassemble_position_batched` code so TF integration stays green. The plan now needs a reset so any future batching work reuses the existing helper instead of rewriting the default `tf_helper` path.</trigger>
  <focus_id>FIX-COMPARE-MODELS-TRANSLATION-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (REASSEMBLY-BATCH-001 / ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / DATA-001 / POLICY-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/DEVELOPER_GUIDE.md, docs/architecture.md, docs/workflows/pytorch.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/{implementation.md,summary.md}, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,summary.md}</documents_read>
  <current_plan_path>plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md</current_plan_path>
  <proposed_changes>Record the rollback, add explicit guardrails (“stable `tf_helper` components remain untouched; batching must hook into `_reassemble_position_batched`/`mk_reassemble_position_batched_real`”), and rewrite the Do Now so it requires overlap-count preservation, crop detection logs, and metric comparisons before resuming dense runs.</proposed_changes>
  <impacts>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 stays blocked until the guarded batching hook lands; any edits to `tf_helper` outside the authorized helper APIs are disallowed.</impacts>
  <ledger_updates>Updated docs/fix_plan.md, galph_memory.md, and this plan to capture the rollback + new guardrails; refreshed input.md.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.3">
  <trigger>`plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/reports/pytest_translation_fix.log` still fails at `ptycho/tf_helper.py:959` with `{{__wrapped__AddV2}} required broadcastable shapes`, and hub reality checks (`analysis/verification_report.json`, `cli/phase_g_dense_translation_fix_train.log`) confirm no new artifacts—`analysis` still lacks SSIM/metrics bundles and verification remains `n_valid=0`.</trigger>
  <focus_id>FIX-COMPARE-MODELS-TRANSLATION-001</focus_id>
  <documents_read>docs/index.md; docs/findings.md (REASSEMBLY-BATCH-001 / ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / DATA-001); docs/INITIATIVE_WORKFLOW_GUIDE.md; docs/DEVELOPER_GUIDE.md; docs/COMMANDS_REFERENCE.md; docs/TESTING_GUIDE.md; docs/development/TEST_SUITE_INDEX.md; specs/overlap_metrics.md; specs/data_contracts.md; docs/fix_plan.md; galph_memory.md; input.md; plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/{implementation.md,summary.md,reports/pytest_translation_fix.log}; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,summary.md}; hub paths (`analysis/verification_report.json`, `analysis/dose_1000/dense/train/logs/logs/debug.log`, `cli/phase_g_dense_translation_fix_train.log`).</documents_read>
  <current_plan_path>plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md</current_plan_path>
  <proposed_changes>Refine Phase B/Do Now so Ralph adds explicit shape/dtype assertions around `canvas + batch_result`, instruments the padded_size being used, replaces the per-element `tf.map_fn` resize with a single `tf.image.resize_with_crop_or_pad(batch_translated, padded_size, padded_size)` call, and ensures both `_reassemble_position_batched` branches derive padded_size from either the kwarg or `params.get_padded_size()`. Require capturing these diagnostics in the pytest log plus GREEN train/test CLI runs.</proposed_changes>
  <impacts>The counted dense rerun remains blocked until `_reassemble_position_batched` stops producing mismatched tensors; verification/reporting SLA stays unmet.</impacts>
  <ledger_updates>Updating docs/fix_plan.md, summary.md, and input.md this loop to highlight the instrumentation + padding-alignment tasks.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.4">
  <trigger>Today’s hub sweep reconfirmed `tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split` still crashes at `ptycho/tf_helper.py:959` (AddV2 broadcasting error), and the so-called “translation fix” CLI reproductions never actually ran because `scripts/compare_models.py` does not support the `--split` flag (`cli/phase_g_dense_translation_fix_train.log` aborts immediately with “unrecognized arguments: --split train”). No new artifacts exist under `analysis/` (`verification_report.json` remains 0/10, blocker log untouched).</trigger>
  <focus_id>FIX-COMPARE-MODELS-TRANSLATION-001</focus_id>
  <documents_read>docs/index.md; docs/prompt_sources_map.json (missing — noted); docs/findings.md (REASSEMBLY-BATCH-001 / ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / DATA-001); docs/INITIATIVE_WORKFLOW_GUIDE.md; docs/DEVELOPER_GUIDE.md; docs/COMMANDS_REFERENCE.md; docs/TESTING_GUIDE.md; docs/development/TEST_SUITE_INDEX.md; docs/GRIDSIZE_N_GROUPS_GUIDE.md; docs/architecture.md; specs/overlap_metrics.md; specs/data_contracts.md; specs/ptychodus_api_spec.md; docs/fix_plan.md; plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/{implementation.md,summary.md,reports/pytest_translation_fix.log}; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,summary.md}; hub artifacts (`analysis/verification_report.json`, `analysis/blocker.log`, `analysis/dose_1000/dense/train/comparison.log`, `cli/phase_g_dense_translation_fix_{train,test}.log`).</documents_read>
  <current_plan_path>plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md</current_plan_path>
  <proposed_changes>Fix the Phase A commands so they actually execute (drop the nonexistent `--split` flag and point the train/test runs at the patched NPZs), emphasize the need to log padded_size/batch diagnostics inside `_reassemble_position_batched`, and reiterate that Ralph must land the shape-assert + resize refactor before rerunning the CLI evidence.</proposed_changes>
  <impacts>Without a runnable Do Now, Ralph cannot perform the third-consecutive loop implementation work and STUDY-SYNTH stays blocked with dwell pressure; ensuring executable commands + instrumentation is critical before we risk another empty run.</impacts>
  <ledger_updates>Updating this plan, docs/fix_plan.md, summary.md, input.md, and galph_memory.md with the corrected commands + evidence expectations.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.5">
  <trigger>No new Ralph evidence has landed since the Nov 13 smoke attempts: `plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/reports/pytest_translation_fix.log` still fails at `ptycho/tf_helper.py:959`, `$HUB/cli/phase_g_dense_translation_fix_{train,test}.log` have not been refreshed, and `{analysis}/verification_report.json` remains `n_valid=0/10`. Code review shows `ReassemblePatchesLayer` still unconditionally wires `mk_reassemble_position_batched_real` without logging when batching engages, so we cannot tell from logs whether padded_size/crop guards ever trigger.</trigger>
  <focus_id>FIX-COMPARE-MODELS-TRANSLATION-001</focus_id>
  <documents_read>docs/index.md; docs/findings.md (REASSEMBLY-BATCH-001 / ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / DATA-001 / TYPE-PATH-001); docs/INITIATIVE_WORKFLOW_GUIDE.md; docs/COMMANDS_REFERENCE.md; docs/TESTING_GUIDE.md; docs/development/TEST_SUITE_INDEX.md; docs/DEVELOPER_GUIDE.md; docs/architecture.md; docs/GRIDSIZE_N_GROUPS_GUIDE.md; specs/overlap_metrics.md; specs/data_contracts.md; docs/fix_plan.md; plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/{implementation.md,summary.md,reports/pytest_translation_fix.log}; plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/{implementation.md,summary.md,reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json,analysis/blocker.log,cli/phase_g_dense_translation_fix_train.log}; input.md; galph_memory.md.</documents_read>
  <current_plan_path>plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md</current_plan_path>
  <proposed_changes>Tighten Phase B/Do Now so Ralph adds (a) a patch-count gate plus tf.print/log output inside `ReassemblePatchesLayer` showing when batching engages, and (b) structured diagnostics inside `_reassemble_position_batched` (padded_size, batch dims, crop counters, tf.debugging assertions) that surface in both pytest and CLI logs before re-running the selectors.</proposed_changes>
  <impacts>No instrumentation means we still have no proof that padded_size alignment or cropping is correct, so the counted dense rerun (and its ACCEPTANCE‑001 / TEST-CLI-001 / PREVIEW-PHASE-001 artifacts) stays blocked.</impacts>
  <ledger_updates>Updating docs/fix_plan.md, the initiative summary, and input.md with the reinforced batching/logging instructions.</ledger_updates>
  <status>approved</status>
</plan_update>

## Guardrails (Reset 2025-11-13)
- `ptycho/tf_helper.py` is treated as stable per CLAUDE.md; batching must leverage the existing helpers (`mk_reassemble_position_batched_real`, `_reassemble_position_batched`, `_flat_to_channel`) rather than rewriting default paths.
- No edits to `reassemble_patches`, `mk_reassemble_position_real`, or other shared helpers unless explicitly authorized here and backed by regression evidence.
- Every change must include overlap-count preservation (weight map or equivalent) and emit a warning/log whenever resizing would crop translated data.
- All guard tests (pytest selector + compare_models CLI train/test) must be GREEN before handing the hub back to STUDY-SYNTH.

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
- [ ] Run the train-split command (no `--split` flag — dataset path controls the split):  
      `PYTHONPATH="$PWD" python scripts/compare_models.py --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 --test_data "$HUB"/data/phase_c/dose_1000/patched_train.npz --output_dir "$HUB"/analysis/dose_1000/dense/train --ms-ssim-sigma 1.0 --tike_recon_path "$HUB"/data/phase_f/dose_1000/dense/train/ptychi_reconstruction.npz --register-ptychi-only |& tee "$HUB"/cli/phase_g_dense_translation_fix_train.log`
- [ ] Repeat for the held-out test split by swapping `--test_data "$HUB"/data/phase_c/dose_1000/patched_test.npz` and `--output_dir "$HUB"/analysis/dose_1000/dense/test"`; log to `..._test.log`.
- [ ] If either command still fails, capture the minimal stack + exit code in `$HUB/red/blocked_<timestamp>.md`.

### Phase B — Harden the translation path
- [ ] Update `ptycho/custom_layers.ReassemblePatchesLayer` to accept a configurable batch size and invoke `hh.mk_reassemble_position_batched_real` (falling back to the existing path for small patch counts).
- [ ] Where necessary, adjust helpers in `ptycho/tf_helper.py` (`mk_reassemble_position_real`, `_reassemble_position_batched`, `reassemble_patches`) so patch/offset tensors stay aligned when chunked; add shape assertions with actionable error messages.
- [ ] Document the new behavior (module docstring or short comment) so future contributors understand when batching engages.
- [ ] Instrument `_reassemble_position_batched` with structured diagnostics: emit tf.print/logs for `total_patches`, `batch_size`, `padded_size`, and `batch_translated.shape`, keep the single `tf.image.resize_with_crop_or_pad` call, increment/log a counter whenever padding trims translated data, and enforce the existing `tf.debugging.assert_equal(tf.shape(canvas), tf.shape(batch_result))`/dtype checks before accumulation.

### Phase C — Regression coverage & validation
- [ ] Ensure `tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split}` both pass (update the existing fixtures/test bodies only if required for the new batching behavior).
- [ ] Run `pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv | tee "$HUB"/green/pytest_compare_models_translation_fix.log`.
- [ ] Re-run the train + test `scripts/compare_models.py` commands; verify exit code 0 and refreshed metrics/plots under `analysis/dose_1000/dense/{train,test}`.
- [ ] Remove or overwrite `analysis/blocker.log`, update `analysis/verification_report.json` summary snippet, and notify STUDY-SYNTH focus that the counted rerun may resume.

## Do Now (Reset 2025-11-13)
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier"`.
2. Reproduce the failure with the train/test `scripts/compare_models.py` commands (Phase A — no `--split` flag; use the train/test NPZ paths) to refresh `$HUB/cli/phase_g_dense_translation_fix_{train,test}.log`. Capture blocker notes (`$HUB/red/blocked_<timestamp>.md`) if Translation still aborts.
3. Implement a guarded batching hook by wiring `ReassemblePatchesLayer` to call `hh.mk_reassemble_position_batched_real` (existing helper) only when `patch_count > batch_size`, and add tf.print/log statements that surface `patch_count`, `batch_size`, and `padded_size` whenever batching engages so the CLI/pytest logs prove which path executed. Do **not** modify `reassemble_patches` or `_flat_to_channel`; add inline comments plus an assertion that the helper sees `total_patches > batch_size` before it switches modes.
4. Enhance `_reassemble_position_batched` so it (a) derives `padded_size` from the kwarg or `params.get_padded_size()`, (b) keeps the single `tf.image.resize_with_crop_or_pad(batch_translated, padded_size, padded_size)` call, (c) emits tf.print/log lines showing batch dims + padded_size + a crop counter, and (d) raises an actionable `tf.debugging.assert_equal` error when `tf.shape(canvas)` and `tf.shape(batch_result)` (or dtypes) diverge. Any resize that would crop translated data must increment/log a counter so we can audit future runs in `$HUB/cli/...` logs.
5. Extend `tests/study/test_dose_overlap_comparison.py` with an intensity/overlap conservation check (compare streamed vs legacy results) and ensure both regression tests pass:  
   `pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv | tee "$HUB"/green/pytest_compare_models_translation_fix.log`
6. Rerun the train/test `scripts/compare_models.py` commands; success criteria: exit 0, refreshed `analysis/dose_1000/dense/{train,test}` artifacts, no crop warnings. Store logs under `$HUB/cli/phase_g_dense_translation_fix_{split}.log`.
7. Update `analysis/blocker.log`, `{analysis}/verification_report.json`, and the plan summary; hand control back to STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 once the guard tests and CLI commands are GREEN.

## Test Strategy
- Unit/functional: `pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv`
- CLI validation: the two `scripts/compare_models.py` commands listed above (train/test splits) with logs captured under `$HUB/cli/`.

Keep failure evidence under `$HUB/red/blocked_<timestamp>.md` (command, exit code, minimal stack) per TEST-CLI-001 whenever a step fails.
