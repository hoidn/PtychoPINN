# Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001

Initiative Header
- ID: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Title: Synthetic fly64 dose/overlap study
- Owner/Date: Ralph / 2025-11-05
- Status: in_progress (High priority)
- Working Plan: this file
- Reports Hub (Phase D): `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics/`
- Reports Hub (Phase E): `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/`
- Reports Hub (Phase G): `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`

> **Plan maintenance:** This is the single, evolving plan for the dose/overlap study. Update this file in place instead of creating new `plan/plan.md` documents. The active reports hub is `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/` for Phase E and `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/` for Phase G execution until a milestone changes—reuse them for logs/summaries unless a new milestone is declared.



<plan_update version="1.0">
  <trigger>Verification preflight (`analysis/verification_report.json`) still reports 0/10 required Phase G artifacts, proving no counted dense rerun happened after the scalar-safe mask + geometry-aware acceptance fixes landed.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Capture the verification shortfall so `{analysis}` expectations (SSIM grid, verification, highlights, metrics bundle, preview text, artifact inventory) stay visible even though the bugfix/tests are complete.
- Add a new “Next Do Now” block that elevates the counted dense rerun + `--post-verify-only` helper, GREEN pytest selectors, and metrics scripts with explicit env/logging guidance.
- Tie evidence requirements back to hub summaries and ledger updates so we know exactly what to publish once the run succeeds.</proposed_changes>
  <impacts>Running `run_phase_g_dense.py` plus verification consumes minutes and large disk space; missing any artifact keeps Phase G blocked and delays study metrics publication.</impacts>
  <ledger_updates>Switch docs/fix_plan.md’s Active Focus to this initiative, rewrite that Do Now, refresh the initiative + hub summaries, and update input.md so Ralph executes the rerun immediately.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>2025-11-14 hub sweep shows `{analysis}/verification_report.json` still 0/10, `analysis/comparison_manifest.json` filtering dose 100000, and `analysis/dose_1000/dense/train/comparison.log` stuck on the pre-fix Translation failure even though the 32-group smoke log and Phase D/E regeneration are GREEN.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/specs/spec-ptychopinn.md, docs/specs/spec-ptycho-core.md, docs/specs/spec-ptycho-runtime.md, docs/specs/spec-ptycho-workflow.md, docs/specs/spec-ptycho-interfaces.md, docs/specs/spec-ptycho-tracing.md, docs/DEVELOPER_GUIDE.md, docs/architecture.md, docs/workflows/pytorch.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, specs/ptychodus_api_spec.md, specs/overlap_metrics.md, prompts/callchain.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/comparison_manifest.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/comparison.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train_smoke/logs/logs/debug.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_stdout.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_g_dense_train.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_e_dense_gs2_dose1000.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Prepend the initiative + hub summaries with the verification/manifests blockers so Do Now stays focused on the counted rerun + verification bundle.
- Keep the Do Now in ready_for_implementation with the guarded pytest selectors, `run_phase_g_dense.py --clobber --dose 1000 --view dense --splits train test`, the fully parameterized `--post-verify-only`, and the metrics helpers, explicitly calling out the required artifacts and `$HUB/red/blocked_*.md` process.
- Restate references to ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001 so failures are captured as blockers rather than partial reruns.</proposed_changes>
  <impacts>Without a counted dose 1000 rerun there is still no SSIM/verification/highlights/metrics/preview/artifact-inventory bundle, so the study cannot move forward.</impacts>
  <ledger_updates>Update docs/fix_plan.md, the initiative summary, hub summary, input.md, and galph_memory.md with the refreshed evidence and Do Now.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>Hub inspection on 2025-11-14 confirms `{analysis}/verification_report.json` still shows 0/10 checks, `cli/run_phase_g_dense*.log` timestamps remain 2025-11-12, `cli/phase_d_dense.log` is empty, and `cli/phase_e_dense_gs2_dose1000.log` reports “No jobs match the specified filters” because dense NPZ files were never regenerated. The only Phase G comparison log (`analysis/dose_100000/dense/train/comparison.log`) continues to fail with `ValueError: Dimensions must be equal, but are 128 and 32`, proving no fresh counted run occurred after the acceptance-floor + scalar-mask fixes.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/comparison_manifest.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_100000/dense/train/comparison.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_stdout.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_e_dense_gs2_dose1000.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Preserve the existing Do Now nucleus but emphasize that the counted run must produce non-empty Phase D/Phase E CLI logs and fresh NPZ/weight artifacts before Phase F/G can proceed.
- Call out the stale comparison failure (ValueError in `Translation.call`) so debugging efforts capture the first failing selector/log if it reproduces.
- Reiterate ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001 guardrails inside the Do Now description and summaries.</proposed_changes>
  <impacts>Without a full rerun, `{analysis}` lacks SSIM/verification/highlights/metrics/preview artifacts, verification preflight blocks Phase G acceptance, and downstream comparison evidence cannot be trusted.</impacts>
  <ledger_updates>Update docs/fix_plan.md, initiative summary, hub summary, input.md, and galph_memory with the refreshed observations; keep Do Now ready_for_implementation so Ralph executes the pytest selectors + clobbered pipeline + metrics helpers.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>The limited compare_models smoke rerun (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_fix.log:1-40`) now dies with `ValueError: Unexpected baseline model output format` even though the prior “Layer "functional_10" expects 2 input(s)” bug is fixed, because the baseline inference bundle loaded via `load_inference_bundle` returns a single complex tensor rather than the `[amplitude, phase]` list that compare_models.py still assumes. Verification remains 0/10 (`analysis/verification_report.json:1-40`) and the translation blocker in `analysis/dose_1000/dense/train/comparison.log` is unchanged until the limited smoke can reach its save/metrics phase.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/DEVELOPER_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/overlap_metrics.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/comparison.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_fix.log, scripts/compare_models.py, tests/study/test_dose_overlap_comparison.py, ptycho/workflows/components.py</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Add a pre-step to the Do Now requiring a converter in `scripts/compare_models.py` that accepts either a two-output list or a single complex tensor from the baseline model and logs the derived amplitude/phase shapes while preserving the two-input call signature.
- Extend `tests/study/test_dose_overlap_comparison.py` with regression coverage for the complex-output path (mocking a complex tensor) and keep the existing input-signature test so future refactors cannot regress it.
- Require a refreshed limited smoke log (tee → `$HUB/cli/compare_models_dense_train_fix.log`) that proves the converter works before repeating the Phase D pytest guards, clobbered Phase G pipeline, post-verify helper, and metrics bundle regeneration per ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001.</proposed_changes>
  <impacts>Until the baseline output format is normalized, the limited smoke cannot proceed past inference, so the counted dense rerun and all verification artifacts remain blocked.</impacts>
  <ledger_updates>Update docs/fix_plan.md, initiative summary, hub summary, input.md, and galph_memory with the new failure signature and the converter/test requirements while keeping the focus ready_for_implementation.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>The follow-up limited smoke (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_fix.log:534-566`) now reaches the converter but crashes in `align_for_evaluation` with `ValueError: too many values to unpack (expected 2)` because `pinn_recon` remains a `(32, 128, 128)` stack that never gets reassembled into a single 2D image before cropping, keeping `{analysis}/verification_report.json:1-43` at `n_valid=0`.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/overlap_metrics.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_fix.log, scripts/compare_models.py, tests/study/test_dose_overlap_comparison.py, ptycho/image/cropping.py</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Prepend the Do Now with a checkpoint that reassembles/logs the PINN prediction stack via `reassemble_position` (using the study offsets) before calling `align_for_evaluation`, ensuring downstream code/tests only see 2D tensors.
- Add regression coverage in `tests/study/test_dose_overlap_comparison.py` for batched PINN predictions and require a fresh limited smoke log (tee → `$HUB/cli/compare_models_dense_train_fix.log`) proving the fix before touching the Phase D selectors or the counted Phase G rerun.
- Keep ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001 guardrails explicitly tied to the rerun so the SSIM/verification/highlights/metrics/preview/artifact-inventory bundle cannot be skipped.</proposed_changes>
  <impacts>Without reassembling the PINN output, compare_models never reaches the metrics/reporting stage, so Phase G remains blocked on missing evidence.</impacts>
  <ledger_updates>Refresh docs/fix_plan.md, initiative summary, hub summary, and input.md with the new pre-step, selectors, and log destinations.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>`analysis/dose_1000/dense/train/comparison.log:435-554` (captured in the latest `$HUB/analysis/dose_1000/dense/train/comparison.log`) shows the Phase G comparison still crashing inside `Translation.call` with `ValueError: Dimensions must be equal, but are 128 and 32`, and `analysis/blocker.log` records `run_phase_g_dense.py --clobber ... --view dense --splits train test` halting on that same command. `{analysis}/verification_report.json` therefore still reports 0/10 checks.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/DEVELOPER_GUIDE.md, docs/architecture.md, docs/workflows/pytorch.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, specs/overlap_metrics.md, docs/specs/spec-ptychopinn.md, docs/specs/spec-ptycho-core.md, docs/specs/spec-ptycho-runtime.md, docs/specs/spec-ptycho-interfaces.md, docs/specs/spec-ptycho-workflow.md, docs/specs/spec-ptycho-tracing.md, docs/fix_plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/comparison.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_stdout.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_stdout_now.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_e_dense_gs2_dose1000.log, input.md, galph_memory.md</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Before rerunning the pipeline, require a targeted fix in `scripts/compare_models.py` (right before `pinn_model.predict(...)`) that forces and logs `params.cfg['gridsize'] = sqrt(channel_count)` so grouped diffraction stacks cannot trigger the Translation B vs B·C mismatch, then run a limited `compare_models.py` smoke (tee to `$HUB/cli/compare_models_dense_train_fix.log`) to prove the ValueError is gone.
- Keep the guarded pytest selectors, clobbered rerun, post-verify helper, and metrics scripts in the Do Now, but add an explicit step to inspect `analysis/dose_1000/dense/train/comparison.log`/`cli/phase_g_dense_train.log` after the rerun and capture `$HUB/red/blocked_<timestamp>_comparison.md` if any Translation stack trace reappears.</proposed_changes>
  <impacts>Without eliminating the Translation crash, `{analysis}` will never contain the SSIM grid / verification / highlights / metrics artifacts demanded by ACCEPTANCE-001 and the verification report will stay at 0/10, blocking the study.</impacts>
  <ledger_updates>Update docs/fix_plan.md, initiative summary, hub summary, input.md, and galph_memory to call out the Translation blocker and the required pre-flight fix/smoke before retrying the counted pipeline.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>2025-11-14 audit shows `analysis/verification_report.json` still reports 0/10 checks, `cli/phase_d_dense.log` remains empty, `cli/phase_e_dense_gs2_dose1000.log` continues to end with “No jobs match the specified filters,” and the only comparison manifest/logs still reference the older dose=100000 run that fails inside `Translation.call`, so no dense Phase G evidence has been produced since 2025-11-12.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, docs/COMMANDS_REFERENCE.md, specs/data_contracts.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/comparison_manifest.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_100000/dense/train/comparison.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_e_dense_gs2_dose1000.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_d/dose_1000/dense, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/dense/gs2</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Record today’s audit so the plan and hub capture the 0/10 verification status, empty Phase D log, and “No jobs match the specified filters” exit in Phase E.
- Add a Do Now checkpoint requiring engineers to confirm the regenerated `phase_d_dense.log`, `phase_e_dense_gs2_dose1000.log`, and `data/phase_{d,e}` outputs exist (or file a `$HUB/red/blocked_*.md`) before running the metrics helpers.
- Keep the Do Now ready_for_implementation with the guarded pytest selectors, counted dense pipeline with `--clobber`, immediate `--post-verify-only` sweep, and Phase G metrics/preview/digest refresh tied to ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / DATA-001 / TYPE-PATH-001.</proposed_changes>
  <impacts>Without a fresh counted run the study cannot publish MS-SSIM/MAE deltas, preview text, or verification logs, so the Phase G acceptance gate stays blocked.</impacts>
  <ledger_updates>Update docs/fix_plan.md, initiative + hub summaries, galph_memory.md, and input.md with the refreshed audit plus new verification step so Ralph executes the rerun immediately.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>2025-11-14T0337Z spot-check confirms nothing has changed: `{analysis}/verification_report.json` still reports `n_valid=0`, the newest CLI artifacts remain from 2025-11-12, and `analysis/dose_1000/dense/train/comparison.log` now captures a `ValueError: Dimensions must be equal, but are 128 and 32` stack trace from `scripts.compare_models` (Translation → projective_warp_xla). `analysis/blocker.log` points at `cli/phase_g_dense_train.log`, so the counted comparison command is still exiting 1 and the SSIM/verification/highlights/metrics/preview/artifact-inventory bundle has never been written.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/COMMANDS_REFERENCE.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, specs/overlap_metrics.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/comparison.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/blocker.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/run_phase_g_dense_stdout.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_e_dense_gs2_dose1000.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/phase_e/dose_1000/dense/gs2/train.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Keep the Do Now unchanged (guard env/HUB → rerun guarded pytest selectors → run `run_phase_g_dense.py --clobber` + fully parameterized `--post-verify-only` → verify Phase D/E outputs exist → rerun metrics/reporting helpers) but explicitly require engineers to inspect `analysis/dose_1000/dense/train/comparison.log` after the rerun and capture a `$HUB/red/blocked_<timestamp>.md` quoting the Translation stack trace if it reoccurs.
- Document that success criteria remain: `{analysis}` must contain SSIM grid summary/log, verification logs, highlights, metrics summary/delta/digest, preview text, and `analysis/artifact_inventory.txt`, and `analysis/verification_report.json` must flip to 10/10 before summarizing results.
- Update the plan/ledger/summaries/input to reflect today’s audit so no one assumes the counted rerun already happened.</proposed_changes>
  <impacts>Lack of a successful Phase G rerun continues to block ACCEPTANCE-001 / PREVIEW-PHASE-001 / TEST-CLI-001 deliverables and keeps the study from shipping MS-SSIM/MAE deltas.</impacts>
  <ledger_updates>Refresh docs/fix_plan.md, initiative summary, hub summary, input.md, and galph_memory with this audit and reiterate the ready_for_implementation Do Now.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>Ralph’s limited smoke run (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_smoke_test/logs/logs/debug.log:360-402`) now reaches PINN inference without the Translation crash from 2025-11-12, but the command exits with `ValueError: Layer "functional_10" expects 2 input(s), but it received 1 input` because `scripts/compare_models.py` still calls `baseline_model.predict(...)` with only the flattened diffraction tensor and never forwards the companion offsets returned by `prepare_baseline_inference_data`. The newer GREEN guard (`cli/compare_models_dense_train_fix.log`) also stopped at argparse handling, so `{analysis}` still has 0/10 valid artifacts (`analysis/verification_report.json`) and the counted Phase G rerun has not started.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/overlap_metrics.md, specs/data_contracts.md, docs/fix_plan.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/comparison.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_fix.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_smoke_test/logs/logs/debug.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Document the new blocker (baseline expects two inputs) so engineers stop iterating on Translation and instead fix the compare_models baseline path before touching the counted pipeline.
- Update the Do Now so the first task is to wire `baseline_model.predict([baseline_input, baseline_offsets], …)` with explicit shape logging, assert that the diffraction channel count is a perfect square (raising early when it is not), and add targeted regression tests under `tests/study/test_dose_overlap_comparison.py` that cover grouped flattening + baseline invocation.
- Require a fresh limited smoke run (`--n-test-groups 32 --register-ptychi-only`) with logs under `$HUB/cli/compare_models_dense_train_fix.log` proving the ValueError is gone before rerunning the guarded pytest selectors and the counted Phase G pipeline.
- Keep the existing rerun/metrics steps (pytest guards → `run_phase_g_dense.py --clobber` → fully parameterized `--post-verify-only` → metrics/report helpers → verification of `{analysis}` contents) and reiterate the `$HUB/red/blocked_<timestamp>.md` expectation if any command still fails.</proposed_changes>
  <impacts>Without the baseline wiring fix, every comparison job exits before metrics are written, so `{analysis}` can never reach the SSIM/verification/highlights/metrics/preview/artifact-inventory bundle required by ACCEPTANCE-001 / TEST-CLI-001 / PREVIEW-PHASE-001.</impacts>
  <ledger_updates>Refresh docs/fix_plan.md, initiative summary, hub summary, input.md, and galph_memory so Ralph immediately lands the compare_models baseline fix + smoke evidence before rerunning the counted pipeline.</ledger_updates>
  <status>approved</status>
</plan_update>

## Problem Statement
We want to study PtychoPINN performance on synthetic datasets derived from the fly64 reconstructed object/probe across multiple photon doses, while manipulating inter-group overlap between solution regions. We will compare against a maximum-likelihood iterative baseline (pty-chi LSQML) using MS-SSIM (phase emphasis, amplitude reported) and related metrics. The study must prevent spatial leakage between train and test.

## Objectives
- Generate synthetic datasets from existing fly64 object/probe for multiple photon doses (e.g., 1e3, 1e4, 1e5).
- Sweep the *inputs we actually control*—image-level subsampling rate and the number of generated groups / solution regions—then record the derived overlap fraction per condition.
- Train PtychoPINN on each condition (gs=1 baseline, gs=2 grouped with K≥C for K-choose-C).
- Reconstruct with pty-chi LSQML (100 epochs to start; parameterizable).
- Run three-way comparisons (PINN, baseline, pty-chi) emphasizing phase MS-SSIM; also report amplitude MS-SSIM, MAE/MSE/PSNR/FRC.
- Record all artifacts under plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/ and update docs/fix_plan.md per loop.


## Architecture

Purpose
- Provide a clear picture of the dose–overlap study system: what components exist, the data they exchange, how configuration flows, and how users or code interact (CLI and Python APIs). This section avoids timeline nomenclature and focuses on stable structure.

Components
- Dataset Producer (`studies/fly64_dose_overlap/generation.py`)
  - Creates synthetic datasets by dose. Emits NPZ files and metadata suitable for downstream consumers.
- Overlap Metrics Engine (`studies/fly64_dose_overlap/overlap.py`)
  - Accepts dataset coordinates; performs deterministic subsampling (`s_img`) and grouping (`n_groups`); computes disc‐overlap metrics (group‑based, image‑based, group↔group COM) and writes per‑split metrics JSON plus a bundle.
- Learning Orchestrator (`studies/fly64_dose_overlap/training.py`)
  - Builds runnable jobs from dataset roots; prepares/bridges configuration; executes model training; persists manifests and logs.
- Baseline Reconstructor (`studies/fly64_dose_overlap/reconstruction.py` + `scripts/reconstruction/ptychi_reconstruct_tike.py`)
  - Enumerates and executes baseline reconstruction jobs; records outputs and execution telemetry.
- Comparison & Analysis Tools (`studies/fly64_dose_overlap/comparison.py`, bin helpers under `plans/active/.../bin/`)
  - Build and optionally execute comparison jobs, transform comparison outputs into summary metrics, generate aggregate reports/digests/highlights.
- Pipeline Runner (`plans/active/.../bin/run_phase_g_dense.py`)
  - A convenience entry that prepares output hubs, validates inputs, calls subordinate tools, and collects derived analysis artifacts. It exposes callable helpers for summary/validation in addition to its CLI.

Data Contracts & Types
- Dataset NPZ (DATA‑001; `docs/specs/spec-ptycho-interfaces.md`)
  - Arrays: diffraction stack; object/probe; coordinate vectors; index arrays where applicable.
  - Required metadata: `_metadata` with canonicalization markers and provenance fields.
- Overlap Metrics JSON (per split) + Bundle (see `docs/specs/overlap_metrics.md`)
  - Fields: `metrics_version`, `gridsize`, `s_img`, `n_groups`, `neighbor_count`, `probe_diameter_px`, `rng_seed_subsample`, averages for metrics (group‑based when applicable), and size counts (`n_images_total`, `n_images_subsampled`, `n_unique_images`, `n_groups_actual`).
- Training Manifest
  - Job configuration, selected inputs, produced artifact paths, integrity hashes (e.g., SHA256), and optional skip summary link.
- Reconstruction Manifest
  - Job list with CLI arguments, return codes, per‑job logs, and output locations.
- Comparison & Analysis Artifacts
  - `comparison_manifest.json` → `metrics_summary.json` → reports (`aggregate_report.md`) and delta artifacts (`metrics_delta_summary.json`, `metrics_delta_highlights.txt`, preview, digest, SSIM grid summary).

Configuration Boundaries
- Study Controls
  - Explicit inputs: `gridsize ∈ {1,2}`, `s_img ∈ (0,1]`, `n_groups ≥ 1`, `neighbor_count` (default 6), `probe_diameter_px` (record actual), deterministic seeds (subsampling, grouping, simulation).
  - Dense/sparse labels are not control inputs here; measured overlaps are reported as metrics.
- Training Bridge
  - Modern configuration objects drive training; legacy modules (where used) are parameterized via a controlled bridge (safe update of the legacy store prior to legacy imports). This keeps learning code explicit while preserving required interop.

Data Flow (Conceptual)
- NPZ datasets (by dose, split) → Overlap metrics engine computes and records sampling parameters and measured overlaps → Learning orchestrator trains models from datasets (and optional selections) → Baseline reconstructor produces reference reconstructions → Comparison and analysis tools summarize metrics and generate reports/digests/highlights.

Interfaces (CLI ↔ Python API)
- Overlap Metrics
  - CLI: `python -m studies.fly64_dose_overlap.overlap --phase-c-root <dir> --output-root <dir> --artifact-root <dir> --gridsize <1|2> --s-img <0..1] --n-groups <int> [--neighbor-count <int> --probe-diameter-px <float> --rng-seed-subsample <int>]`
  - API: `compute_overlap_metrics(coords, gridsize, s_img, n_groups, neighbor_count=6, probe_diameter_px=None, rng_seed_subsample=None) -> OverlapMetrics`; `generate_overlap_views(train_path, test_path, output_dir, ...) -> dict`.
- Learning
  - CLI: `python -m studies.fly64_dose_overlap.training --phase-c-root <dir> --phase-d-root <dir> --artifact-root <dir> --dose <int> [--dry-run]`
  - API: `build_training_jobs(...) -> list`; `run_training_job(job, dry_run=False) -> result`.
- Baseline Reconstruction
  - CLI: `python -m studies.fly64_dose_overlap.reconstruction --phase-c-root <dir> --phase-d-root <dir> --artifact-root <dir> --dose <int> --view <dense|sparse> --split <train|test> [--dry-run]`
  - API: `build_ptychi_jobs(...) -> list`; `run_ptychi_job(job, dry_run=False) -> result`.
- Comparison & Analysis
  - CLI: `python -m studies.fly64_dose_overlap.comparison [--dry-run|--execute] ...`; helpers `report_phase_g_dense_metrics.py`, `analyze_dense_metrics.py` are primarily CLIs (with internal functions tested via unit tests).
  - API: `build_comparison_jobs(...) -> list`; `execute_comparison_jobs(jobs, artifact_root) -> manifest`. Runner exposes callable helpers like `summarize_phase_g_outputs()`.

Relationships & Invariants
- Consumers and producers are loosely coupled through filesystem contracts (NPZ datasets; JSON manifests; CSV/Markdown summaries).
- Measured overlap replaces spacing/packing gates. Inputs control sampling (`s_img`, `n_groups`); outputs report observed overlaps.
  - Train/test separation and provenance are preserved across the pipeline; logs and manifests bind CLI invocations to artifacts.

Artifacts & Logging
- Artifact stores organize:
  - CLI transcripts under `cli/`.
  - Structured outputs and summaries under `analysis/`.
  - Optional archives of prior outputs under `archive/` when clobbering is requested by runners.


### Interfaces Policy (Library‑First, CLI‑Thin)

- Principle
  - Core logic is exposed as importable functions returning structured results. CLIs are thin wrappers for argument parsing, file I/O, and provenance (stdout/logs/exit codes).

- Preferred usage by component
  - Overlap metrics: use Python API (`compute_overlap_metrics`, `generate_overlap_views`) in code; keep CLI for ad‑hoc/manual use.
  - Learning orchestration: use Python API (`build_training_jobs`, `run_training_job`); keep CLI for one‑offs and transcripts.
  - Baseline reconstruction: keep an external tool boundary for the baseline engine; wrap invocation in a Python helper that shells out (structured args/telemetry) to preserve isolation and logs.
  - Comparison & analysis: use Python API (`build_comparison_jobs`, `execute_comparison_jobs`) for orchestration; CLIs remain for dry‑runs and manual invocation.
  - Pipeline runner: prefer callable helpers (extract to a small importable module if needed); the CLI remains a thin wrapper.
  - Reporting helpers (aggregate report, digest): expose pure functions that produce report/digest content; CLIs only handle path parsing and IO.
  - Verifiers and aux tools (`verify_dense_pipeline_artifacts.py`, `ssim_grid.py`, `check_dense_highlights_match.py`): provide function entry points (e.g., `verify(hub: Path) -> Result`) with CLI shims mapping exceptions to exit codes.

- Error handling & return types
  - Functions raise exceptions and return dataclasses/dicts with stable fields; CLIs translate exceptions to user‑friendly messages and numeric exit codes.

- Explicit subprocess boundaries
  - Maintain subprocess calls for external tools such as `scripts/reconstruction/ptychi_reconstruct_tike.py` to preserve environment isolation and reproducible logs.


## Deliverables
1. Dose-swept synthetic datasets with spatially separated train/test splits.
2. Group-level overlap tooling that logs `image_subsampling`, `n_groups_requested`, and the *measured* overlap fraction for every run (historical “dense/sparse” labels become reporting shorthand only).
3. Trained PINN models per condition (gs1 and gs2 variants) with fixed seeds.
4. pty-chi LSQML reconstructions per condition (100 epochs baseline; tunable).
5. Comparison outputs (plots, aligned NPZs, CSVs) with MS-SSIM (phase, amplitude) and summary tables.
6. Study summary.md aggregating findings per dose/view.

## Exit Criteria
1. Phase D overlap metrics implemented (API + CLI) per `specs/overlap_metrics.md`; no spacing/packing gates; dense/sparse labels deprecated.
2. Phase D tests GREEN (disc overlap unit tests; Metric 1/2/3; gs=1 skips Metric 1; integration bundle fields recorded).
3. Phase E TensorFlow training restored; at least one gs1 and one gs2 real run; manifests record `bundle_path` + SHA256; logs archived in the Phase E hub.
4. Phase G evidence present: `ssim_grid_summary.md`, `verification_report.json`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_digest.md`, `artifact_inventory.txt`; hub summaries updated with MS‑SSIM/MAE deltas.
5. Test registry synchronized: update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md`; save `--collect-only` logs under the active Reports Hub.

## Backend Selection (Policy for this Study)
- PINN training/inference: **TensorFlow only.** This initiative depends on the legacy `ptycho_train` stack because it is the fully tested backend per CLAUDE.md.
- Iterative baseline (pty-chi): PyTorch under the hood is acceptable for Phase F scripts, but keep it isolated from the PINN pipeline.
- PyTorch parity belongs to future work. Log the remaining work as TODO items rather than mixing stacks mid-initiative.

## Phases Overview
- Phase A — Design & Constraints: Study design constants, seeds, split policy.
- Phase B — Test Infrastructure: Validator + test scaffolding for DATA‑001 and split integrity.
- Phase C — Dataset Generation: Dose‑swept synthetic datasets; y‑axis split; manifests.
- Phase D — Overlap Metrics: Overlap‑driven sampling; Metric 1/2/3 reporting (no spacing gates).
- Phase E — Training (TF): gs1 baseline + gs2 runs; SHA256 manifests.
- Phase F — Baseline (pty‑chi): LSQML (100 epochs) with artifact capture.
- Phase G — Comparison & Analysis: SSIM grid; verification; highlights; metrics bundle.

## Integration Guarantees (Spec‑Forward)
- Controls → Training Data Flow:
  - `s_img` and `rng_seed_subsample` SHALL map to data loading as `n_subsample = ⌊s_img · N⌋` and `subsample_seed = rng_seed_subsample` via `ptycho.workflows.components.load_data()`.
  - `n_groups` SHALL be the target count for solution regions; for `gridsize=1` this equals the number of positions; for `gridsize=2` grouping uses K‑NN with allowed duplication per `docs/specs/overlap_metrics.md`.
  - `neighbor_count` SHALL be threaded into validation/training to keep K consistent with Phase D metrics (K ≥ C for gs2).
- Evidence:
  - Per‑split metrics JSON and `metrics_bundle.json` are archived alongside Phase D NPZs.
  - Training manifests/logs MUST echo `s_img`, `n_groups`, `neighbor_count`, and derived `n_subsample`.

## Phases

### Phase A — Design & Constraints (COMPLETE)
**Status:** Complete — constants encoded in `studies/fly64_dose_overlap/design.py::get_study_design()`

Checklist
- [x] Dose list fixed (1e3, 1e4, 1e5)
- [x] Seeds set (sim=42, grouping=123, subsampling=456)
- [x] y‑axis train/test split policy selected and documented

**Dose sweep:**
- Dose list: [1e3, 1e4, 1e5] photons per exposure

**Control knobs (primary inputs):**
- Image subsampling rates `s_img ∈ {1.0, 0.8, 0.6}` — fraction of diffraction frames retained per dose.
- Requested group counts `n_groups ∈ {512, 768, 1024}` (per `docs/GRIDSIZE_N_GROUPS_GUIDE.md`) — scales the number of solution regions per field of view while keeping intra-group K-NN neighborhoods tight (K=7 ≥ C=4 for gs2).

**Measured overlap metrics:**
- For every `(dose, s_img, n_groups)` combination we emit the Metric 1/2/3 disc-overlap averages defined in `specs/overlap_metrics.md` (plus `probe_diameter_px`, `neighbor_count`, RNG seeds). These values are persisted per split (train/test) and bundled in `metrics_bundle.json`.
- Labels such as “dense” or “sparse” are legacy shorthands only; artifacts, manifests, tests, and analysis must rely on the recorded knobs (`s_img`, `n_groups`) plus the measured overlap metrics.

**Patch geometry:**
- Patch size N=128 pixels (nominal from fly64 reconstructions)

**Train/test split:**
- Axis: 'y' (top vs bottom halves)
- Ensures spatial separation to prevent leakage

**RNG seeds (reproducibility):**
- Simulation: 42
- Grouping: 123
- Subsampling: 456

**Metrics configuration:**
- MS-SSIM sigma: 1.0
- Emphasize phase: True
- Report amplitude: True
- FRC threshold: 0.5

**Test coverage:**
- `tests/study/test_dose_overlap_design.py::test_study_design_constants` (PASSED)
- `tests/study/test_dose_overlap_design.py::test_study_design_validation` (PASSED)
- `tests/study/test_dose_overlap_design.py::test_study_design_to_dict` (PASSED)

### Phase B — Test Infrastructure Design (COMPLETE)
**Status:** COMPLETE — validator + 11 tests PASSED with RED/GREEN evidence

Checklist
- [x] Validator enforcing DATA‑001 and split integrity
- [x] 11 tests GREEN with red/green/collect artifacts
- [x] Documentation updated with validator scope and findings references
- Working Plan: `reports/2025-11-04T025541Z/phase_b_test_infra/plan.md`
- Deliverables:
- `studies/fly64_dose_overlap/validation.py::validate_dataset_contract` enforcing DATA-001 keys/dtypes, amplitude requirement, geometry metadata (`geometry_acceptance_bound`, `effective_min_acceptance`) for observability, and y-axis split integrity. ✅
  - Validator now asserts each manifest includes `image_subsampling`, requested `n_groups`, and the derived `overlap_fraction`, ensuring downstream code/tests never assume hard-coded view labels. ✅
  - pytest coverage in `tests/study/test_dose_overlap_dataset_contract.py` (11 tests, all PASSED) with logged red/green runs. ✅
  - Updated documentation (`implementation.md`, `test_strategy.md`, `summary.md`) recording validator scope and findings references (CONFIG-001, DATA-001, OVERSAMPLING-001). ✅
- Artifact Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T025541Z/phase_b_test_infra/`
- Test Summary: 11/11 PASSED, 0 FAILED, 0 SKIPPED
- Execution Proof: red/pytest.log (1 FAILED stub), green/pytest.log (11 PASSED), collect/pytest_collect.log (11 collected)

### Phase C — Dataset Generation (Dose Sweep) (COMPLETE)
**Status:** Complete — orchestration pipeline with 5 tests PASSED (RED→GREEN evidence captured)

Checklist
- [x] Orchestration implemented (simulate → canonicalize → patch → split → validate)
- [x] CLI entry for dose sweep
- [x] 5 tests GREEN; artifacts recorded
- [x] Manifest written and paths stable across reruns

**Deliverables:**
- `studies/fly64_dose_overlap/generation.py` orchestrates: simulate → canonicalize → patch → split → validate for each dose ✅
- CLI entry point: `python -m studies.fly64_dose_overlap.generation --base-npz <path> --output-root <path>` ✅
- pytest coverage in `tests/study/test_dose_overlap_generation.py` (5 tests, all PASSED with monkeypatched dependencies) ✅
- Updated documentation and test artifacts ✅

**Pipeline Workflow:**
1. `build_simulation_plan(dose, base_npz_path, design_params)` → constructs dose-specific `TrainingConfig`/`ModelConfig` with n_images from base dataset
2. `generate_dataset_for_dose(...)` → orchestrates 5 stages:
   - Stage 1: Simulate diffraction with `simulate_and_save()` (CONFIG-001 bridge handled internally)
   - Stage 2: Canonicalize with `transpose_rename_convert()` (DATA-001 NHW layout enforced)
   - Stage 3: Generate Y patches with `generate_patches()` (K=7 neighbors per design)
   - Stage 4: Split train/test on y-axis with `split_dataset()` (spatial separation)
   - Stage 5: Validate both splits with `validate_dataset_contract()` (Phase B validator)
3. CLI entry iterates over `StudyDesign.dose_list`, captures logs, writes `run_manifest.json`

**Artifact Hub:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/`
**Test Summary:** 5/5 PASSED, 0 FAILED, 0 SKIPPED
**Execution Proof:**
- red/pytest.log (3 FAILED with TypeError on TrainingConfig.gridsize)
- green/pytest.log (5 PASSED after fixing ModelConfig.gridsize separation)
- collect/pytest_collect.log (5 collected)

**Findings Applied:**
- CONFIG-001: `simulate_and_save()` handles `update_legacy_dict(p.cfg, config)` internally
- DATA-001: Canonical NHW layout enforced by `transpose_rename_convert_tool`
- OVERSAMPLING-001: K=7 neighbor_count preserved in patch generation for Phase D



### Phase D — Overlap Metrics (CLI evidence archived)
**Status:** Implementation, tests, and the gs1/gs2 CLI evidence bundle are complete (commit `d94f24f7` + `$HUB/summary/summary.md` + `analysis/artifact_inventory.txt`). This phase now feeds Phase E manifests and Phase G reruns; no open tasks remain here.

Deprecation Note
- Dense/Sparse labels and geometry/spacing acceptance gates remain deprecated. Every run must rely on explicit `s_img`/`n_groups` controls and record measured overlaps per `specs/overlap_metrics.md`.

Checklist
- [x] Implement API: compute Metric 1 (gs=2), Metric 2, Metric 3; disc overlap with parameterized `probe_diameter_px`.
- [x] Add CLI controls: `--gridsize`, `--s-img`, `--n-groups`, `--neighbor-count`, `--probe-diameter-px`.
- [x] Update tests: disc overlap unit tests; Metric 1/2/3; gs=1 skips Metric 1; integration bundle fields.
- [x] Record per-split metrics JSON and aggregated bundle under the Phase D hub (gs1 `s_img=1.0,n_groups=512` and gs2 `s_img=0.8,n_groups=512` using `data/phase_c/dose_1000`). Evidence: `metrics/gs1_s100_n512/*.json`, `metrics/gs2_s080_n512/*.json`, `cli/phase_d_overlap_gs{1,2}.log`, `analysis/artifact_inventory.txt`.

**Reality (2025-11-13):**
- `studies/fly64_dose_overlap/overlap.py` + `tests/study/test_dose_overlap_overlap.py` remain as above (commit `d94f24f7`, `$HUB/green/pytest_phase_d_overlap.log`).
- Both CLI profiles completed on 2025-11-11 (see `$HUB/summary/summary.md`). Metrics highlight:
  - **gs1 s_img=1.0, n_groups=512:** Metric 2 train/test ≈0.891, Metric 3 ≈0.266; Metric 1 N/A by design.
  - **gs2 s_img=0.8, n_groups=512:** Metric 1 train/test ≈0.892, Metric 2 ≈0.878, Metric 3 ≈0.269.
- `analysis/artifact_inventory.txt` and `summary/summary.md` list the commands, JSON paths, and Metric 1/2/3 deltas. This satisfies the guardrail blocking Phase E/G.

**Hand-off:**
- Reference artifacts from `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics/`.
- Use `metrics/gs{1,2}_*/metrics_bundle.json` plus `analysis/artifact_inventory.txt` when Phase E manifests or Phase G comparisons need overlap statistics.
- All new blocking work now lives under Phase G (dense rerun + verification bundle); see the section below for the active Do Now.

**Findings Applied:** CONFIG-001 (pure NPZ loading), DATA-001 (canonical layout), OVERSAMPLING-001 (K≥C preserved for gs=2), ACCEPTANCE-001 (geometry-aware floors recorded and logged). Dense/sparse labels remain deprecated per `docs/GRIDSIZE_N_GROUPS_GUIDE.md`.

### Phase E — Train PtychoPINN (PAUSED — awaiting TensorFlow rework)
**Status:** Paused. We must restore the TensorFlow training pipeline before any further runs; PyTorch work is retained below as historical context but no longer authoritative for this initiative.

Checklist
- [ ] E0 — TensorFlow pipeline restoration (delegate to ptycho_train; CONFIG‑001 ordering; add tests)
- [ ] E0.5 — Metadata alignment (Phase D parity: `s_img`, `n_groups`, overlap metrics)
- [ ] E0.6 — Evidence rerun (gs2 dense + gs1 baseline) with SHA256 proofs under Phase E hub

**Immediate Deliverables (blocking before resuming evidence):**
- **E0 — TensorFlow pipeline restoration:** Update `studies/fly64_dose_overlap/training.py` plus `run_phase_e_job.py` so they delegate to the TensorFlow `ptycho_train` workflows, honoring CONFIG-001 ordering. Ship pytest coverage for the TF path (CLI filters, manifest writing, skip summary) and capture a deterministic CLI command under the existing Phase E hub.
- **E0.5 — Metadata alignment:** Ensure manifests and `training_manifest.json` mirror the Phase D metadata fields (`image_subsampling`, `n_groups_requested`, `overlap_fraction`) so Phase G comparisons can correlate overlap statistics with training runs.
- **E0.6 — Evidence rerun:** Re-run the counted dense gs2 + baseline gs1 TensorFlow jobs with SHA256 proofs stored under `plans/active/.../phase_e_training_bundle_real_runs_exec/`, updating docs/fix_plan.md and the hub summary.

**Deferred — PyTorch parity context (informational only):**
- Prior attempts E1–E5.5 wired PyTorch runners, MemmapDatasetBridge hooks, and skip-reporting CLI outputs (see artifact hubs below). These serve as references for future parity work but must not dictate the current backend choice.

**Artifact Hubs (historical evidence, still referenced for context):**
- E1 Job Builder: `reports/2025-11-04T060200Z/phase_e_training_e1/`
- E3 Run Helper: `reports/2025-11-04T070000Z/phase_e_training_e3/`, `reports/2025-11-04T080000Z/phase_e_training_e3_cli/`
- E4 CLI Integration: `reports/2025-11-04T090000Z/phase_e_training_e4/`
- E5 MemmapDatasetBridge: `reports/2025-11-04T133500Z/phase_e_training_e5/`, `reports/2025-11-04T150500Z/phase_e_training_e5_path_fix/`
- E5.5 Skip Summary: `reports/2025-11-04T161500Z/phase_e_training_e5_real_run/`, `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/`
- E5 Documentation Sync: `reports/2025-11-04T084850Z/phase_e_training_e5_doc_sync/`

**Key Constraints & References:**
- CONFIG-001 compliance: `update_legacy_dict(params.cfg, config)` must run before any TensorFlow data loading or model construction.
- DATA-001 compliance: Phase D NPZ layout enforced via `build_training_jobs` path construction; canonical contract assumed.
- OVERSAMPLING-001: neighbor_count=7 satisfies K≥C=4 for gridsize=2 throughout.
- BACKEND POLICY: TensorFlow is the only supported PINN backend for this initiative; PyTorch tasks are explicitly deferred.

**Historical Test Coverage (PyTorch context):**
- `test_build_training_jobs_matrix`
- `test_run_training_job_invokes_runner`
- `test_run_training_job_dry_run`
- `test_training_cli_filters_jobs`
- `test_training_cli_manifest_and_bridging`
- `test_execute_training_job_delegates_to_pytorch_trainer`
- `test_training_cli_invokes_real_runner`
- `test_build_training_jobs_skips_missing_view`

These tests must be revisited/rewritten for the TensorFlow rework before Phase E can be marked complete again.

**Future deterministic CLI (to be produced):**
Document the TensorFlow command once E0 ships; for now this slot remains `TBD (TensorFlow training command)` and blocks the Do Now nucleus.

#### Execution Guardrails (2025-11-12)
- Reuse `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/` for all dense gs2 and baseline gs1 real-run artifacts until both bundles + SHA256 proofs exist. Do **not** mint new timestamped hubs for this evidence gap.
- Before proposing new manifest/test tweaks, promote the Phase C/D regeneration + training CLI steps into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_e_job.py` (T2 script) and reference it from future How-To Maps. Prep-only loops are not permitted once the script exists.
- Each new Attempt touching Phase E/G must deliver at least one of:
  * A successful dense gs2 or baseline gs1 training CLI run (stdout must include `bundle_path` + `bundle_sha256`) with artifacts stored under the hub above.
  * A `python -m studies.fly64_dose_overlap.comparison --dry-run=false ...` execution whose manifest captures SSIM/MS-SSIM metrics plus `n_success/n_failed`.
- Training loss guardrail: once a run is visually validated, copy its manifest to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/reference/training_manifest.json` and treat it as the “golden” baseline. Every new run must execute `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_training_loss.py --reference plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/reference/training_manifest.json --candidate <current_manifest> --dose <value> --view <value> --gridsize <value>` (with optional `--tolerance` override) and archive the checker output next to the manifest. The guardrail fails if `final_loss` is missing, non-finite, or exceeds the reference loss by more than the configured tolerance.

### Phase F — pty-chi LSQML Baseline

Checklist
- [ ] Run LSQML 100 epochs; capture logs/metrics; archive under baseline hub paths; document environment/version
- Run scripts/reconstruction/ptychi_reconstruct_tike.py with algorithm='LSQML', num_epochs=100 per test set; capture outputs.

### Phase G — Comparison & Analysis

Deprecation Note
- Any legacy references to “dense/sparse” and spacing acceptance in this section are historical; the authority for overlap is the measured metrics per `specs/overlap_metrics.md` with explicit `s_img`/`n_groups`.

Checklist
- [ ] Produce SSIM grid summary/log
- [ ] Produce verification report/log
- [ ] Run highlights checker and capture log
- [ ] Write metrics summary and digest; artifact inventory present
- [ ] Update hub summaries and docs/fix_plan with MS‑SSIM/MAE deltas
- Use scripts/compare_models.py three-way comparisons with --ms-ssim-sigma 1.0 and registration; produce plots, CSVs, aligned NPZs; write per-condition summaries.
- After `summarize_phase_g_outputs` completes, run `plans/active/.../bin/report_phase_g_dense_metrics.py --metrics <hub>/analysis/metrics_summary.json` (optionally with `--ms-ssim-threshold 0.80`). Treat the generated **MS-SSIM Sanity Check** table as the go/no-go gate: any row flagged `LOW (...)` indicates reconstruction quality is suspect and the Attempt must be marked blocked until re-run.
- `plans/active/.../bin/analyze_dense_metrics.py` now also embeds the same sanity table inside `metrics_digest.md`; archive this digest under the hub’s `analysis/` directory so reviewers can see absolute MS-SSIM values without hunting through CSVs.

<plan_update version="1.0">
  <trigger>Limited smoke on 2025-11-13 (32-group test) now proves the PINN reassembly/logging path is fixed (`cli/compare_models_dense_train_fix.log:502-656`), Phase D already regenerated the dose 1000 dense NPZ + metrics bundle (`cli/phase_d_dense.log:12-36`), and Phase E produced a fresh gs2 `wts.h5.zip` (`data/phase_e/dose_1000/dense/gs2/train.log:1-27`), yet `{analysis}/verification_report.json` still reports `n_valid=0` and `phase_g_dense_train.log:1-21` shows the counted run filtered dose=100000, so no Phase G evidence exists for the real 1000-dose rerun.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/specs/spec-ptychopinn.md, docs/specs/spec-ptycho-core.md, docs/specs/spec-ptycho-runtime.md, docs/specs/spec-ptycho-workflow.md, docs/specs/spec-ptycho-interfaces.md, docs/specs/spec-ptycho-tracing.md, specs/overlap_metrics.md, specs/data_contracts.md, specs/ptychodus_api_spec.md, docs/architecture.md, docs/DEVELOPER_GUIDE.md, prompts/callchain.md, docs/fix_plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary/summary.md, galph_memory.md, input.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/verification_report.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/comparison_manifest.json, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/compare_models_dense_train_fix.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train/logs/logs/debug.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_d_dense.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_e_dense_gs2_dose1000.log, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/phase_g_dense_train.log</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md</current_plan_path>
  <proposed_changes>- Record that the limited smoke + Phase D/E regeneration checkpoints are complete (with the cited logs) so the active Do Now can focus solely on steps 4–10: rerun the guarded pytest selectors, execute the counted `run_phase_g_dense.py --clobber` for dose 1000, immediately run the fully parameterized `--post-verify-only`, then regenerate the metrics/preview/digest + verification bundle.
- Highlight that `{analysis}/verification_report.json` stays `n_valid=0` and `analysis/comparison_manifest.json` still filters dose=100000 so Ralph knows which artifacts/logs must change before we can claim success.
- Reiterate the requirement to write `$HUB/red/blocked_<timestamp>.md` with command + error signatures if any rerun fails, and to capture MS-SSIM/MAE deltas + command transcripts in the hub summary once verification hits 10/10.</proposed_changes>
  <impacts>Each counted rerun consumes minutes and re-generates large NPZ/model bundles; without a precise handoff Ralph could easily rerun the wrong dose/view again, leaving Phase G evidence missing.</impacts>
  <ledger_updates>Update docs/fix_plan.md (Active Focus + Latest Attempt), the initiative summary, hub summary, and input.md with the limited-smoke evidence and the focused rerun steps; keep Do Now ready_for_implementation.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Phase G — Active Checklist (2025-11-12)
- [x] Wire post-verify automation (`run_phase_g_dense.py::main`) so every dense run automatically executes SSIM grid → verifier → highlights checker with success banner references (commit 74a97db5).
- [x] Add pytest coverage for collect-only + execution chain (`tests/study/test_phase_g_dense_orchestrator.py::{test_run_phase_g_dense_collect_only_post_verify_only,test_run_phase_g_dense_post_verify_only_executes_chain}`) and archive GREEN logs under `reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/`.
- [x] Normalize success-banner path prints to hub-relative strings (`run_phase_g_dense.py::main`, both full run and `--post-verify-only`) and extend `test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` to assert the relative `cli/` + `analysis/` lines (TYPE-PATH-001). — commit `7dcb2297`.
- [x] Extend `test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview` so the **full** execution path asserts hub-relative `CLI logs: cli`, `Analysis outputs: analysis`, and `analysis/artifact_inventory.txt` strings (TYPE-PATH-001) to guard the counted run prior to execution evidence.
- [x] Deduplicate the success banner's "Metrics digest" lines in `plans/active/.../bin/run_phase_g_dense.py::main` so only one Markdown path prints and the CLI log line stays distinct, avoiding conflicting guidance before we archive evidence.
- [x] Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` so it asserts **exactly one** `Metrics digest:` line appears in stdout (guarding against future banner duplication) and continue to check for the `Metrics digest log:` reference. — commit `4cff9e38`.
- [x] Add a follow-on assertion in `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` to ensure `stdout.count("Metrics digest log: ") == 1` so CLI log references cannot duplicate when future banner edits land (TYPE-PATH-001, TEST-CLI-001). — commit `32b20a94`.
- [x] Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest` so it also asserts the success banner references `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, and the SSIM grid summary/log paths. This keeps verification evidence lines guarded alongside the digest banner (TEST-CLI-001, TYPE-PATH-001). — commit `6a51d47a`.
- [x] Extend `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` so the `--post-verify-only` path also asserts the SSIM grid summary/log plus verification report/log/highlights lines and ensures stubbed CLI logs exist (TEST-CLI-001, TYPE-PATH-001). — commit `ba93f39a`.
- [x] Replace the placeholder `geometry_aware_floor` logic in `studies/fly64_dose_overlap/overlap.py::generate_overlap_views` with a per-split bounding-box acceptance bound: compute the theoretical maximum acceptance as `(area / (pi * (threshold / 2) ** 2)) / n_positions`, clamp the bound to ≤0.10, guard against zero with a tiny epsilon, and persist both `geometry_acceptance_bound` and the resulting `effective_min_acceptance` through `SpacingMetrics`, `metrics_bundle.json`, and `_metadata`. (Implemented in `studies/fly64_dose_overlap/overlap.py:334-555`; verified locally + via `$HUB/green/pytest_dense_acceptance_floor.log`.) (docs/findings.md: STUDY-001, ACCEPTANCE-001; specs/data_contracts.md §12.)
- [x] Extend `tests/study/test_dose_overlap_overlap.py` with `test_generate_overlap_views_dense_acceptance_floor` to pin the low-acceptance scenario and verify the metrics bundle records `geometry_acceptance_bound`/`effective_min_acceptance`. (`tests/study/test_dose_overlap_overlap.py:523-661`; GREEN log at `$HUB/green/pytest_dense_acceptance_floor.log`.) (docs/development/TEST_SUITE_INDEX.md:62.)
- [x] Hardened `studies/fly64_dose_overlap/overlap.py::filter_dataset_by_mask` to bypass scalar/0-D metadata and added `tests/study/test_dose_overlap_overlap.py::test_filter_dataset_by_mask_handles_scalar_metadata` (GREEN log: `$HUB/green/pytest_filter_dataset_by_mask.log`). (DATA-001, specs/overlap_metrics.md §Behavior.)
- [ ] Rerun the counted dense pipeline (after the fix/test above) with logs under the hub: `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`, then immediately invoke `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` before rerunning the metrics helpers if `analysis/metrics_summary.json` is stale so `{analysis}` gains SSIM grid summary/log, verification report/log, highlights log, metrics summary/digest, preview text, and `artifact_inventory.txt`. (docs/TESTING_GUIDE.md §§Phase G orchestrator + metrics; docs/findings.md: PREVIEW-PHASE-001, TEST-CLI-001.)


- **2025-11-13T235900Z bug triage:** `cli/compare_models_dense_train_fix.log:534-566` now reaches the converter but dies in `align_for_evaluation` because `pinn_recon` stays batched `(32, 128, 128)` and `_center_crop` still expects a 2D tensor. Until we reassemble/log the PINN stack before alignment, compare_models cannot finish and `{analysis}/verification_report.json` remains 0/10.
- **Do Now — PINN reassembly + counted rerun (ready_for_implementation):**
  1. Guard the working directory and env vars:
     ```bash
     test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"
     export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
     export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier
     ```
  2. Update `scripts/compare_models.py` so the PINN prediction stack is reassembled via `reassemble_position` (using the Phase G offsets and `args.stitch_crop_size`) before calling `align_for_evaluation`, log the before/after shapes, and add `test_pinn_reconstruction_reassembles_batched_predictions` to `tests/study/test_dose_overlap_comparison.py`.
  3. Run the compare_models regression suite (existing cases + the new test) and archive the GREEN log:
     ```bash
     pytest tests/study/test_dose_overlap_comparison.py::test_baseline_model_predict_receives_both_inputs \
           tests/study/test_dose_overlap_comparison.py::test_baseline_complex_output_converts_to_amplitude_phase \
           tests/study/test_dose_overlap_comparison.py::test_prepare_baseline_inference_data_grouped_flatten_helper \
           tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_batched_predictions -vv \
       | tee "$HUB"/green/pytest_compare_models_reassembly.log
     ```
  4. Re-run the limited smoke with `--n-test-groups 32 --register-ptychi-only` and capture `$HUB/cli/compare_models_dense_train_fix.log`. The log must reach the metrics/report footer; otherwise file `$HUB/red/blocked_<timestamp>_compare_models.md` with the stack trace before touching the pipeline.
  5. Re-run the two Phase D overlap selectors so scalar metadata + geometry floor behavior stay GREEN:
     ```bash
     pytest tests/study/test_dose_overlap_overlap.py::test_filter_dataset_by_mask_handles_scalar_metadata -vv \
       | tee "$HUB"/green/pytest_filter_dataset_by_mask.log
     pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor -vv \
       | tee "$HUB"/green/pytest_dense_acceptance_floor.log
     ```
  6. Execute the counted dense pipeline with clobber:
     ```bash
     python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
       --hub "$HUB" --dose 1000 --view dense --splits train test --clobber \
       |& tee "$HUB"/cli/run_phase_g_dense_stdout.log
     ```
     Immediately confirm `$HUB/cli/phase_d_dense.log` records the regenerated overlap run and that `data/phase_d/dose_1000/dense/{train.npz,test.npz}` plus `data/phase_e/dose_1000/dense/gs2/wts.h5.zip` have fresh timestamps; failures → `$HUB/red/blocked_<timestamp>.md`.
  7. Run the fully parameterized post-verify helper:
     ```bash
     python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
       --hub "$HUB" --dose 1000 --view dense --splits train test --post-verify-only \
       |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log
     ```
  8. If `analysis/metrics_summary.json` predates the rerun, execute `report_phase_g_dense_metrics.py` and `analyze_dense_metrics.py` with `--hub "$HUB"` so `{analysis}` gains refreshed `metrics_summary.json`, `metrics_delta_highlights_preview.txt` (phase-only), and `metrics_digest.md`.
  9. Verify `{analysis}` now contains `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json` (`n_valid=10`), `verify_dense_stdout.log`, `check_dense_highlights.log`, the metrics bundle, `preview.txt`, and `analysis/artifact_inventory.txt`. Any missing artifact or failing check → `$HUB/red/blocked_<timestamp>.md` with the exact command/log path.
 10. Update `summary.md`, `summary/summary.md`, docs/fix_plan.md, and galph_memory with MS-SSIM ±0.000 / MAE ±0.000000 deltas, the pytest/CLI/metrics commands executed, and the confirmation that `analysis/verification_report.json` is GREEN.
- **2025-11-14T011500Z audit:** `analysis/verification_report.json` still reports `n_valid=0`, the newest CLI log remains `run_phase_g_dense_stdout.log` (timestamp 2025-11-12T22:24Z), `cli/phase_d_dense.log` is empty, and `cli/phase_e_dense_gs2_dose1000.log` logs “No jobs match the specified filters” because dense NPZ files were never detected. `analysis/comparison_manifest.json` + `analysis/dose_100000/dense/train/comparison.log` still reference the old dose=100000 run that failed with `ValueError: Dimensions must be equal, but are 128 and 32` inside `Translation.call`, so there is no usable Phase G MS-SSIM/MAE evidence. Do Now stays in ready_for_implementation with the guards above: rerun the two pytest selectors, execute the counted dense pipeline with `--clobber`, verify Phase D/Phase E outputs actually materialize, run the fully parameterized `--post-verify-only` sweep, and regenerate the metrics/digest/preview/inventory bundle under `$HUB/analysis`.
- **2025-11-14T021800Z audit:** The hub remains unchanged: `analysis/verification_report.json` is still 0/10, `analysis/metrics_*` artifacts are absent, `cli/phase_d_dense.log` is a zero-byte file, and `cli/phase_e_dense_gs2_dose1000.log` still ends with “No jobs match the specified filters.” `data/phase_d/dose_1000/dense/{train.npz,test.npz}` and `data/phase_e/dose_1000/dense/gs2/train.log` retain their 2025-11-12 timestamps and there is still no `wts.h5.zip`, while the only Phase G comparison manifest/log pair references the older dose=100000 job that fails in `Translation.call`. Reaffirm the guarded pytest selectors, counted dense pipeline with `--clobber`, fully parameterized `--post-verify-only` sweep, metrics helpers, and the new checkpoint that requires engineers to verify Phase D/Phase E outputs (or record a `$HUB/red/blocked_<timestamp>.md`) before announcing success.
- **2025-11-14T033700Z audit:** No new CLI/analysis artifacts exist after 2025-11-12, `analysis/verification_report.json` still reports `n_valid=0`, and `analysis/dose_1000/dense/train/comparison.log` now contains a `ValueError: Dimensions must be equal, but are 128 and 32` stack trace (Translation → projective_warp_xla) emitted by `scripts.fly64_dose_overlap.comparison`. `analysis/blocker.log` confirms the counted comparison command is still exiting 1 and points at `cli/phase_g_dense_train.log`, so Phase G never produced the SSIM/verification/highlights/metrics/preview/artifact-inventory bundle. Keep the Do Now unchanged: guard `/home/ollie/Documents/PtychoPINN`, export `AUTHORITATIVE_CMDS_DOC` + HUB, rerun the two pytest selectors with GREEN logs, execute the counted pipeline with `--clobber`, immediately run the fully parameterized `--post-verify-only` helper, verify Phase D/E outputs/logs (`data/phase_d/dose_1000/dense/{train.npz,test.npz}`, `data/phase_e/dose_1000/dense/gs2/wts.h5.zip`, CLI transcripts) have fresh timestamps, rerun the metrics/reporting helpers, and only then update docs once `verification_report.json` shows 10/10. If the Comparison failure reproduces, capture a `$HUB/red/blocked_<timestamp>.md` quoting the Translation stack trace plus the exact command/exit code before pausing.
- **2025-11-13T010930Z audit:** `git status --porcelain` only lists the deleted `data/phase_c/run_manifest.json` inside this hub, so per the evidence-only rule we skipped `git pull --rebase` (recorded as `evidence_only_dirty=true`). `{analysis}` is still just `blocker.log`, `{cli}` tops out at `phase_c_generation.log`, `phase_d_dense.log`, and the various `run_phase_g_dense_stdout*.log` files, and `cli/run_phase_g_dense_post_verify_only.log` remains the argparse usage banner because the command omitted `--dose/--view/--splits`. Do Now stays ready_for_implementation: guard the working directory, rerun the pytest selector, execute the counted dense run, immediately run the fully parameterized `--post-verify-only` command, refresh the metrics helpers if `analysis/metrics_summary.json` is stale, and do not stop until `{analysis}` contains the SSIM grid / verification / highlights / metrics / preview / artifact inventory bundle with MS-SSIM ±0.000 / MAE ±0.000000 deltas recorded across hub summaries and docs.
- **2025-11-11T131617Z reality check:** After stashing/restoring the deleted `data/phase_c/run_manifest.json` plus CLI/pytest logs to satisfy `git pull --rebase`, `{analysis}` still only contains `blocker.log` while `{cli}` is limited to `phase_c_generation.log`, `phase_d_dense.log`, and `run_phase_g_dense_stdout.log`. No SSIM grid, verification, preview, metrics, or artifact-inventory outputs exist yet, so the counted dense run + immediate `--post-verify-only` sweep remain the blocking deliverables for this phase.
- **2025-11-12T210000Z retrospective:** Latest hub inspection confirms nothing changed since the prior attempt—`analysis/` still only holds `blocker.log`, `cli/` still has the short trio of logs, and `data/phase_c/run_manifest.json` remains deleted (must be regenerated by the counted run, not restored manually). `cli/phase_d_dense.log` shows the last execution ran from `/home/ollie/Documents/PtychoPINN2` and failed with `ValueError: Object arrays cannot be loaded when allow_pickle=False`, so enforcing `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` before every command stays mandatory. A quick `git log -10 --oneline` retrospective shows no new Ralph commits landed since the previous Do Now, so the focus remains ready_for_implementation with the same pytest + CLI guardrails.
- **2025-11-11T215207Z geometry audit:** Dense Phase D now runs from this repo but still aborts because only 42/5088 train positions (0.8 %) survive the 38.4 px spacing guard even after the greedy fallback; bounding-box analysis (56 399 px² area) caps the theoretical acceptance at ≈0.96 %, so the legacy 10 % floor can never succeed. `{analysis}` still only contains `blocker.log`; there are no SSIM grid, verification, highlights, metrics, preview, or artifact-inventory artifacts yet. Next step is to encode the geometry-aware acceptance floor, land a pytest, and rerun the dense pipeline with the post-verify sweep so `{analysis}` fills with evidence.
- **2025-11-11T222901Z audit:** Re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001 / ACCEPTANCE-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md §12, the hub plan + summaries, `analysis/blocker.log`, `cli/phase_d_dense.log`, `studies/fly64_dose_overlap/overlap.py`, `tests/study/test_dose_overlap_overlap.py`, and `$HUB/green/pytest_dense_acceptance_floor.log`. Geometry-aware acceptance bound + pytest are already landed (cf. `studies/fly64_dose_overlap/overlap.py:334-555` and the GREEN log), but `{analysis}` still only contains `blocker.log` because the counted rerun and `--post-verify-only` sweep never executed after the code change (`cli/phase_d_dense.log` still shows the “minimum 10.0%” message). Do Now stays ready_for_implementation: guard `pwd -P`, export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`, rerun `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor -vv | tee "$HUB"/green/pytest_dense_acceptance_floor.log`, then run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` immediately followed by `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`, then rerun `report_phase_g_dense_metrics.py` and `analyze_dense_metrics.py` if `analysis/metrics_summary.json` predates the rerun so `{analysis}` captures `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, preview verdict text, and `artifact_inventory.txt` with MS-SSIM ±0.000 / MAE ±0.000000 deltas logged in hub summary + docs/fix_plan.md + galph_memory. Blockers belong under `$HUB/red/blocked_<timestamp>.md`.

- **2025-11-11T220602Z spec audit:** `{analysis}` still only contains `blocker.log` even though `cli/` now has multiple `run_phase_g_dense_stdout*.log` files; there is still no SSIM grid summary/log, verification report/log, highlights log, preview text, metrics summary/digest, or artifact inventory. Reviewing `studies/fly64_dose_overlap/overlap.py` and `tests/study/test_dose_overlap_overlap.py` shows the helper currently emits `geometry_aware_floor` (50 % of the theoretical bound, floored at 1 %) instead of the required `geometry_acceptance_bound` (actual bounding-box acceptance capped at 10 %). Ready_for_implementation Do Now: (1) update `SpacingMetrics` + `_metadata` + `metrics_bundle.json` to expose `geometry_acceptance_bound` and set `effective_min_acceptance = clamp(bound, ε, 0.10)` without the 0.5 fudge; (2) update `test_generate_overlap_views_dense_acceptance_floor` (plus fixtures) to assert the renamed JSON keys and bounded logic; (3) guard `pwd -P` plus `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, run `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor -vv | tee "$HUB"/green/pytest_dense_acceptance_floor.log`, then execute the counted `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` immediately followed by `--post-verify-only`; rerun the metrics helpers if `analysis/metrics_summary.json` predates the run, and do not exit until `{analysis}` lists SSIM grid summary/log, verification report/log, highlights log, metrics summary/digest, preview verdict, and `artifact_inventory.txt` with MS-SSIM ±0.000 / MAE ±0.000000 deltas logged across the hub summaries + docs/fix_plan.md.


#### Next Do Now — Baseline wiring + dense rerun (2025-11-14T071500Z, ready_for_implementation)
1. Guard from `/home/ollie/Documents/PtychoPINN`: `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` and export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier"`; tee every command under `$HUB` (POLICY-001 / TYPE-PATH-001 / PYTHON-ENV-001).
2. Update `scripts/compare_models.py` so the grouped baseline path is safe before we touch the counted pipeline:
   - After computing the diffraction channel count (`diffraction_channels = test_container.X.shape[-1]`), assert it is a perfect square by checking `diffraction_channels == required_gridsize ** 2`; raise `ValueError` with the channel count if not. Log the resolved gridsize even when no correction is needed, and keep the forced `params.cfg['gridsize'] = required_gridsize` path that landed in `c63c99eb`.
   - When we flatten grouped diffraction for the baseline path, also capture the flattened offsets from `prepare_baseline_inference_data` and call `baseline_model.predict([baseline_input, baseline_offsets], batch_size=32, verbose=1)` so Lightning receives the coordinate tensor it expects (compare to the failure in `$HUB/cli/compare_models_smoke_test/logs/logs/debug.log:370-402`). Log both tensor shapes for traceability and warn when offsets are missing.
   - Add regression coverage in `tests/study/test_dose_overlap_comparison.py`: (a) a `test_prepare_baseline_inference_data_flattens_grouped_offsets` that constructs a fake container with 4-channel diffraction and asserts both the flattened data and offsets match `(channels * groups, 128, 128, 1)`; (b) extend `test_execute_comparison_jobs_invokes_compare_models` (or add a new test) to monkeypatch the baseline model so the test can assert `predict` is invoked with two tensors (data + offsets) and that channel-flattening happens for grouped inputs.
3. Re-run the limited smoke before the expensive rerun, capturing logs under `$HUB/cli/compare_models_dense_train_fix.log`:
   ```bash
   python scripts/compare_models.py \
     --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 \
     --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 \
     --test_data "$HUB"/data/phase_c/dose_1000/patched_train.npz \
     --output_dir "$HUB"/analysis/dose_1000/dense/train \
     --n-test-groups 32 --register-ptychi-only \
     |& tee "$HUB"/cli/compare_models_dense_train_fix.log
   ```
   If the baseline still errors, stop and save the stack trace to `$HUB/red/blocked_<timestamp>_compare_models.md` instead of continuing.
4. Re-run the GREEN guards with fresh logs (`$HUB/green/pytest_filter_dataset_by_mask.log` and `$HUB/green/pytest_dense_acceptance_floor.log`) using the two Phase D selectors documented in docs/TESTING_GUIDE.md.
5. Execute the counted dense pipeline with clobber (`python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`) so Phase C/D/E artifacts are regenerated rather than restored.
6. Immediately run the fully parameterized helper to refresh verification/highlights (`python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`).
7. If `analysis/metrics_summary.json` predates the rerun, execute `report_phase_g_dense_metrics.py` and `analyze_dense_metrics.py` again so `{analysis}` receives `metrics_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, and the preview text alongside the SSIM/verification logs.
8. Verification gate: confirm `$HUB/cli/phase_d_dense.log` records the overlap command, `$HUB/cli/phase_e_dense_gs2_dose1000.log` shows real jobs, `data/phase_d/dose_1000/dense/{train.npz,test.npz}` and `data/phase_e/dose_1000/dense/gs2/wts.h5.zip` exist, and `analysis/verification_report.json` flips to 10/10 with the SSIM grid / verification / highlights / metrics / preview / artifact-inventory bundle. If any artifact/log is still missing—or `analysis/dose_1000/dense/train/comparison.log` / `$HUB/cli/phase_g_dense_train.log` still report a failure—capture a `$HUB/red/blocked_<timestamp>.md` with the command, exit code, and error signature before handing the focus back.

- **2025-11-14T071500Z audit:** Translation crash is fixed (PINN inference now succeeds) but the limited smoke log (`cli/compare_models_smoke_test/logs/logs/debug.log:370-402`) shows the baseline path now fails with `ValueError: Layer "functional_10" expects 2 input(s), but it received 1 input`. The GREEN guard (`cli/compare_models_dense_train_fix.log`) also shows the command never completed, and `{analysis}/verification_report.json` is still 0/10 because the counted rerun has not been restarted since 2025-11-12. The updated Do Now requires wiring the baseline offsets, adding regression tests, rerunning the smoke command, and only then executing the counted pipeline + metrics/report helpers until the verification bundle exists.
- **2025-11-12T231800Z audit:** After stash→pull→pop the repo remains dirty only under this hub; `{analysis}` still just holds `blocker.log`, `cli/` still stops at `{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout}.log`, and there are zero SSIM grid, verification, preview, metrics, or artifact-inventory artifacts. `cli/phase_d_dense.log` again ends with `ValueError: Object arrays cannot be loaded when allow_pickle=False`, confirming the dense rerun never advanced past Phase D inside `/home/ollie/Documents/PtychoPINN`.
- **2025-11-13T003000Z audit:** `timeout 30 git pull --rebase` now runs cleanly (no stash cycle needed) and `git log -10 --oneline` shows the overlap `allow_pickle=True` fix (`5cd130d3`) is already on this branch, so the lingering ValueError in `cli/phase_d_dense.log` is stale output from `/home/ollie/Documents/PtychoPINN2`. The active hub still lacks any counted dense evidence: `analysis/` only holds `blocker.log`, `{cli}` contains `{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout(_retry,_v2).log}`, and there are zero SSIM grid summaries, verification/previews, metrics delta files, or artifact inventory snapshots. With the fix landed locally, the only remaining action is to rerun `run_phase_g_dense.py --clobber ...` followed immediately by `--post-verify-only` **from /home/ollie/Documents/PtychoPINN** so Phase C manifests regenerate and `{analysis,cli}` fill with real artifacts.
- **2025-11-13T012200Z audit:** `git status --porcelain` only lists the stale `cli/phase_d_dense.log` inside the active hub, so per the evidence-only rule I skipped `git pull --rebase` (recorded `evidence_only_dirty=true`) and exported `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before editing. Re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001 / ACCEPTANCE-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, specs/data_contracts.md §12, docs/fix_plan.md, galph_memory.md, input.md, `summary.md`, `summary/summary.md`, `analysis/blocker.log`, `cli/phase_d_dense.log`, and `cli/run_phase_g_dense_post_verify_only.log`. `{analysis}` is still just `blocker.log` and the post-verify CLI log remains the argparse usage banner because the command missed `--dose/--view/--splits`, so SSIM grid / verification / highlights / metrics / preview / inventory artifacts do not exist yet. Reissue the same ready_for_implementation Do Now with the fully parameterized post-verify command and the MS-SSIM ±0.000 / MAE ±0.000000 reporting requirements.

#### Next Do Now — Translation fix + counted dense rerun (2025-11-14T060000Z)

1. **Guard environment:** From `/home/ollie/Documents/PtychoPINN`, run `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` and export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier"`. Every command must `tee` into `$HUB` (TYPE-PATH-001 / POLICY-001 / PYTHON-ENV-001).
2. **Normalize baseline outputs:** Update `scripts/compare_models.py` so grouped runs convert the baseline model’s single complex tensor into amplitude/phase arrays when `predict()` returns a complex tensor while still supporting legacy `[amplitude, phase]` lists. Log both shapes, continue forcing/logging `params.cfg['gridsize']`, and keep the `[baseline_input, baseline_offsets]` call so `Translation.call` never sees mismatched offsets (DATA-001 / TYPE-PATH-001).
3. **Regression tests:** Extend `tests/study/test_dose_overlap_comparison.py` so `test_baseline_model_predict_receives_both_inputs` still asserts the two-input signature **and** add `test_baseline_complex_output_converts_to_amplitude_phase` that injects a mocked complex tensor through the new converter. Run  
   `pytest tests/study/test_dose_overlap_comparison.py::test_prepare_baseline_inference_data_grouped_flatten_helper tests/study/test_dose_overlap_comparison.py::test_baseline_model_predict_receives_both_inputs tests/study/test_dose_overlap_comparison.py::test_baseline_complex_output_converts_to_amplitude_phase -vv |& tee "$HUB"/green/pytest_compare_models_baseline_output.log`  
   (failures → `$HUB/red/blocked_<timestamp>_pytest_compare_models.md`). Keep the Selector in `input.md` aligned with the new test node.
4. **Limited smoke:** Repeat the focused compare_models command to prove the converter works before touching the counted pipeline:  
   `python scripts/compare_models.py --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 --test_data "$HUB"/data/phase_c/dose_1000/patched_train.npz --output_dir "$HUB"/analysis/dose_1000/dense/train --n-test-groups 32 --register-ptychi-only |& tee "$HUB"/cli/compare_models_dense_train_fix.log`.  
   If the run still raises `Unexpected baseline model output format` (or any new exception), capture `$HUB/red/blocked_<timestamp>_compare_models.md` with the command + stack trace before proceeding.
5. **Phase D guards:** Re-run the targeted selectors with fresh GREEN logs under `$HUB/green/` per ACCEPTANCE-001 / TEST-CLI-001:  
   - `pytest tests/study/test_dose_overlap_overlap.py::test_filter_dataset_by_mask_handles_scalar_metadata -vv | tee "$HUB"/green/pytest_filter_dataset_by_mask.log`  
   - `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor -vv | tee "$HUB"/green/pytest_dense_acceptance_floor.log`
6. **Counted dense rerun:** `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` so Phase C/D/E artifacts regenerate instead of being restored.
7. **Fully parameterized post-verify:** `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` to refresh SSIM grid + verifier/highlights evidence immediately after the rerun.
8. **Metrics + preview bundle:** If `analysis/metrics_summary.json` predates the rerun, execute `report_phase_g_dense_metrics.py` and `analyze_dense_metrics.py` again so `{analysis}` contains `metrics_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, and the preview text with MS-SSIM ±0.000 / MAE ±0.000000 deltas (PREVIEW-PHASE-001).
9. **Verification gate:** Confirm `$HUB/cli/phase_d_dense.log`, `$HUB/cli/phase_e_dense_gs2_dose1000.log`, `data/phase_d/dose_1000/dense/{train.npz,test.npz}`, and `data/phase_e/dose_1000/dense/gs2/wts.h5.zip` have fresh timestamps, `analysis/verification_report.json` flips to 10/10, and `{analysis}` contains the SSIM grid / verification / highlights / metrics / preview / artifact-inventory bundle. If any artifact/log is missing—or `analysis/dose_1000/dense/train/comparison.log` / `$HUB/cli/phase_g_dense_train.log` still fails—stop and capture `$HUB/red/blocked_<timestamp>.md` with the exact command, exit code, and first stack trace before handing the focus back.
- **2025-11-11T191109Z audit:** `timeout 30 git pull --rebase` reported “Already up to date,” and revisiting docs/index.md + docs/findings.md reconfirmed the governing findings (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001). The hub has **no progress**: `analysis/` still only contains `blocker.log`, `{cli}` is limited to `phase_c_generation.log`, `phase_d_dense.log`, and `run_phase_g_dense_stdout(_retry,_v2).log`, and `cli/phase_d_dense.log` again ends with the local `ValueError: Object arrays cannot be loaded when allow_pickle=False` stack trace even though the fix landed. There are zero SSIM grid summaries, verification/highlights logs, preview text, metrics digests, or artifact-inventory files. The counted dense rerun + immediate `--post-verify-only` sweep therefore remains the blocking deliverable; rerun both commands from `/home/ollie/Documents/PtychoPINN` with the mapped pytest guard and tee’d CLI logs into this hub.
- **2025-11-11T180800Z audit:** Repeated the `git stash push --include-untracked` → `timeout 30 git pull --rebase` → `git stash pop` flow to stay synced while preserving the deleted `data/phase_c/run_manifest.json` and local `.bak` notes. Hub contents remain unchanged: `analysis/` only has `blocker.log` and `cli/` still holds `phase_c_generation.log`, `phase_d_dense.log`, and the `run_phase_g_dense_stdout*.log` variants—there are still zero SSIM grid, verification, preview, metrics, highlights, or artifact-inventory outputs. Importantly, `cli/phase_d_dense.log` shows the failed overlap step was executed **inside** `/home/ollie/Documents/PtychoPINN` (not the secondary clone) and still hit `ValueError: Object arrays cannot be loaded when allow_pickle=False`, which means the post-fix pipeline has never been re-run. Guard every command with `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` and regenerate the Phase C manifest via the counted run rather than restoring it manually.
- **2025-11-12T235900Z audit:** The hub does contain `cli/run_phase_g_dense_post_verify_only.log`, but it is only the argparse usage banner because the command omitted `--dose/--view/--splits`. `{analysis}` still only has `blocker.log`, so SSIM grid/verification/highlights/metrics/preview/inventory artifacts remain absent. Re-run the counted pipeline and copy the full post-verify command (with dose/view/splits) directly from this plan to avoid another no-op.
- **2025-11-11T185738Z retrospective:** Rechecked the hub immediately after the latest stash→pull→pop sync and scanned `git log -10 --oneline`; no Ralph commits landed after `e3b6d375` (reports evidence), so the prior Do Now was never executed. The hub is still missing every Phase G artifact: `analysis/` contains only `blocker.log`, `{cli}` stops at `{phase_c_generation,phase_d_dense,run_phase_g_dense_stdout(_retry,_v2).log}`, and `analysis/metrics_delta_highlights_preview.txt`, SSIM grid summaries, verification/highlights logs, metrics digests, and artifact inventories **do not exist**. `cli/phase_d_dense.log` continues to end with the pre-fix `ValueError: Object arrays cannot be loaded when allow_pickle=False` from this repo, confirming the counted dense rerun plus immediate `--post-verify-only` sweep still never happened.
- **2025-11-11T192358Z audit:** Re-read docs/index.md plus docs/findings.md (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001) and audited the active hub again: `analysis/` still only holds `blocker.log`, while `cli/` is capped at `phase_c_generation.log`, `phase_d_dense.log`, and the trio of `run_phase_g_dense_stdout*.log`. `cli/phase_d_dense.log` ends with the local `ValueError: Object arrays cannot be loaded when allow_pickle=False`, proving the counted dense run never succeeded inside this repo. Reissue the ready_for_implementation Do Now: (1) rerun the mapped pytest collect-only selector plus `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` (logs under `$HUB/collect` and `$HUB/green`) so banner guards stay GREEN; (2) execute `plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber` from `/home/ollie/Documents/PtychoPINN` with stdout piped to `$HUB/cli/run_phase_g_dense_stdout.log`; (3) immediately run `--post-verify-only` into `$HUB/cli/run_phase_g_dense_post_verify_only.log`; (4) verify the rerun produces `analysis/ssim_grid_summary.md`, `analysis/ssim_grid.log`, `analysis/verification_report.json`, `analysis/verify_dense_stdout.log`, `analysis/check_dense_highlights.log`, `analysis/metrics_summary.json`, `analysis/metrics_delta_highlights_preview.txt` (phase-only per PREVIEW-PHASE-001), `analysis/metrics_digest.md`, and `analysis/artifact_inventory.txt`; and (5) publish MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid references, and verification/highlights paths across the hub summaries, docs/fix_plan.md, and galph_memory before closing the loop.
- **2025-11-11T205022Z audit:** `git status -sb` showed `?? docs/fix_plan.md.bak` (whitelisted) plus `?? docs/iteration_scores_262-291.csv`, so I ran `timeout 30 git pull --rebase` to sync before editing. Re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md §Phase G orchestrator, docs/development/TEST_SUITE_INDEX.md (study selectors), the hub `plan/plan.md`, hub summaries, `analysis/blocker.log`, and `cli/phase_d_dense.log`. Nothing has improved: `{analysis}` still only has `blocker.log`, `{cli}` stops at `phase_c_generation.log`, `phase_d_dense.log`, and the `run_phase_g_dense_stdout*` variants, the deleted `data/phase_c/run_manifest.json` still needs regeneration instead of manual restore, and `cli/phase_d_dense.log` continues to end with `ValueError: Object arrays cannot be loaded when allow_pickle=False` emitted **inside `/home/ollie/Documents/PtychoPINN`** even though commit `5cd130d3` fixed overlap.py. Reissue the ready_for_implementation Do Now with no wiggle room:
  1. `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` and export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` plus `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier` before touching anything (TYPE-PATH-001, TEST-CLI-001).
  2. Recreate GREEN banners: `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log` and `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log`.
  3. Execute the counted dense pipeline with clobber archive + regenerated Phase C manifest: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` (POLICY-001 + DATA-001 guard). This must recreate `data/phase_c/run_manifest.json` instead of restoring backups.
  4. Immediately run `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log` so SSIM grid → verifier → highlights checker rerun with the fixed overlap pipeline.
  5. If `analysis/metrics_summary.json` predates the new outputs, rerun `python plans/active/.../bin/report_phase_g_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json` followed by `python plans/active/.../bin/analyze_dense_metrics.py --hub "$HUB"` so `metrics_delta_highlights_preview.txt` (phase-only per PREVIEW-PHASE-001) and `metrics_digest.md` align with the new evidence set.
  6. Confirm `{analysis}` now contains **all** of the following: `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, and `artifact_inventory.txt`. Record MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, selector names, and CLI log paths in `$HUB/summary/summary.md`, docs/fix_plan.md, and galph_memory; stash any failure transcript under `$HUB/red/blocked_$(date -u +%FT%H%M%SZ).md` with the exact command + exit code before stopping.
- **2025-11-11T215650Z audit:** `git status -sb` still lists non-hub docs (`docs/ITERATION_AUDIT_INDEX.md`, `docs/iteration_analysis_audit_full.md`, `docs/iteration_scores_262-291.csv`, etc.), so I ran `timeout 30 git pull --rebase` (already up to date), re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md §Phase G, docs/development/TEST_SUITE_INDEX.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, specs/data_contracts.md, specs/ptychodus_api_spec.md, docs/architecture.md, galph_memory.md, input.md, the hub plan, summary.md, summary/summary.md, and `cli/phase_d_dense.log`. A scripted geometry probe over `data/phase_c/dose_1000/patched_{train,test}.npz` shows: train span 335.82×167.94 px (area 56 399 px²) ⇒ ≤48.7 viable positions and a 0.957 % acceptance ceiling; greedy acceptance is 42/5088 (0.825 %). Test bound is 0.934 % with greedy acceptance 0.786 %. Consequently the hard-coded 10 % threshold is impossible to satisfy even with perfect packing, so we must compute/log an adaptive floor (e.g., `min(0.10, theoretical_bound × PACKING_MARGIN)` with PACKING_MARGIN≈0.85) and persist the derived values in the metrics JSON/bundle before rerunning Phase G. Do Now: (1) Extend `studies/fly64_dose_overlap/overlap.py` with span/area/`theoretical_max_acceptance`/`adaptive_acceptance_floor` fields (printed + stored via `SpacingMetrics.to_dict()`), and replace the guard to use the adaptive floor for both direct and greedy selection; (2) add `tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor` plus any supporting assertions (update existing tests accordingly, then refresh `docs/TESTING_GUIDE.md` + `docs/development/TEST_SUITE_INDEX.md` after GREEN); (3) guard `pwd -P`, export `AUTHORITATIVE_CMDS_DOC` + `HUB`, rerun `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor -vv | tee "$HUB"/green/pytest_dense_acceptance_floor.log`, then execute `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` immediately followed by `python plans/active/.../bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; (4) rerun the metrics helpers if needed so `{analysis}` finally holds SSIM grid summary/log, verification report/log, highlights log, preview verdict, `metrics_summary.json`, `metrics_delta_highlights_preview.txt`, `metrics_digest.md`, and `artifact_inventory.txt`, then log MS-SSIM ±0.000 / MAE ±0.000000 deltas + selector + CLI references in the hub summaries/docs/memory; (5) blockers go under `$HUB/red/blocked_<timestamp>.md`. Note: `docs/prompt_sources_map.json` and `docs/pytorch_runtime_checklist.md` still do not exist—no changes required until they appear upstream.
- **2025-11-11T213530Z audit:** `git status -sb` still shows `?? docs/fix_plan.md.bak` and `?? docs/iteration_scores_262-291.csv`, so I reran `timeout 30 git pull --rebase` (already up to date) before exporting `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`. Re-read docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / TYPE-PATH-001 / STUDY-001 / TEST-CLI-001 / PREVIEW-PHASE-001 / PHASEC-METADATA-001), docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/TESTING_GUIDE.md §Phase G orchestrator, docs/development/TEST_SUITE_INDEX.md:62, specs/data_contracts.md, docs/fix_plan.md, galph_memory.md, the hub plan, summary.md, summary/summary.md, `analysis/blocker.log`, `cli/phase_d_dense.log`, and `cli/run_phase_g_dense_clobber.log`. Hub reality check: `{analysis}` still only contains `blocker.log`; `{cli}` now has additional partial stdout logs but there is **no** `run_phase_g_dense_post_verify_only.log`; `data/phase_c/run_manifest.json` plus the `patched*.npz` train/test splits are missing entirely (the prior `--clobber` attempt never finished); and `cli/phase_d_dense.log` again shows the local `ValueError: Object arrays cannot be loaded when allow_pickle=False`, confirming the overlap.py fix has never actually executed inside this workspace. Ready_for_implementation Do Now remains unchanged: guard `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` plus the HUB/COMMAND exports, rerun the two pytest selectors with logs under `$HUB/collect` and `$HUB/green`, execute `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` immediately followed by `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`, rerun `report_phase_g_dense_metrics.py`/`analyze_dense_metrics.py` if `analysis/metrics_summary.json` is stale, and do not stop until `{analysis}` holds the SSIM grid summary/log, verification_report.json, verify/highlights logs, metrics_summary.json, metrics_delta_highlights_preview.txt, metrics_digest.md, and artifact_inventory.txt with MS-SSIM ±0.000 / MAE ±0.000000 deltas + preview verdict captured in the hub summaries, docs/fix_plan.md, and galph_memory. Any failed command must be recorded under `$HUB/red/blocked_$(date -u +%FT%H%M%SZ).md`.
- **2025-11-11T195519Z audit:** `git status -sb` shows only the deleted `data/phase_c/run_manifest.json` plus `docs/fix_plan.md.bak`, so we skipped `git pull --rebase` under the evidence-only dirty exemption and re-confirmed via `git log -10 --oneline` that no new Ralph commits landed since the last directive. The hub is unchanged: `analysis/` still only contains `blocker.log`; `{cli}` stops at `phase_c_generation.log`, `phase_d_dense.log`, and `run_phase_g_dense_stdout(_retry,_v2).log`; and `analysis/metrics_summary.json`, `analysis/metrics_delta_highlights_preview.txt`, SSIM grid summaries/logs, verification/highlights outputs, metrics digest, preview text, and artifact inventory files **do not exist**. `cli/phase_d_dense.log` still ends with `ValueError: Object arrays cannot be loaded when allow_pickle=False` even though `studies/fly64_dose_overlap/overlap.py` now forces `allow_pickle=True`, proving the failure evidence predates the fix and the dense rerun never re-executed here. Reissue the ready_for_implementation Do Now with explicit guardrails: (1) guard `pwd -P` equals `/home/ollie/Documents/PtychoPINN`, export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` + `HUB` from the How-To Map, and rerun the mapped pytest collect-only selector plus `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain` with logs under `$HUB/collect` and `$HUB/green`; (2) execute `python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log`, let it regenerate `data/phase_c/run_manifest.json`, and confirm `{analysis}` gains `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_delta_highlights_preview.txt` (phase-only, no “amplitude”), `metrics_digest.md`, and `artifact_inventory.txt`; (3) immediately run `--post-verify-only` into `$HUB/cli/run_phase_g_dense_post_verify_only.log` so the shortened chain refreshes SSIM grid + verification artifacts and rewrites `analysis/artifact_inventory.txt`; (4) run the report helper (`report_phase_g_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json`) if the sanity table is stale; and (5) update `$HUB/summary/summary.md`, docs/fix_plan.md, and galph_memory with MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid + verification/highlights file names, pytest selectors, and CLI paths. If any command fails, capture `tee` output under `$HUB/red/blocked_<timestamp>.md`, leave artifacts in place, and stop so the supervisor can triage.
- **2025-11-11T202702Z audit:** Evidence-only dirty exemption applied (`git status -sb` shows just the deleted `plans/.../data/phase_c/run_manifest.json` plus `docs/fix_plan.md.bak`), so pull was skipped while `git log -10 --oneline` confirmed zero new Ralph commits. `docs/prompt_sources_map.json` is still absent, so docs/index.md remains the authoritative source list; docs/findings.md guardrails (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001) continue to govern the run. The hub is still empty: `analysis/` only has `blocker.log`; `{cli}` is limited to `phase_c_generation.log`, `phase_d_dense.log`, `run_phase_g_dense_stdout(_retry,_v2).log`, and the partial `run_phase_g_dense_clobber.log` that died mid-dose; Phase C manifests remain deleted; and there are no SSIM grid summaries/logs, verification outputs, highlights, preview text, metrics summary/digest, or artifact inventory files anywhere inside this repo. `cli/phase_d_dense.log` shows the failure happened **inside `/home/ollie/Documents/PtychoPINN`** and still ends with `ValueError: Object arrays cannot be loaded when allow_pickle=False`, proving the overlap.py fix has never been rerun here. Reissue the ready_for_implementation Do Now verbatim: (1) guard `pwd -P` equals `/home/ollie/Documents/PtychoPINN`, export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` and `HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier`, rerun `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv | tee "$HUB"/collect/pytest_collect_post_verify_only.log`, and execute `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv | tee "$HUB"/green/pytest_post_verify_only.log`; (2) run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log` so Phase C manifests are regenerated and `{analysis}` gains `ssim_grid_summary.md`, `ssim_grid.log`, `verification_report.json`, `verify_dense_stdout.log`, `check_dense_highlights.log`, `metrics_summary.json`, `metrics_delta_highlights_preview.txt` (phase-only per PREVIEW-PHASE-001), `metrics_digest.md`, and `artifact_inventory.txt`; (3) immediately follow with `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --post-verify-only |& tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log`; (4) rerun `report_phase_g_dense_metrics.py --hub "$HUB" --metrics "$HUB"/analysis/metrics_summary.json` if the sanity table is stale; and (5) publish MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict, SSIM grid references, verification/highlights links, pytest selectors, and CLI log paths in `$HUB/summary/summary.md`, docs/fix_plan.md, and galph_memory. Capture any blocker under `$HUB/red/blocked_$(date -u +%FT%H%M%SZ).md` before stopping.
- [ ] Execute a counted dense Phase C→G run with `--clobber` into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`, populating `{analysis,cli}` with real artifacts (phase logs, metrics_delta_summary.json, metrics_delta_highlights_preview.txt, ssim_grid_summary.md, verification_report.json, verify_dense_stdout.log, check_dense_highlights.log, artifact_inventory.txt). **Owner:** Ralph. **Evidence:** CLI/stdout log archived at `$HUB/cli/run_phase_g_dense_stdout.log`.
- [ ] Immediately rerun `run_phase_g_dense.py --hub "$HUB" --post-verify-only` against the fresh artifacts, confirming the shortened chain regenerates SSIM grid + verification outputs and refreshes `analysis/artifact_inventory.txt`. Archive CLI output as `$HUB/cli/run_phase_g_dense_post_verify_only.log`.
- [ ] Update `$HUB/summary/summary.md`, docs/fix_plan.md, and galph_memory with MS-SSIM ±0.000 / MAE ±0.000000 deltas, preview verdict (PREVIEW-PHASE-001), verifier/highlight log references, pytest selectors, and doc/test guard status.
- [ ] If verification surfaces discrepancies, capture blocker logs under `$HUB/red/`, append to docs/fix_plan.md Attempts History with failure signature, and rerun once resolved.


- **2025-11-13T000500Z dwell gate:** `{analysis}` still shows only `blocker.log` and `cli/run_phase_g_dense_post_verify_only.log` remains the argparse usage banner, so the focus is now frozen in `blocked_escalation`. See `analysis/dwell_escalation_report.md` for the Tier 3 summary and required actions; no further planning/doc loops may target this initiative until Ralph lands the counted rerun + post-verify evidence described above.

## Execution Hygiene
- **Hub reuse:** Stick to the two active hubs noted at the top of this plan (Phase E training + Phase G comparison). Per `docs/INITIATIVE_WORKFLOW_GUIDE.md`, only create a new timestamped hub when fresh production evidence is produced; otherwise append to the existing hub’s `summary.md`.
- **Dirty hub protocol:** Supervisors must log residual artifacts (missing SSIM grid, deleted manifest, etc.) inside the hub’s `summary/summary.md` *and* `docs/fix_plan.md` before handing off so Ralph is never asked to reconstruct Phase C blindly.
- **Command guards:** All How-To Map entries must be copy/pasteable. Prepend `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"` plus the required `AUTHORITATIVE_CMDS_DOC`/`HUB` exports and avoid `plans/...` ellipses, ensuring evidence stays local to this repo.

### Phase H — Documentation & Gaps
- Note the current code/doc oversampling status and any deviations; update docs/fix_plan.md with artifact paths and outcomes.

## Future Work / Out-of-Scope
- PyTorch training parity: port the Phase E CLI/tests back onto `ptycho_torch` only after TensorFlow evidence is stable; track as a separate fix-plan item so it does not block this initiative.
- Continuous overlap sweeps: expand beyond the current `{s_img, m_group}` grid once the derived overlap reporting proves stable; consider automating the sweep via `studies/.../design.py`.

## Risks & Mitigations
- pty-chi dependency not vendored: document version and environment; cache outputs.
- Oversampling misuse: confirm K≥C and log branch choice; include spacing histograms.
- Data leakage: enforce y-axis split; avoid mixed halves in train/test.

## Evidence & Artifacts
- All runs produce logs/plots/CSV in reports/<timestamp>/ per condition. Summaries collected in summary.md.
