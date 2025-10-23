# EB2 Debug Triage (2025-10-23T091500Z)

## Context
- Focus: `[ADR-003-BACKEND-API]` Phase EB2 — scheduler & gradient accumulation knobs
- Trigger: Ralph's Attempt #62 (commit 6de34107 + evidence commit 6765d545) claimed EB2 completion but integration regression surfaced in full-suite log.

## Observations
- `pytest_full_suite.log` (`plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/green/pytest_full_suite.log`) shows
  - `tests/torch/test_bundle_loader_returns_modules`
  - `tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer`
  failing with `RuntimeError: Early stopping conditioned on metric val_loss which is not available`.
- Lightning module logs metrics under `model.val_loss_name` (e.g., `poisson_val_loss`), so `_train_with_lightning` monitoring `val_loss` cannot resolve the metric when validation is enabled.
- Do Now mapped selector `tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_accumulation` was not authored or executed; artifact directory lacks the expected `pytest_workflows_accum_{red,green}.log` pair.
- Plan + ledger drift:
  - `phase_e_execution_knobs/plan.md` still lists EB2.A/B/C rows as `[ ]` despite Attempt #62 claiming completion.
  - `docs/fix_plan.md` Attempt #37 (misnumbered) asserts EB2 COMPLETE, conflicting with failing integration tests.

## Hypotheses
1. **Monitor mismatch** *(high confidence)* — `_train_with_lightning` should source `monitor_metric`/`checkpoint filename` from `model.val_loss_name` whenever validation data is present. Adjusting monitor wiring plus targeted unit test should resolve Lightning runtime failure.
2. **Workflow coverage gap** *(medium confidence)* — Lack of workflow-level test for accumulation means CLI → Trainer propagation of `accum_steps` still unverified. Add `TestLightningExecutionConfig::test_trainer_receives_accumulation` per EB2 plan to prevent regressions.
3. **Plan hygiene** *(supporting)* — Update `phase_e_execution_knobs/plan.md` and `docs/fix_plan.md` to reflect incomplete EB2 state, avoiding premature completion signals.

## Recommended Next Steps
1. Patch `_train_with_lightning` to derive `monitor_metric` (and checkpoint filename template) from `model.val_loss_name` when validation container exists; fall back to execution_config default otherwise. Add focused pytest asserting Lightning callbacks monitor the dynamic metric.
2. Author the missing workflow test for accumulation (`TestLightningExecutionConfig::test_trainer_receives_accumulation`) ensuring Trainer receives `accumulate_grad_batches` override. Capture RED→GREEN logs under new timestamped folder.
3. Rerun mapped selectors + targeted integration test (`tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer`) to prove regression resolved; archive logs to the new reports directory.
4. Update `phase_e_execution_knobs/plan.md` EB2 rows and `docs/fix_plan.md` Attempts history to record the failure and prescribe the fixes above.

## References
- specs/ptychodus_api_spec.md §4.9, §7.1 — Execution knob contract / CLI tables
- docs/workflows/pytorch.md §12 — Training execution flags (needs accumulation note once fixed)
- plans/active/ADR-003-BACKEND-API/reports/2025-10-23T081500Z/eb2_plan.md — Authoritative EB2 checklist
