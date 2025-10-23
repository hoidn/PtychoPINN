Summary: Fix EB2 regression by aligning Lightning monitor callbacks with model.val_loss_name and proving scheduler/accum coverage end-to-end.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Phase EB2 — Scheduler & Gradient Accumulation
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_accumulation -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_monitor_uses_val_loss_name -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/{summary.md,red/,green/,pytest_full_suite.log}
Do Now:
- EB2.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-23T081500Z/eb2_plan.md — add workflow-level RED tests covering accumulate_grad_batches + dynamic monitor (trainer should watch model.val_loss_name); run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_accumulation -vv` and `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_monitor_uses_val_loss_name -vv` expecting failure; capture logs under plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/red/. tests: targeted.
- EB2.B @ plans/active/ADR-003-BACKEND-API/reports/2025-10-23T081500Z/eb2_plan.md — update `_train_with_lightning` to derive monitor/checkpoint strings from `model.val_loss_name`, ensure callbacks respect accumulation override, then rerun mapped selectors to GREEN and `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`; archive outputs under plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/green/ plus refreshed `pytest_full_suite.log` if executed. tests: mapped.
- EB2.C1-C3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-23T081500Z/eb2_plan.md — once tests pass, refresh spec/workflow note if wording changes, mark plan rows `[x]`, append Attempt #63 in docs/fix_plan.md, and author loop summary in the new timestamp directory. tests: none.
If Blocked: Capture failing selector logs in plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/blockers.md, note hypothesis plus stack trace, and set EB2 rows `[P]` in plan + ledger with explanation.
Priorities & Rationale:
- specs/ptychodus_api_spec.md §4.9: execution knobs must honour Lightning parity, including dynamic metrics.
- docs/workflows/pytorch.md §12: documentation already advertises scheduler/accum flags—behaviour must match.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md EB2 rows `[P]`: unblock checklist by finishing wiring + tests.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-23T091500Z/summary.md: supervisor triage describing monitor mismatch and missing workflow coverage.
- tests/torch/test_integration_workflow_torch.py: integration regression is the acceptance bar; must return to PASS.
How-To Map:
- RED phase commands: store `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_accumulation -vv` and `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_monitor_uses_val_loss_name -vv` outputs in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/red/pytest_workflows_accum_red.log` and `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/red/pytest_workflows_monitor_red.log`.
- Implementation: update `ptycho_torch/workflows/components.py` to read `model.val_loss_name` when validation present; adjust checkpoint filename template and EarlyStopping monitor accordingly; if helpful, add helper to fetch metric for reuse.
- GREEN phase commands: rerun both workflow selectors plus integration selector, saving to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/green/pytest_workflows_accum_green.log`, `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/green/pytest_workflows_monitor_green.log`, and `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/green/pytest_integration_green.log`. If full suite executed, emit consolidated `pytest_full_suite.log` at root of timestamp.
- Documentation/ledger: append concise summary + test matrix to `.../summary.md`, update `phase_e_execution_knobs/plan.md` EB2 rows to `[x]`, add Attempt #63 entry in `docs/fix_plan.md` referencing new artifacts.
Pitfalls To Avoid:
- Do not leave failing integration tests undocumented; capture exact stderr in blockers if unresolved.
- Maintain CONFIG-001 ordering—no edits that bypass `create_training_payload()` when touching workflows.
- Keep monitor default fallback behaviour (train_loss) for runs without validation; add regression test for this path if touched.
- Avoid mutating physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Ensure new tests stay pytest-native (no unittest mixins) and name fixtures clearly.
- Reuse existing helper warnings; do not duplicate logger logic while editing callbacks.
Pointers:
- ptycho_torch/workflows/components.py:700 — Lightning callback wiring.
- ptycho_torch/model.py:1048 — val_loss_name derivation.
- tests/torch/test_workflows_components.py — add workflow selectors.
- tests/torch/test_integration_workflow_torch.py:200 — integration workflow harness.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-23T091500Z/summary.md — regression context.
Next Up: EB2.C documentation polish (if small follow-ups remain) or EB3 logger decision once scheduler/accum path is green.
