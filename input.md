Summary: Drive EB2 scheduler/accumulation wiring from CLI to Lightning with full TDD artifacts.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Phase EB2 — Scheduler & Gradient Accumulation
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_scheduler_flag_roundtrip -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_accumulate_grad_batches_roundtrip -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py::TestExecutionConfigOverrides::test_scheduler_override_applied -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py::TestExecutionConfigOverrides::test_accum_steps_override_applied -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_accumulation -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/{eb2_plan.md,summary.md,red/,green/}
Do Now:
1. EB2.A1-EB2.A3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/eb2_plan.md — extend TestExecutionConfigCLI with scheduler/accumulation RED cases, run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_scheduler_flag_roundtrip -vv` and `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_accumulate_grad_batches_roundtrip -vv` expecting failure, archive logs to red/; tests: targeted.
2. EB2.A+EB2.B (wiring) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/eb2_plan.md — implement CLI/helper/factory/trainer overrides and rerun all mapped selectors to GREEN, storing outputs in green/; tests: mapped.
3. EB2.C1-C3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/eb2_plan.md — redline spec/workflow tables, update plan states + docs/fix_plan.md with Attempt #61 follow-up, produce summary/spec_redline artifacts; tests: none.
If Blocked: Log blocker + evidence in plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/blockers.md and leave EB2 rows `[P]`.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:272 — Optimization knobs section still treats scheduler/accum as pending exposure.
- ptycho_torch/train.py:404 — CLI missing scheduler/accum flags keeps Lightning cadence hard-coded.
- ptycho_torch/config_factory.py:200 — Factory audit trail ignores scheduler/accum overrides today.
- ptycho_torch/workflows/components.py:720 — Trainer already consumes accum_steps; need provenance via CLI overrides for parity.
- tests/torch/test_cli_train_torch.py — TDD harness must expand before implementation to guard regressions.
How-To Map:
- RED phase: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_scheduler_flag_roundtrip -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/red/pytest_cli_scheduler_red.log` (expect fail) then `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_accumulate_grad_batches_roundtrip -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/red/pytest_cli_accum_red.log`.
- After wiring, rerun all mapped selectors with `CUDA_VISIBLE_DEVICES=""` and store outputs under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/green/pytest_*_green.log` (keep filenames consistent with eb2_plan.md guidance).
- For factory coverage include `pytest tests/torch/test_config_factory.py::TestExecutionConfigOverrides::test_scheduler_override_applied -vv` and accum variant; for workflow include `pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_accumulation -vv`.
- Post-tests, update `specs/ptychodus_api_spec.md` (§4.9, §7.1) and `docs/workflows/pytorch.md` §12; capture deltas in `spec_redline.md` and final narrative in `summary.md` within the timestamped directory.
- Finish by marking EB2 rows `[x]` in `phase_e_execution_knobs/plan.md`, updating `docs/fix_plan.md`, and confirming artifacts logged.
Pitfalls To Avoid:
- Skipping RED evidence before implementation.
- Changing default scheduler/accum semantics (must remain `'Default'` / `1`).
- Breaking `--device` backward-compat warnings while touching CLI.
- Touching physics core modules (`ptycho/model.py`) beyond wiring validation.
- Leaving logs or scratch files outside the timestamped reports directory.
- Forgetting to refresh `override_matrix.md` if precedence logic changes.
- Mixing unittest style in new pytest tests.
Pointers:
- ptycho_torch/train.py:404
- ptycho_torch/cli/shared.py:60
- ptycho_torch/config_factory.py:248
- ptycho_torch/workflows/components.py:720
- ptycho_torch/model.py:1270
- specs/ptychodus_api_spec.md:272
- docs/workflows/pytorch.md:320
Next Up: EB2 follow-through documentation polish or EB3 logger decision prep once scheduler/accumulation paths are green.
