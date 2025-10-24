Summary: Enable Lightning logger selection via CLI/factory/workflow so PyTorch training captures metrics (Phase EB3.B).
Mode: TDD
Focus: [ADR-003-BACKEND-API] Phase EB3 — Logger backend decision
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_cli_train_torch.py -k logger -vv; pytest tests/torch/test_config_factory.py -k logger -vv; pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_logger -vv; pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-23T110500Z/phase_b_logger_impl/2025-10-23T130000Z/{red/,green/,summary.md}
Do Now:
- [ADR-003-BACKEND-API] EB3.B1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — author logger RED tests (CLI helpers, factory overrides, trainer wiring) and capture failing selectors under red/; tests: see How-To Map (RED selectors).
- [ADR-003-BACKEND-API] EB3.B2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — implement CSV default + TensorBoard/MLflow options, deprecate --disable_mlflow, and thread logger through CLI/shared.py, config_factory, workflows/components; tests: none (implementation only).
- [ADR-003-BACKEND-API] EB3.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — rerun targeted selectors + integration workflow on CPU, archive GREEN logs plus summary.md; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -k logger -vv && CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k logger -vv && CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_logger -vv && CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv.
If Blocked: Capture blockers in summary.md, leave plan rows `[P]`, and log evidence (stack traces, config notes) in the artifact hub before exiting.
Priorities & Rationale:
- decision/approved.md — supervisor authorized CSV default, TensorBoard option, mlflow deprecation, and MLFlowLogger follow-up; implementation must reflect this contract.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-23T110500Z/plan.md (B1–B3) — authoritative checklist for Phase EB3.B.
- specs/ptychodus_api_spec.md:275-320 — execution config fields and CLI mapping your changes must satisfy.
- docs/workflows/pytorch.md:320-370 — workflow documentation requires logger guidance after implementation; keep behaviour aligned.
- docs/findings.md#policy-001 — no new mandatory deps; ensure optional logger choices honour policy.
How-To Map:
- RED phase: Write pytest cases in `tests/torch/test_cli_train_torch.py` (e.g., `TestExecutionConfigCLI` with default CSV + tensorboard/mlflow selections and `--disable_mlflow` DeprecationWarning), `tests/torch/test_config_factory.py` (`TestExecutionConfigLogger` verifying factory output / optional deps), and `tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_logger`. Run each with `CUDA_VISIBLE_DEVICES=""` and stash logs under `red/pytest_cli_logger_red.log`, `red/pytest_factory_logger_red.log`, `red/pytest_workflows_logger_red.log`.
- Implementation: Update argparse in `ptycho_torch/train.py` to add `--logger`/`--logger-backend`, wire through `cli/shared.py` helper to populate execution config dict (deprecate `--disable_mlflow` via `warnings.warn`). Extend `ptycho_torch/config_factory.py` to build logger instances (CSV default, tensorboard optional, mlflow optional with ImportError guidance) and return them in payload. Replace `logger=False` in `_train_with_lightning` (workflows/components.py) with the constructed logger, keeping CONFIG-001 ordering.
- GREEN validation: Re-run RED selectors after implementation (store outputs in `green/`). Finish with `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv` and, if time allows, `pytest tests/torch/test_cli_shared.py -k logger -vv` to cover helper docstrings. Summarize results (pass/fail counts, warning expectations) in `summary.md`.
- Artifacts: Place new or updated docs/tests references in summary, include explicit warning text for `--disable_mlflow`, and list generated files (e.g., CSV log path) if applicable.
Pitfalls To Avoid:
- Do not instantiate Lightning loggers before calling `update_legacy_dict`; maintain CONFIG-001 ordering.
- Avoid importing tensorboard/mlflow unconditionally—guard optional deps and raise actionable errors/warnings.
- Keep TDD discipline: capture RED logs before implementing fixes.
- Ensure warnings use `warnings.warn(..., DeprecationWarning)` so tests can assert behaviour.
- No hard-coded filesystem paths; respect execution config `output_dir`.
- Do not delete existing mlflow autologging yet—only deprecate flag per decision.
- Keep pytest additions purely pytest-style (no unittest mixins).
- Limit artifact size; don’t commit TensorBoard event files to repo (archive path only).
- Maintain ASCII output in docs/logs; avoid non-ASCII characters in warnings/tests.
- Capture RNG seeds/config overrides in summary if tests rely on randomness.
Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-23T110500Z/plan.md (Phase B table)
- plans/active/ADR-003-BACKEND-API/reports/2025-10-23T110500Z/decision/approved.md
- specs/ptychodus_api_spec.md:275
- docs/workflows/pytorch.md:320
- ptycho_torch/train.py:460
- ptycho_torch/cli/shared.py:90
- ptycho_torch/config_factory.py:210
- ptycho_torch/workflows/components.py:730
Next Up: Phase EB3.C documentation sync (`plan.md` C1–C4) once logger wiring and tests are green.
