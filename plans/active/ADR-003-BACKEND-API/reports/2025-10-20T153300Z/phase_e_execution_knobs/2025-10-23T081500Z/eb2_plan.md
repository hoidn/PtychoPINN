# Phase EB2 — Scheduler & Gradient Accumulation Implementation Plan

## Context
- Initiative: ADR-003-BACKEND-API (PyTorch backend API standardization)
- Phase Goal: Expose Lightning scheduler selection and gradient accumulation knobs end-to-end so CLI callers can control training cadence without editing source.
- Dependencies: Phase EB1 completed (checkpoint/early-stop knobs), factories + execution config merged (Phase C), CLI thin wrappers green (Phase D).
- References:
  - `ptycho/config/config.py:210-260` — `PyTorchExecutionConfig` fields (`scheduler`, `accum_steps`), validation rules.
  - `ptycho_torch/train.py:404-615` — Training CLI argparse schema (new flags land here).
  - `ptycho_torch/cli/shared.py:1-170` — `build_execution_config_from_args()` helper that maps CLI args → `PyTorchExecutionConfig`.
  - `ptycho_torch/config_factory.py:200-360` — `create_training_payload()` override precedence + audit trail.
  - `ptycho_torch/workflows/components.py:700-780` — Lightning Trainer construction (`accumulate_grad_batches` hook).
  - `ptycho_torch/model.py:990-1310` — `PtychoPINN_Lightning.configure_optimizers()` scheduler plumbing (expects `training_config.scheduler`).
  - `tests/torch/test_cli_train_torch.py` — Existing CLI execution config round-trip tests to extend for new flags.
  - `tests/torch/test_config_factory.py` — Execution config override coverage (add scheduler/accum cases).
  - `tests/torch/test_workflows_components.py` — Lightning trainer callback tests (add assertions for accumulation + scheduler handshake if feasible).
  - `specs/ptychodus_api_spec.md` §§4.9, 7.1 — Normative execution config contract + CLI table (must redline).
  - `docs/workflows/pytorch.md` §12 — Training execution flag table (mirror spec updates).

### Phase EB2.A — CLI & Helper Exposure
Goal: Accept scheduler / accumulation overrides via CLI and thread them through helper layer.
Prereqs: EB1 complete; CLI thin wrapper + shared helpers available.
Exit Criteria: CLI recognizes `--scheduler` & `--accumulate-grad-batches`, help text documents defaults, and `build_execution_config_from_args()` populates the new fields (with validation).

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| EB2.A1 | Extend `ptycho_torch/train.py` argparse schema with `--scheduler` (choices: `Default`, `Exponential`, `MultiStage`, `Adaptive`) and `--accumulate-grad-batches` (int, default `1`). Include backward-compatible help text sourced from `factory_design.md` §4.3 and mark CLI examples in docstrings. | [ ] | Update CLI immediately after learning-rate flag to keep grouping. Default values must match `PyTorchExecutionConfig.scheduler` and `.accum_steps`. Document CLI/SPEC defaults match and warn that `--accumulate-grad-batches` >1 implies larger effective batch sizes. |
| EB2.A2 | Update `build_execution_config_from_args()` (training branch) to forward `scheduler` and `accum_steps` into the dataclass. Emit validation errors via dataclass (`ValueError`) and update quiet-mode logic so progress bar behaviour unchanged. | [ ] | Ensure helper pulls defaults from `getattr(args, 'scheduler', 'Default')` and `getattr(args, 'accumulate_grad_batches', 1)`. While populating config, also update audit trail decisions (to be recorded in EB2.B). Add TODO note for inference mode (scheduler not supported there). |
| EB2.A3 | Expand CLI tests to cover new flags before implementation (TDD RED). Add parametrised cases in `tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI` to assert scheduler + accumulation propagate to factory execution_config object. | [ ] | Add new tests (initially RED) referencing authoritative selector `pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_scheduler_flag_roundtrip -vv`. Capture RED log in `reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/red/pytest_cli_scheduler_red.log`. |

### Phase EB2.B — Factory & Trainer Wiring
Goal: Thread scheduler/accumulation overrides through config factory and Lightning execution so runtime behaviour matches CLI expectations.
Prereqs: EB2.A flags + helper populate `PyTorchExecutionConfig`.
Exit Criteria: `create_training_payload()` records overrides, `PTTrainingConfig` receives scheduler/accum steps, Lightning trainer & module use the new values, and TDD coverage shows overrides applied.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| EB2.B1 | Update `create_training_payload()` to: (a) include `scheduler` & `accum_steps` in `overrides` ingestion, (b) copy values into `PTTrainingConfig` (for legacy module expectations), and (c) write them into `overrides_applied` audit map. | [ ] | Honor precedence: explicit CLI overrides > defaults. When CLI omitted values, maintain existing defaults (`'Default'`, `1`). After instantiation, set `pt_training_config.scheduler` / `.accum_steps` from either `overrides` or `execution_config` to keep Lightning module in sync. |
| EB2.B2 | Ensure `PyTorchExecutionConfig` returned by factory reflects CLI override even when user passes explicit execution_config (merge precedence). Update `build_execution_config_from_args()` tests accordingly. | [ ] | Add assertions in `tests/torch/test_config_factory.py::TestExecutionConfigOverrides` verifying `execution_config.scheduler` & `.accum_steps` propagate. Store GREEN proof at `.../green/pytest_factory_scheduler_green.log`. |
| EB2.B3 | Validate Lightning stack respects overrides: `_train_with_lightning` should pass `accumulate_grad_batches` from execution config, and `PtychoPINN_Lightning.configure_optimizers()` should read scheduler from training config (ensure CLI override reached there). Add workflow-level test capturing both behaviours. | [ ] | Extend `tests/torch/test_workflows_components.py::TestLightningExecutionConfig` (new class) or reuse checkpoint suite to assert trainer kwargs contain `accumulate_grad_batches=<override>` and module scheduler string matches. Use `tmp_path` fixtures for dataset and stub scheduler results. |

### Phase EB2.C — Documentation & Spec Alignment
Goal: Synchronize normative docs and ledger after implementation.
Prereqs: EB2.A–B tests GREEN.
Exit Criteria: Spec + workflow guide tables document new flags, plan + ledger updated, artifacts archived under timestamp.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| EB2.C1 | Update `specs/ptychodus_api_spec.md` §4.9 optimization section + §7.1 CLI table with scheduler/accum rows (include defaults, validation, interactions). | [ ] | Mirror CLI help text; note scheduler choices map to Lightning helpers. Mention that accumulation multiplies effective batch size and interacts with Poisson loss stability. |
| EB2.C2 | Update `docs/workflows/pytorch.md` §12 training table + narrative (link to spec, caution on accumulation vs GPU memory). | [ ] | Ensure doc table order matches spec to avoid drift. |
| EB2.C3 | Mark EB2 rows complete in `phase_e_execution_knobs/plan.md`, append Attempt entry to `docs/fix_plan.md`, and capture `summary.md` + `spec_redline.md` under this timestamp directory. | [ ] | Summary must list selectors executed with pass counts. Reference CLI + factory + workflow logs. |

## Artifact Routing
- RED evidence → `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/red/`
- GREEN evidence → `.../green/`
- Summaries/spec diffs → `.../{summary.md,spec_redline.md}`
- Blockers → `.../blockers.md`

## Test Selectors (authoritative)
- CLI execution config tests: `pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_scheduler_flag_roundtrip -vv`
- CLI accumulation test: `pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_accumulate_grad_batches_roundtrip -vv`
- Factory overrides: `pytest tests/torch/test_config_factory.py::TestExecutionConfigOverrides::test_scheduler_override_applied -vv`
- Workflow trainer wiring: `pytest tests/torch/test_workflows_components.py::TestLightningExecutionConfig::test_trainer_receives_accumulation -vv`
- Optional smoke: `pytest tests/torch/test_model_manager.py::test_load_tensorflow_checkpoint_with_pytorch_backend -vv` (regression guard to ensure scheduler changes don’t break persistence)

## Risks & Notes
- Scheduler values must remain case-sensitive to match existing `PtychoPINN_Lightning` logic (`'Default'`, `'MultiStage'`, `'Adaptive'`). Document conversions if enforcing lowercase.
- Gradient accumulation interacts with Lightning + manual accumulation in `PtychoPINN_Lightning.training_step`; ensure overrides keep both paths in sync (training_step divides loss by `self.accum_steps`).
- Update `override_matrix.md` if new precedence rules discovered while wiring scheduler overrides.
- Keep CPU-friendly defaults for CI (`scheduler='Default'`, `accum_steps=1`). Validate large accumulation via targeted tests only if deterministic.

