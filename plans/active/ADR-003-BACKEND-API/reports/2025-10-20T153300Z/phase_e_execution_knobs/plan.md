# Phase E Execution Knob Hardening Blueprint

## Context
- Initiative: ADR-003-BACKEND-API — PyTorch backend standardisation
- Phase Goal: Deliver Phase E.B tasks (execution knob hardening) with TDD, ensuring CLI, factories, and Lightning workflows surface the remaining PyTorchExecutionConfig controls without regressing CONFIG-001/CONFIG-002 guarantees.
- Dependencies:
  - Governance dossier (`reports/2025-10-20T133500Z/phase_e_governance/plan.md`)
  - Spec updates (`specs/ptychodus_api_spec.md` §4.9, §7.2 checkpoint table)
  - CLI thin-wrapper plan (`reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md`)
  - Execution config design docs (`reports/2025-10-19T232336Z/phase_b_factories/{factory_design.md,override_matrix.md}`)
- Storage: Use `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/<timestamp>/` for all Phase E.B artifacts (evidence, logs, blockers).

## Shared Guardrails
- Maintain CONFIG-001 bridge ordering (`update_legacy_dict` runs before data/model construction).
- Respect CONFIG-002: PyTorchExecutionConfig never populates `params.cfg`; changes remain runtime-scoped.
- TDD cadence: author RED selectors first, capture logs under `reports/.../phase_e_execution_knobs/<TS>/red/`, then GREEN evidence under `.../green/`.
- All new tests must use native pytest style; keep selectors CPU-only (`CUDA_VISIBLE_DEVICES=""`).
- Update spec/workflow docs/fix_plan in the same loop when behaviour changes (checkpoint defaults, CLI tables, findings).

### Phase EB1 — Checkpoint & Early-Stop Controls
Goal: Surface Lightning checkpoint/early-stop knobs via CLI + factories, and wire callbacks inside `_train_with_lightning`.
Prereqs: Phase E.A complete; review `spec_redline.md` + CLI inventory backlog.
Exit Criteria: CLI accepts flags, factories merge overrides, Lightning trainer attaches configured `ModelCheckpoint` + `EarlyStopping`, tests/documentation updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| EB1.A | Schema + spec audit | [P] | `checkpoint_mode` landed in `ptycho/config/config.py` but spec §4.9/§7 tables still missing the field; workflow guide also not refreshed. Complete doc updates and capture delta summary under the next reports timestamp. |
| EB1.B | CLI flag parsing | [x] | `ptycho_torch/train.py` now exposes all checkpoint flags (commit 496a8ce3). Confirm defaults stay backward compatible before closing out. |
| EB1.C | Shared helper + factory wiring | [x] | `build_execution_config_from_args` + config factory propagate checkpoint knobs; overrides appear in `TestExecutionConfigOverrides` (GREEN). No further action unless follow-up tests demand adjustments. |
| EB1.D | Lightning callback integration | [P] | `_train_with_lightning` instantiates callbacks when `enable_checkpointing=True`, but workflow tests still fail because they patch `lightning.Trainer` instead of `lightning.pytorch.Trainer`. Fix the selector scaffolding so callbacks are observable and re-run targeted tests. |
| EB1.E | TDD coverage & runs | [P] | CLI/config factory selectors are green, yet `tests/torch/test_workflows_components.py::TestLightningCheckpointCallbacks::{test_model_checkpoint_callback_configured,test_early_stopping_callback_configured}` remain RED (see `.../2025-10-20T154207Z/green/workflows_checkpoint_callbacks_tests.log`). Update mocks + rerun commands listed in How-To. |
| EB1.F | Documentation & ledger sync | [ ] | Pending: spec/workflow guide updates, new Attempt #58 entry in `docs/fix_plan.md`, and loop summary under the fresh timestamp directory. |

### Phase EB2 — Scheduler & Gradient Accumulation
Goal: Expose `scheduler` and `accum_steps` knobs end-to-end.
Prereqs: EB1 complete (callbacks stable); gather scheduler rationale from `factory_design.md` §4.3.
Exit Criteria: CLI flags wired, trainer applies scheduler/accumulation, tests cover CLI → factory → trainer propagation.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| EB2.A | CLI + helper extension | [ ] | Add `--scheduler` (choices: Default/Exponential/MultiStage) and `--accumulate-grad-batches` flags; map via `build_execution_config_from_args`. |
| EB2.B | Factory + trainer wiring | [ ] | Thread new fields through `config_factory` + `_train_with_lightning`, integrating scheduler instantiation (reuse `ptycho_torch/schedulers.py`). |
| EB2.C | TDD & docs | [ ] | Extend CLI/config_factory/workflow tests; update spec/workflow docs. |

### Phase EB3 — Logger Backend Decision
Goal: Resolve MLflow/TensorBoard governance and implement or deprecate flag.
Prereqs: EB1–EB2 complete; consult `open_questions.md` logger section.
Exit Criteria: Decision log + implementation/tests capturing logger behaviour.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| EB3.A | Decision record | [ ] | Draft `logger_decision.md` summarising options (MLflow autolog vs TensorBoard vs disable). |
| EB3.B | Implementation & tests | [ ] | Implement chosen logger wiring or deprecation warnings; add tests covering CLI messaging. |
| EB3.C | Docs update | [ ] | Update workflow guide/logger documentation + findings if policy-level change. |

### Phase EB4 — Runtime Smoke Extensions
Goal: Capture deterministic smoke logs for expanded knob combinations.
Prereqs: EB1–EB3 green.
Exit Criteria: Smoke logs stored, plan checklist updated, delta documented.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| EB4.A | CLI smoke scripts | [ ] | Run training CLI with `--accelerator auto`, `--checkpoint-save-top-k 2`, `--early-stop-patience 5` on minimal dataset; store logs under `runtime_smoke/`. |
| EB4.B | Ledger + plan updates | [ ] | Document smoke results in `summary.md`, update `docs/fix_plan.md` attempt and `implementation.md` Phase E.B states. |

## Artifact Checklist
- RED logs → `reports/2025-10-20T153300Z/phase_e_execution_knobs/<TS>/red/`
- GREEN logs → `.../green/`
- Summaries/blockers → `.../<TS>/{summary.md,blockers.md}`
- Spec/doc diffs captured via `spec_redline.md` appendices when fields change.

## References
- specs/ptychodus_api_spec.md §4.9, §7.2
- docs/workflows/pytorch.md §12 (CLI tables)
- ptycho/config/config.py:178-310 (PyTorchExecutionConfig definition)
- ptycho_torch/cli/shared.py:1-160 (execution config builder)
- ptycho_torch/config_factory.py:200-360 (training payload)
- ptycho_torch/workflows/components.py:640-720 (`_train_with_lightning`)
- tests/torch/test_cli_train_torch.py, tests/torch/test_config_factory.py, tests/torch/test_workflows_components.py (existing execution config coverage)
