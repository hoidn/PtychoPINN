# ADR-003 Phase C Execution Plan — PyTorch Execution Config Integration (2025-10-20T004233Z)

## Context
- Initiative: ADR-003-BACKEND-API (Standardize PyTorch backend API)
- Phase Goal: Promote the PyTorch execution configuration from placeholder annotations to a first-class dataclass, thread it through factory payloads, workflows, and CLI wrappers, and land workflow-level tests + documentation updates.
- Dependencies:
  - `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T002041Z/phase_b3_implementation/{plan.md,summary.md}` (factory payload baseline + GREEN evidence)
  - `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/{factory_design.md,override_matrix.md,open_questions.md}` (field inventory + override rules)
  - `specs/ptychodus_api_spec.md` §4, §6 (CONFIG-001 + backend execution contract)
  - `docs/workflows/pytorch.md` §§5–13 (workflow + CLI parity requirements)
  - `docs/findings.md` (CONFIG-001, POLICY-001, FORMAT-001 enforcement)
- Reporting Discipline: Store every artefact for this phase under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/<ISO8601>/` (e.g., `pytest_execution_config_red.log`, `design_delta.md`, `summary.md`). No loose files at repo root.
- Targeted Test Selectors:
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv`
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestTrainWithLightningGreen::test_execution_config_passes_to_trainer -vv`
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestTrainWithLightningGreen::test_execution_config_controls_determinism -vv`
  - Future CLI validations: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -k execution_config -vv` (author as part of Phase C4)

### Phase C1 — Canonicalise PyTorchExecutionConfig
Goal: Define `PyTorchExecutionConfig` in `ptycho/config/config.py`, export it, and cover it with TDD before wiring it into payloads.
Prereqs: Factory design §2.2 approval (Option A, canonical location) and override matrix §2 field list.
Exit Criteria: Dataclass defined with full field set + docstrings, exported via `__all__`, tests document defaults, SPEC/DOCS call out execution config availability.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1.A1 | Reconcile field list + defaults | [x] | ✅ Recorded in `phase_c_execution/design_delta.md` (2025-10-20T004233Z) with rationales for 22 execution fields sourced from `override_matrix.md` §5. |
| C1.A2 | Implement dataclass + exports | [x] | ✅ `PyTorchExecutionConfig` lives in `ptycho/config/config.py` with docstring + POLICY-001/CONFIG-001 references (commit 8159dcf1). |
| C1.A3 | Author RED tests | [x] | ✅ `tests/torch/test_execution_config.py` captured RED log `pytest_execution_config_red.log` before implementation (stored under `phase_c_execution/`). |
| C1.A4 | GREEN the tests + update specs/docs | [x] | ✅ Selector `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_execution_config.py -vv` GREEN (log: `pytest_execution_config_green.log`); spec §4.8/§6 and workflow guide §12 updated. |
| C1.A5 | Ledger & summary updates | [x] | ✅ `phase_c_execution/summary.md` documents C1 completion; fix ledger Attempt #11 references artefacts. |

### Phase C2 — Wire Execution Config Through Factories
Goal: Replace placeholder `Any` annotations with the new dataclass, ensure payloads materialise execution config objects, and extend tests for override precedence.
Prereqs: Phase C1 GREEN.
Exit Criteria: Factory payloads emit `PyTorchExecutionConfig` instances, override precedence covers execution knobs, tests assert runtime knob propagation.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C2.B1 | Update payload dataclasses | [x] | ✅ `TrainingPayload`/`InferencePayload` now type-hint `PyTorchExecutionConfig`; default instantiation path implemented (commit 447cecf8). |
| C2.B2 | Merge overrides into execution config | [x] | ✅ Factories merge execution knobs into `overrides_applied`; precedence documented in `phase_c2_factory_wiring/summary.md`. |
| C2.B3 | Extend factory tests | [x] | ✅ `TestExecutionConfigOverrides` added to `tests/torch/test_config_factory.py`; RED/GREEN logs stored under `phase_c2_factory_wiring/`. |
| C2.B4 | Document audit trail | [x] | ✅ `phase_c_execution/summary.md` updated with override precedence; fix ledger Attempt #12 references evidence. |

### Phase C3 — Workflow Integration (components.py)
Goal: Inject execution config into Lightning workflow helpers and ensure runtime knobs influence Trainer instantiation + dataloaders.
Prereqs: Phases C1–C2 complete.
Exit Criteria: `_train_with_lightning` accepts execution config, passes knobs to Lightning Trainer + DataLoader; inference workflow respects deterministic + num_workers; tests assert behaviour.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C3.C1 | Update `_train_with_lightning` signature | [x] | ✅ 2025-10-20 — Execution config parameter threaded per `phase_c3_workflow_integration/summary.md` (tasks C3.A1–C3.A3). RED/GREEN logs: `pytest_workflows_execution_{red,green}.log`. |
| C3.C2 | Integrate execution config in inference helpers | [x] | ✅ 2025-10-20 — DataLoader wiring complete (plan §C3.B) with batch size + num_workers overrides; evidence captured in `summary.md` + GREEN selector log. |
| C3.C3 | Extend workflow tests | [x] | ✅ 2025-10-20 — Added Trainer/inference pytest coverage (plan §C3.C). RED log stored prior to implementation; GREEN selector confirms deterministic checks. |
| C3.C4 | Update plan + summary | [x] | ✅ 2025-10-20 — `phase_c_execution/summary.md` updated, fix ledger Attempt #95 logged, artifacts consolidated under `phase_c3_workflow_integration/`. |

### Phase C4 — CLI + Documentation Finalisation
Goal: Collapse CLI wrappers onto factories with execution config exposure and align documentation.
Prereqs: Phases C1–C3 GREEN.
Exit Criteria: `ptycho_torch/train.py` / `inference.py` delegate to factories, expose execution knobs via CLI flags, CLI tests/dcos updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C4.D1 | Update CLI argument surfaces | [ ] | Map Lightning knobs to argparse options (learning_rate, accelerator, num_workers, deterministic). Reference `override_matrix.md` for naming. Ensure CLI defaults match dataclass defaults. |
| C4.D2 | Refactor CLI execution path | [ ] | Replace ad-hoc config construction with factory calls returning payloads, then call workflows with execution_config. Maintain CONFIG-001 ordering via factory. |
| C4.D3 | Add CLI regression tests | [ ] | Author pytest CLI harness (extend `tests/torch/test_cli_train_torch.py`) verifying new flags round-trip into execution config. Capture logs under plan path. |
| C4.D4 | Documentation & ledger close-out | [ ] | Update `docs/workflows/pytorch.md` CLI sections, refresh `specs/ptychodus_api_spec.md` CLI tables, and log Attempt in `docs/fix_plan.md`. Prepare close-out summary for Phase D handoff. |

## Verification Checklist (to close Phase C)
- [ ] All new/updated tests GREEN with logs stored under `phase_c_execution/`
- [ ] `Phase C` section of `implementation.md` updated with completion evidence
- [ ] `docs/fix_plan.md` reflects Attempts + artefact paths for each sub-phase
- [ ] `phase_c_execution/summary.md` captures decisions, open questions, and next-phase hooks (Phase D)
