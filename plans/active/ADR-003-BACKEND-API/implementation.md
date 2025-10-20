# ADR-003 PyTorch Backend API Implementation Plan

## Context
- Initiative: ADR-003-BACKEND-API (Standardize PyTorch backend API)
- Phase Goal: Implement the ADR-003 two-layer architecture while preserving parity with TensorFlow workflows and CLI behaviour.
- Dependencies: `specs/ptychodus_api_spec.md` §4 (reconstructor contract), `docs/workflows/pytorch.md`, config bridge docs (`plans/active/INTEGRATE-PYTORCH-001/reports/*`), existing CLI scripts (`ptycho_torch/train.py`, `ptycho_torch/inference.py`).

### Phase A — Architecture Carve-Out
Goal: Confirm current implementation surfaces, inventory execution knobs, and align plan with existing initiatives (INTEGRATE-PYTORCH-001).
Prereqs: Review ADR-003 rationale, inspect CLI scripts + `config_bridge` usage, confirm PyTorch execution settings inventory.
Exit Criteria: Shared inventory of CLI/programmatic parameters, execution config requirements, and bridge override needs captured in report.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Inventory CLI flags and programmatic entry points | [x] | ✅ 2025-10-19 — Captured 19 add_argument calls to `logs/a1_cli_flags.txt`. Mapped 9 training flags + 10 inference flags to config fields with file:line citations. Identified 5 critical semantic mismatches (epoch naming, activation defaults, neighbor count, probe size, n_groups terminology) + 11 feature gaps (6 PyTorch, 5 TF). Artifact: `cli_inventory.md` (21KB). |
| A2 | Catalogue backend-specific execution knobs | [x] | ✅ 2025-10-19 — Cataloged 54 unique parameters across 11 categories (Lightning, MLflow, distributed, logging, scheduler, checkpointing, inference). Recommended 35 knobs for `PyTorchExecutionConfig` dataclass. Identified 9 hardcoded values requiring CLI exposure (learning_rate, accelerator, num_workers, early_stop_patience, scheduler_type). Artifact: `execution_knobs.md` (265 lines). |
| A3 | Confirm overlaps with existing plans | [x] | ✅ 2025-10-19 — Audited 15 topics against INTEGRATE-PYTORCH-001 + TEST-PYTORCH-001. Confirmed 7 complete (CLI, config bridge, Lightning, MLflow, persistence, fixtures, parity). Identified 2 critical gaps for ADR-003 ownership (factory patterns, PyTorchExecutionConfig). Flagged 1 governance gap (ADR-003.md missing). No blockers for Phase B. Artifact: `overlap_notes.md` (17KB), `summary.md` (compact summary). |

### Phase B — Configuration Factories
Goal: Centralize canonical `TF*Config` construction for PyTorch backend via shared factories.
Prereqs: Phase A inventory complete; consensus on execution config fields.
Exit Criteria: Factory modules authored with TDD coverage and linked to config bridge tests.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Draft factory design doc | [x] | ✅ 2025-10-19 — Delivered 4 comprehensive design documents (1,629 lines total): `factory_design.md` (420 lines, factory architecture + integration strategy), `override_matrix.md` (584 lines, 80+ fields mapped with precedence rules), `open_questions.md` (625 lines, governance decisions + spec impacts), `summary.md` (exit criteria validation). Mapped all PyTorch CLI flags to config fields with file:line citations. Identified 16 missing CLI flags, 4 naming divergences, 2 critical default mismatches (nphotons, K). Defined 5-level override precedence, CONFIG-001 compliance checkpoints, and factory testing strategy (RED/GREEN phases). **Supervisor Decision (2025-10-19T234458Z):** Option A approved — implement `PyTorchExecutionConfig` in `ptycho/config/config.py` with execution-only docstring. Phase B2 now unblocked. Artifacts: `reports/2025-10-19T232336Z/phase_b_factories/`. |
| B2 | Implement training/inference factory functions | [x] | ✅ 2025-10-19–20 — Factory skeleton + RED coverage landed, then converted from FALSE GREEN → TRUE RED. Skeleton + initial log captured under `reports/2025-10-19T234600Z/phase_b2_skeleton/`; follow-up loop removed all `pytest.raises` guards, reactivated 56 assertions, and stored failing selector evidence in `reports/2025-10-20T000736Z/phase_b2_redfix/{summary.md, pytest_factory_redfix.log}` (`CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv` → 19 FAILED on NotImplementedError). Plan ready to advance to B3 implementation. |
| B3 | Deliver factory implementation + parity updates | [x] | ✅ 2025-10-20 — Factories implemented per `reports/2025-10-20T002041Z/phase_b3_implementation/{plan.md,summary.md}`. All 19 factory tests GREEN (`pytest_factory_green.log`), full regression clean (262 passed). CONFIG-001 override path validated; validation tests updated to expect ValueError/FileNotFoundError. Phase C handoff pending (see new plan in `reports/2025-10-20T004233Z/phase_c_execution/`). |

### Phase C — Core Workflow Refactor
Goal: Embed canonical + execution config pattern into `ptycho_torch/workflows/components.py` and downstream helpers.
Prereqs: Factories merged; execution config dataclass finalised.
Exit Criteria: Workflows accept dual configs, Lightning orchestration centralized, legacy params handshake verified.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Introduce `PyTorchExecutionConfig` | [ ] | Follow `reports/2025-10-20T004233Z/phase_c_execution/plan.md` §C1: reconcile field list, add dataclass + exports in `ptycho/config/config.py`, land RED→GREEN tests (`tests/torch/test_execution_config.py`), and update spec/workflow docs with execution-config contract. |
| C2 | Refactor training/inference workflows | [ ] | Execute plan §C2: update factory payloads to emit real `PyTorchExecutionConfig` instances, merge overrides, and extend factory tests for execution-knob precedence (`pytest ... -k ExecutionConfig`). Document decisions in `phase_c_execution/summary.md`. |
| C3 | Update helper pathways | [ ] | Execute plan §C3: thread execution config through `_train_with_lightning` and inference helpers, assert Lightning Trainer kwargs via `tests/torch/test_workflows_components.py`. Capture RED/GREEN logs under `phase_c_execution/`. |
| C4 | Add workflow-level tests | [ ] | Execute plan §C4: expose knobs via CLI flags, refactor CLI wrappers onto factories, add CLI regression tests, and refresh docs/specs. Store evidence + Attempt entries per plan checklist. |

### Phase D — CLI Thin Wrappers
Goal: Reduce CLI scripts to argument parsing + factory/execution config assembly.
Prereqs: Phase C workflows stable and tested.
Exit Criteria: CLI scripts delegate entirely to workflows; backwards-compat CLI behaviour preserved.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Update training CLI | [ ] | Refactor `ptycho_torch/train.py` to call factories + workflows; revalidate CLI tests/documentation. |
| D2 | Update inference CLI | [ ] | Same for `ptycho_torch/inference.py`; include bundle loading path. |
| D3 | Smoke tests & docs | [ ] | Run targeted CLI commands + update `docs/workflows/pytorch.md` usage examples. |

### Phase E — Legacy API Deprecation & Governance
Goal: Retire or wrap `ptycho_torch/api/`, update documentation, and capture governance sign-off.
Prereqs: Phases B–D complete with passing tests.
Exit Criteria: `api/` surface deprecated, ADR status set to Accepted, ledger updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E1 | Deprecate `ptycho_torch/api/` | [ ] | Add deprecation notice or refactor to delegate to new workflows; log decision. |
| E2 | Update documentation & ADR | [ ] | Publish ADR acceptance summary, update `docs/workflows/pytorch.md`, `specs/ptychodus_api_spec.md` if needed. |
| E3 | Final ledger & plan updates | [ ] | Mark plan complete, update docs/fix_plan attempts, ensure governance artefacts stored. |

## Reporting Discipline
- All artefacts stored under `plans/active/ADR-003-BACKEND-API/reports/<ISO8601>/` with descriptive filenames.
- Tests follow native pytest style; full-suite runs coordinated with existing integration plans.
- Reference this plan from docs/fix_plan.md entry `[ADR-003-BACKEND-API]`.
