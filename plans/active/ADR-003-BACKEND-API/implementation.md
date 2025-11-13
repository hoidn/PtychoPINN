# ADR-003 PyTorch Backend API Implementation Plan

<plan_update version="1.0">
  <trigger>GPU execution is now the default policy for PyTorch backend work; plan narratives referencing CPU-only safety need to be updated.</trigger>
  <focus_id>ADR-003-BACKEND-API</focus_id>
  <documents_read>docs/index.md, docs/workflows/pytorch.md, plans/active/ADR-003-BACKEND-API/implementation.md</documents_read>
  <current_plan_path>plans/active/ADR-003-BACKEND-API/implementation.md</current_plan_path>
  <proposed_changes>Update command references to pin CUDA devices and annotate sections that previously called out CPU-safe defaults with guidance to migrate toward GPU-first execution.</proposed_changes>
  <impacts>Factories and workflows must ultimately adopt CUDA defaults; interim plan text flags the requirement so future phases prioritize the change.</impacts>
  <ledger_updates>Reference this plan update in docs/fix_plan.md when logging the next ADR-003 attempt.</ledger_updates>
  <status>approved</status>
</plan_update>

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
| B2 | Implement training/inference factory functions | [x] | ✅ 2025-10-19–20 — Factory skeleton + RED coverage landed, then converted from FALSE GREEN → TRUE RED. Skeleton + initial log captured under `reports/2025-10-19T234600Z/phase_b2_skeleton/`; follow-up loop removed all `pytest.raises` guards, reactivated 56 assertions, and stored failing selector evidence in `reports/2025-10-20T000736Z/phase_b2_redfix/{summary.md, pytest_factory_redfix.log}` (`CUDA_VISIBLE_DEVICES="0" pytest tests/torch/test_config_factory.py -vv` → 19 FAILED on NotImplementedError). Plan ready to advance to B3 implementation. |
| B3 | Deliver factory implementation + parity updates | [x] | ✅ 2025-10-20 — Factories implemented per `reports/2025-10-20T002041Z/phase_b3_implementation/{plan.md,summary.md}`. All 19 factory tests GREEN (`pytest_factory_green.log`), full regression clean (262 passed). CONFIG-001 override path validated; validation tests updated to expect ValueError/FileNotFoundError. Phase C handoff pending (see new plan in `reports/2025-10-20T004233Z/phase_c_execution/`). |

### Phase C — Core Workflow Refactor
Goal: Embed canonical + execution config pattern into `ptycho_torch/workflows/components.py` and downstream helpers.
Prereqs: Factories merged; execution config dataclass finalised.
Exit Criteria: Workflows accept dual configs, Lightning orchestration centralized, legacy params handshake verified.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Introduce `PyTorchExecutionConfig` | [x] | ✅ 2025-10-20 — Dataclass landed in `ptycho/config/config.py` with `__all__` export and POLICY-001/CONFIG-001 docstring. Field reconciliation recorded in `reports/2025-10-20T004233Z/phase_c_execution/design_delta.md`; RED→GREEN evidence captured (`pytest_execution_config_red.log`, `pytest_execution_config_green.log`). Spec §4.8/§6 and `docs/workflows/pytorch.md` §12 updated to describe execution config contract. |
| C2 | Wire execution config through factories | [x] | ✅ 2025-10-20 — Factory payloads now emit `PyTorchExecutionConfig` instances (type hints updated from `Any` to concrete type). Default instantiation logic added (historic CPU-safe defaults: accelerator='cpu', deterministic=True, num_workers=0). Override merging implemented with audit trail recording applied knobs in `overrides_applied`. Extended `tests/torch/test_config_factory.py` with `TestExecutionConfigOverrides` (6 new tests, RED→GREEN cycle captured). Full test suite GREEN (268 passed). **Follow-up:** migrate the dataclass defaults toward CUDA once GPU runtime profiling is complete. Evidence: `reports/2025-10-20T010900Z/phase_c2_factory_wiring/{summary.md, pytest_factory_execution_*.log, pytest_full_suite.log}`. |
| C3 | Update helper pathways | [x] | ✅ 2025-10-20 — Execution config threaded through workflows per `reports/2025-10-20T025643Z/phase_c3_workflow_integration/summary.md`; logs stored at `reports/2025-10-20T025643Z/phase_c3_workflow_integration/pytest_workflows_execution_{red,green}.log`, full suite log archived, `train_debug.log` relocated. |
| C4 | Add workflow-level tests | [x] | ✅ Phase C4 close-out (2025-10-20T123500Z) — CLI flag exposure, refactors, validation, and documentation complete. Evidence consolidated under `reports/2025-10-20T033100Z/phase_c4_cli_integration/` with blocker resolutions (`phase_c4d_blockers/plan.md`), GREEN logs (`phase_c4d_at_parallel/summary.md`), docs update summary (`2025-10-20T120500Z`), and comprehensive close-out (`2025-10-20T123500Z/phase_c4f_closeout/summary.md`). Deferred knobs tracked for Phase D handoff. |

### Phase D — CLI Thin Wrappers
Goal: Reduce CLI scripts to argument parsing + factory/execution config assembly.
Prereqs: Phase C workflows stable and tested.
Exit Criteria: CLI scripts delegate entirely to workflows; backwards-compat CLI behaviour preserved.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Update training CLI | [x] | ✅ 2025-10-20 — Training CLI thin wrapper complete. Phase B.B3 implementation (helpers, validation, refactor) landed with GREEN logs in `reports/2025-10-20T111500Z/phase_d_cli_wrappers_training_impl/`. Phase B.B4 documentation refresh + hygiene complete: `docs/workflows/pytorch.md` updated with `--quiet`, `--device` deprecation, helper-based flow; `tests/torch/test_cli_shared.py` docstring revised to GREEN status; `train_debug.log` relocated to artifact hub. Summary: `reports/2025-10-20T112811Z/phase_d_cli_wrappers_training_docs/summary.md`. |
| D2 | Update inference CLI | [x] | ✅ 2025-10-20 — `phase_d_cli_wrappers/plan.md` C1–C4 finished: blueprint, RED tests, refactor, doc refresh. Evidence: `reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup/summary.md`, CLI inference selector GREEN, integration selector GREEN, doc updates captured. |
| D3 | Smoke tests & docs | [x] | ✅ 2025-10-20 — Deterministic smoke runs captured (`reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/`), handoff summary authored, hygiene verified, plan + ledger now updated via supervisor Attempt #54. Phase D COMPLETE. |

### Phase E — Legacy API Deprecation & Governance
Goal: Retire or wrap `ptycho_torch/api/`, update documentation, and capture governance sign-off.
Prereqs: Phases B–D complete with passing tests.
Exit Criteria: `api/` surface deprecated, ADR status set to Accepted, ledger updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E1 | Governance dossier | [x] | ✅ Phase E.A complete (Attempts #55–57). ADR addendum authored (`reports/2025-10-20T134500Z/phase_e_governance_adr_addendum/`), spec §§4.7–4.9 redlined (`reports/2025-10-20T150020Z/phase_e_governance_spec_redline/`), workflow guide §§11–13 refreshed + CONFIG-002 finding added (`reports/2025-10-20T151734Z/phase_e_governance_workflow_docs/`). |
| E2 | Execution knobs hardening | [x] | ✅ EB1–EB4 complete (Attempts #60–71). Logger backend implemented (CSVLogger default, TensorBoard/MLflow optional, `--disable_mlflow` deprecated). Docs sync artifacts at `phase_e_execution_knobs/2025-10-23T110500Z/docs/2025-10-24T041500Z/` (spec §4.9/§7.1 + workflow guide §12 updated, CONFIG-LOGGER-001 added to findings). Runtime smoke at `phase_e_execution_knobs/runtime_smoke/2025-10-24T061500Z/`. |
| E3 | Deprecation & closure | [x] | ✅ Phase E.C complete (Attempts #72, #TBD-closeout). E.C1: API deprecation warning landed (`reports/2025-10-24T070500Z/phase_e_governance/api_deprecation/2025-10-24T070500Z/`, tests GREEN). E.C2/E.C3: Governance ledger sync + archival decisions documented (`reports/2025-11-04T093500Z/phase_e_governance_closeout/docs/summary.md`). |

## Reporting Discipline
- All artefacts stored under `plans/active/ADR-003-BACKEND-API/reports/<ISO8601>/` with descriptive filenames.
- Tests follow native pytest style; full-suite runs coordinated with existing integration plans.
- Reference this plan from docs/fix_plan.md entry `[ADR-003-BACKEND-API]`.
