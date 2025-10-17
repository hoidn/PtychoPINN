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
| A1 | Inventory CLI flags and programmatic entry points | [ ] | Document mapping of `train.py`/`inference.py` args → config fields; capture results under `reports/<timestamp>/cli_inventory.md`. |
| A2 | Catalogue backend-specific execution knobs | [ ] | List Lightning/MLflow/device parameters currently used; map to proposed `PyTorchExecutionConfig`. |
| A3 | Confirm overlaps with existing plans | [ ] | Reference `INTEGRATE-PYTORCH-001` plan to avoid duplication; note cross-plan dependencies. |

### Phase B — Configuration Factories
Goal: Centralize canonical `TF*Config` construction for PyTorch backend via shared factories.
Prereqs: Phase A inventory complete; consensus on execution config fields.
Exit Criteria: Factory modules authored with TDD coverage and linked to config bridge tests.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Draft factory design doc | [ ] | Outline required inputs/outputs, validation rules, and override semantics. Store under `reports/<timestamp>/factory_design.md`. |
| B2 | Implement training/inference factory functions | [ ] | Author `ptycho_torch/config_factory.py` with unit tests (pytest) ensuring override enforcement. |
| B3 | Update config bridge tests | [ ] | Extend `tests/torch/test_config_bridge.py` to assert factories produce valid canonical configs end-to-end. |

### Phase C — Core Workflow Refactor
Goal: Embed canonical + execution config pattern into `ptycho_torch/workflows/components.py` and downstream helpers.
Prereqs: Factories merged; execution config dataclass finalised.
Exit Criteria: Workflows accept dual configs, Lightning orchestration centralized, legacy params handshake verified.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Introduce `PyTorchExecutionConfig` | [ ] | Add dataclass to `config_params.py` with device/strategy/precision flags plus docs. |
| C2 | Refactor training/inference workflows | [ ] | Update `run_pytorch_training_workflow`/`run_pytorch_inference_workflow` to consume execution config, ensure CONFIG-001 guard remains. |
| C3 | Update helper pathways | [ ] | Ensure `_train_with_lightning`, inference loaders, and persistence helpers honor execution config settings. |
| C4 | Add workflow-level tests | [ ] | Add pytest coverage verifying device selection, MLflow toggles, and params.cfg updates. |

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

