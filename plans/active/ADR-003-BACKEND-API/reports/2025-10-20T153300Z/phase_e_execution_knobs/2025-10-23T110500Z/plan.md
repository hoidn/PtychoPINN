# Phase EB3 — Logger Governance & Implementation Plan

## Context
- Initiative: ADR-003-BACKEND-API (Standardize PyTorch backend API)
- Phase Goal: Resolve logging/experiment tracking strategy for the PyTorch CLI, implement the agreed behaviour (logger wiring or formal deprecation), and align documentation/tests before handing off to governance close-out.
- Dependencies:
  - `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md` (Phase E master checklist)
  - `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/open_questions.md` §Q2 (MLflow/logger ownership)
  - `specs/ptychodus_api_spec.md` §4.9 (PyTorchExecutionConfig fields), §7.1 (training CLI table)
  - `docs/workflows/pytorch.md` §12 (training CLI guide)
  - Lightning logger reference: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.html
- Artifact Hub: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/`
  - Store option analysis under `analysis/`
  - Decision record under `decision/`
  - Implementation evidence logs under `impl/<timestamp>/`
  - Documentation diffs and summaries under `docs/`

### Phase A — Decision Analysis & Requirements
Goal: Determine the authoritative logging policy (enable Lightning logger(s) vs. explicit deprecation) with governance-ready evidence.
Prereqs: Review Phase B factory documents and existing CLI behaviour (no logger flag today).
Exit Criteria: Decision record summarising chosen path, dependency analysis, and acceptance rationale (approved by supervisor loop).

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Catalogue current logging hooks and legacy expectations. Inspect `ptycho_torch/workflows/components.py` and legacy CLI under `ptycho_torch/api/` to document existing MLflow/TensorBoard usage (or lack thereof). Summarise findings in `analysis/current_state.md`. | [ ] | Run `rg "mlflow" -n ptycho_torch` and `rg "logger" -n ptycho_torch` to confirm zero-touch baseline. Note any TODOs referencing MLflow. Include comparison to TensorFlow logging (see `ptycho/workflows/components.py`). |
| A2 | Evaluate Lightning logger options and dependency impact. Create `analysis/options_matrix.md` comparing at minimum: `None` (no logger, explicit disable), `TensorBoardLogger`, `MLFlowLogger`, `CSVLogger`. Assess dependency footprint, CI viability, existing install extras, and API surface changes. | [ ] | Use Lightning docs + project findings. Cross-reference POLICY-001 (torch required) to ensure no new mandatory deps without governance approval. |
| A3 | Draft decision proposal (`decision/proposal.md`) recommending path forward (e.g., adopt TensorBoardLogger by default with optional CLI flag, or emit DeprecationWarning and keep logging disabled). Include pros/cons, acceptance criteria, and required CLI/config changes. Circulate to supervisor (galph) for approval before implementation. | [ ] | Reference `open_questions.md` §Q2 to align canonical vs execution config responsibilities. Highlight how `logger_backend` field will be used (values, defaults). |

### Phase B — Implementation & Tests (TDD)
Goal: Implement approved logger behaviour with full CLI → factory → workflow coverage.
Prereqs: Phase A decision approved (proposal accepted and preserved in `decision/approved.md`).
Exit Criteria: Tests capture logger wiring, CLI/help text updated, and Lightning trainer receives expected logger instance or explicit warning.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Author RED tests capturing desired behaviour. Depending on decision:
- If enabling logger(s): add CLI tests (`tests/torch/test_cli_train_torch.py`) ensuring `--logger-backend` flag or equivalent populates execution config; add workflow tests verifying `_train_with_lightning` passes logger to `Trainer` (monkeypatch Lightning logger classes).
- If deprecating logging: add tests asserting CLI warns when unsupported value passed and that execution config defaults to `None`.
Store RED logs under `impl/<TS>/red/`. | [ ] | Follow TDD guidance in `docs/DEVELOPER_GUIDE.md` and `docs/TESTING_GUIDE.md`. Use selectors like `pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_logger_backend_roundtrip -vv` (update test names accordingly). |
| B2 | Implement logger wiring or deprecation warnings across CLI (`ptycho_torch/train.py`), shared helper (`cli/shared.py`), execution config factory, and `_train_with_lightning`. Ensure optional dependencies handled via try/except with actionable messages, and maintain CPU-only determinism. | [ ] | Preserve CONFIG-001 ordering (call `update_legacy_dict` before creating loggers if they rely on config). If logger requires extra packages, guard with informative ImportError instructing to install `[logging]` extra. |
| B3 | GREEN validation: rerun targeted selectors plus integration workflow (`CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`). Capture GREEN logs under `impl/<TS>/green/` and ensure no regressions in full suite (or record justification if subset only). | [ ] | Use runtime guardrails from TEST-PYTORCH-001 Phase D (≤90s). Include log excerpts proving logger interaction (e.g., trainer initialization args). |

### Phase C — Documentation, Spec, and Ledger Sync
Goal: Align normative docs and plans with implemented logger policy.
Prereqs: Phase B GREEN evidence.
Exit Criteria: Spec/workflow guide, findings ledger, and plan/ledger entries updated; artifacts archived.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Update `specs/ptychodus_api_spec.md` §4.9 and §7.1 to reflect logger behaviour (field default, allowed values, CLI exposure or deprecation warning). Generate `docs/` diff and store as `docs/spec_redline.md`. | [ ] | Mirror EB2 doc process: use `git diff` to populate `docs/spec_redline.md`. Reference Lightning logger semantics directly. |
| C2 | Refresh `docs/workflows/pytorch.md` §12 with new flag description or warning banner. Include guidance for enabling/disabling logging and highlight dependency requirements. | [ ] | Keep table text ASCII, align with spec phrasing. |
| C3 | Update `docs/findings.md` if policy-level decision (e.g., POLICY-LOGGING-001) and mark `plans/active/ADR-003-BACKEND-API/implementation.md` Phase E rows accordingly. Append fix_plan Attempt entry with artifact links and update plan checklists to `[x]`. | [ ] | Ensure Attempt includes artifact directory `2025-10-23T110500Z`. |

### Phase D — Optional Smoke & CI Prep (if logger enabled)
Goal: Capture deterministic smoke run demonstrating logger output and define CI gating strategy.
Prereqs: Logger implementation active.
Exit Criteria: Smoke log stored and CI guidance recorded; may be skipped if decision is to keep logging disabled.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Run training CLI with logger enabled on minimal dataset; archive log output (e.g., generated TensorBoard files or warning message). Store under `smoke/<TS>/`. | [ ] | Use TEST-PYTORCH-001 fixture to keep runtime <60s. Document generated files and clean up if large. |
| D2 | Document CI guidance in `docs/ci_logger_notes.md` (integration strategy, env vars, how to disable in CI). | [ ] | Align with Phase D3 style from TEST-PYTORCH-001. |

## Success Criteria
- Approved decision record stored with rationale and supervisor sign-off.
- CLI/help text, shared helpers, and factories handle `logger_backend` (or emit explicit warnings) consistently.
- Workflow tests confirm Lightning trainer receives expected logger configuration or deliberate absence.
- Normative docs + findings updated the same loop as implementation.
- All artifacts captured in timestamped subdirectories with cross-references from docs/fix_plan.md and plan checklists.
