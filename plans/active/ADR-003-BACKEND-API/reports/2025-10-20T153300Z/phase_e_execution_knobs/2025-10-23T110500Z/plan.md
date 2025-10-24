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
  - Decision record: `decision/approved.md` (2025-10-23) — CSV default + TensorBoard option approved; `--disable_mlflow` deprecation and MLflow refactor follow-up required.
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
| A1 | Catalogue current logging hooks and legacy expectations. Inspect `ptycho_torch/workflows/components.py` and legacy CLI under `ptycho_torch/api/` to document existing MLflow/TensorBoard usage (or lack thereof). Summarise findings in `analysis/current_state.md`. | [x] | **COMPLETE (Attempt #65):** Audit delivered at `analysis/current_state.md` (12 KB). Key findings: (1) MLflow fully active (train.py:75-80, 306-340), (2) Lightning logger intentionally disabled (components.py:760), (3) PyTorch has MORE logging than TensorFlow baseline, (4) Loss metrics currently lost (self.log() calls not captured). Used parallel Explore subagent for comprehensive search. |
| A2 | Evaluate Lightning logger options and dependency impact. Create `analysis/options_matrix.md` comparing at minimum: `None` (no logger, explicit disable), `TensorBoardLogger`, `MLFlowLogger`, `CSVLogger`. Assess dependency footprint, CI viability, existing install extras, and API surface changes. | [x] | **COMPLETE (Attempt #65):** Options matrix delivered at `analysis/options_matrix.md` (26 KB). Evaluated 6 options (None, CSV, TensorBoard, MLflow, WandB, Neptune) across 5 criteria (deps, pros/cons, CI, user impact). **Recommendation:** CSVLogger (Tier 1 MVP) + TensorBoardLogger (Tier 2 optional). Both satisfy POLICY-001 (zero new deps). Used general-purpose subagent for Lightning docs research. |
| A3 | Draft decision proposal (`decision/proposal.md`) recommending path forward (e.g., adopt TensorBoardLogger by default with optional CLI flag, or emit DeprecationWarning and keep logging disabled). Include pros/cons, acceptance criteria, and required CLI/config changes. Circulate to supervisor (galph) for approval before implementation. | [x] | **COMPLETE (Attempt #65):** Decision proposal delivered at `decision/proposal.md` (18 KB). **Recommendation:** Enable CSVLogger by default (`logger_backend='csv'`), support TensorBoard/MLflow as opt-in, deprecate `--disable_mlflow` with warning. Includes TDD implementation plan (7 tests, ~230 lines code, <2hr effort), acceptance criteria, risks/mitigations, and 4 open questions for supervisor approval (Q1: approve CSV default?, Q2: include TensorBoard in EB3.B?, Q3: deprecate --disable_mlflow?, Q4: track MLflow refactor follow-up?). |

### Phase B — Implementation & Tests (TDD)
Goal: Implement approved logger behaviour with full CLI → factory → workflow coverage.
Prereqs: Phase A decision approved (proposal accepted and preserved in `decision/approved.md`).
Exit Criteria: Tests capture logger wiring, CLI/help text updated, and Lightning trainer receives expected logger instance or explicit warning.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Author RED tests capturing desired behaviour. Depending on decision:
- If enabling logger(s): add CLI tests (`tests/torch/test_cli_train_torch.py`) ensuring `--logger-backend` flag or equivalent populates execution config; add workflow tests verifying `_train_with_lightning` passes logger to `Trainer` (monkeypatch Lightning logger classes).
- If deprecating logging: add tests asserting CLI warns when unsupported value passed and that execution config defaults to `None`.
Store RED logs under `impl/<TS>/red/`. | [P] | New CLI/factory/workflow tests landed in commit 43ea2036, but RED evidence was not archived. Capture pre-fix selector output under `impl/2025-10-24T025339Z/red/` (or newer) before marking complete. |
| B2 | Implement logger wiring or deprecation warnings across CLI (`ptycho_torch/train.py`), shared helper (`cli/shared.py`), execution config factory, and `_train_with_lightning`. Ensure optional dependencies handled via try/except with actionable messages, and maintain CPU-only determinism. | [x] | Commit 43ea2036 threads logger configuration end-to-end (CSV default, TensorBoard/MLflow optional) and updates PyTorchExecutionConfig. No further code changes required unless regressions appear. |
| B3 | GREEN validation: rerun targeted selectors plus integration workflow (`CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`). Capture GREEN logs under `impl/<TS>/green/` and ensure no regressions in full suite (or record justification if subset only). | [P] | Full test suite log (`pytest_full_suite_green.log`) captured, but targeted CLI/factory/workflow selectors and loop summary are missing. Re-run mapped tests, archive outputs under `impl/2025-10-24T025339Z/green/`, and author `summary.md` before closing. |

### Phase C — Documentation, Spec, and Ledger Sync
Goal: Align normative docs and plans with implemented logger policy.
Prereqs: Phase B GREEN evidence.
Exit Criteria: Spec/workflow guide, findings ledger, and plan/ledger entries updated; artifacts archived.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Update `specs/ptychodus_api_spec.md` §4.9 and §7.1 to reflect logger behaviour (field default, allowed values, CLI exposure or deprecation warning). Generate `docs/` diff and store as `docs/spec_redline.md`. | [ ] | Mirror EB2 doc process: use `git diff` to populate `docs/spec_redline.md`. Reference Lightning logger semantics directly. |
| C2 | Refresh `docs/workflows/pytorch.md` §12 with new flag description or warning banner. Include guidance for enabling/disabling logging and highlight dependency requirements. | [ ] | Keep table text ASCII, align with spec phrasing. |
| C3 | Update `docs/findings.md` if policy-level decision (e.g., POLICY-LOGGING-001) and mark `plans/active/ADR-003-BACKEND-API/implementation.md` Phase E rows accordingly. Append fix_plan Attempt entry with artifact links and update plan checklists to `[x]`. | [ ] | Ensure Attempt includes artifact directory `2025-10-23T110500Z`. |
| C4 | Record MLflow logger refactor backlog item and reference future fix_plan entry once created. | [ ] | During documentation sync, add a note in plan summary and fix_plan Attempts pointing to the follow-up (Lightning `MLFlowLogger` migration) approved in `decision/approved.md`. |

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
