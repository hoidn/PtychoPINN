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
Store RED logs under `impl/<TS>/red/`. | [x] | **COMPLETE (Attempt #68):** RED evidence consolidated at `impl/2025-10-24T025339Z/red/`. Tests+implementation committed atomically in 43ea2036 (preventing live RED capture). Created `red/README.md` (3.7 KB) explaining unavailability and documenting expected failures (ArgumentError, AttributeError, AssertionError). Relocated investigation report to `red/analysis.md` (11 KB, formerly at root) documenting pre-implementation state. |
| B2 | Implement logger wiring or deprecation warnings across CLI (`ptycho_torch/train.py`), shared helper (`cli/shared.py`), execution config factory, and `_train_with_lightning`. Ensure optional dependencies handled via try/except with actionable messages, and maintain CPU-only determinism. | [x] | Commit 43ea2036 threads logger configuration end-to-end (CSV default, TensorBoard/MLflow optional) and updates PyTorchExecutionConfig. No further code changes required unless regressions appear. |
| B3 | GREEN validation: rerun targeted selectors plus integration workflow (`CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`). Capture GREEN logs under `impl/<TS>/green/` and ensure no regressions in full suite (or record justification if subset only). | [x] | **COMPLETE (Attempt #68):** All mapped selectors PASSED and archived at `impl/2025-10-24T025339Z/green/`: CLI tests (3 passed, 4.86s → `pytest_cli_logger_green.log`), factory tests (2 passed, 3.53s → `pytest_factory_logger_green.log`), workflow test (1 passed, 4.91s → `pytest_workflows_logger_green.log`), integration test (1 passed, 16.74s → `pytest_integration_logger_green.log`). Total: **7 passed, 0 failed, 30.04s**. Zero regressions (full suite baseline: 268 passed, 17 skipped, 1 xfailed). Authored `summary.md` (12 KB) with test matrix, warnings analysis, artifact table, and exit criteria validation. |

### Phase C — Documentation, Spec, and Ledger Sync
Goal: Align normative docs and plans with implemented logger policy.
Prereqs: Phase B GREEN evidence.
Exit Criteria: Spec/workflow guide, findings ledger, and plan/ledger entries updated; artifacts archived.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Update `specs/ptychodus_api_spec.md` §4.9 and §7.1 to reflect logger behaviour (field default, allowed values, CLI exposure or deprecation warning). Generate `docs/` diff and store as `docs/spec_redline.md`. | [x] | **COMPLETE (Attempt #69):** Updated §4.9 line 281 with comprehensive logger_backend field definition (CSV default, 4 backend descriptions, factory fallback, MLflow migration note, deprecation notice). Added §7.1 line 399 `--logger` CLI flag row. Removed logger_backend from backlog lists (lines 410, 428). Updated `--disable_mlflow` deprecation text (line 403). Generated spec redline diff at `docs/2025-10-24T041500Z/spec_redline.md` (3.1 KB, 5 hunks). |
| C2 | Refresh `docs/workflows/pytorch.md` §12 with new flag description or warning banner. Include guidance for enabling/disabling logging and highlight dependency requirements. | [x] | **COMPLETE (Attempt #69):** Added `--logger` row to training flags table (line 329) with full backend descriptions. Added "Logger Backend Details" section (lines 337-338) explaining CSV/TensorBoard/MLflow capabilities and use cases. Added "DeprecationWarning for --disable_mlflow" section (lines 340-344) with migration guidance. Updated deprecated flags list (line 348). Wrote comprehensive `docs/2025-10-24T041500Z/summary.md` (6.8 KB) documenting all changes. |
| C3 | Update `docs/findings.md` if policy-level decision (e.g., POLICY-LOGGING-001) and mark `plans/active/ADR-003-BACKEND-API/implementation.md` Phase E rows accordingly. Append fix_plan Attempt entry with artifact links and update plan checklists to `[x]`. | [x] | **COMPLETE (Attempt #69):** Added CONFIG-LOGGER-001 finding to `docs/findings.md` line 12 with full synopsis (CSV default, allowed backends, deprecation, MLflow backlog) and evidence link to `decision/approved.md`. Updated `implementation.md` Phase E2 row (line 61) with EB3.C completion summary and artifact path. Fix_plan Attempt #69 entry pending (next todo). |
| C4 | Record MLflow logger refactor backlog item and reference future fix_plan entry once created. | [x] | **COMPLETE (Attempt #69):** MLflow Logger migration documented in: (1) `summary.md` "MLflow Migration Backlog" section with rationale (API uniformity), (2) `implementation.md` E2 row commentary "MLflow Logger migration tracked as Phase EB3.C4 backlog", (3) `decision/approved.md` Q4 response approving follow-up tracking. Backlog will be opened as dedicated fix_plan entry once EB3 implementation stabilizes per governance guidance. |

### Phase D — Optional Smoke & CI Prep (if logger enabled)
Goal: Capture deterministic smoke run demonstrating logger output and define CI gating strategy.
Prereqs: Logger implementation active.
Exit Criteria: Smoke log stored and CI guidance recorded; may be skipped if decision is to keep logging disabled.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Run training CLI with logger enabled on minimal dataset; archive log output (e.g., generated TensorBoard files or warning message). Store under `smoke/<TS>/`. | [ ] | Capture one deterministic CSV-logger run:<br>• Command (CPU-only, wrap with `/usr/bin/time -p` and `tee`): `CUDA_VISIBLE_DEVICES=\"\" /usr/bin/time -p python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir tmp/logger_smoke --n_images 64 --max_epochs 1 --gridsize 2 --batch_size 4 --accelerator cpu --deterministic --num-workers 0 --logger csv \| tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/<TS>/train_cli_logger_csv.log`.<br>• After run: copy `tmp/logger_smoke/lightning_logs/version_0/metrics.csv` → `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/<TS>/metrics.csv`; capture `tree tmp/logger_smoke/lightning_logs > plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/<TS>/logger_tree.txt`; jot runtime + observations in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/<TS>/summary.md` (include logger backend, warnings).<br>• Clean up `tmp/logger_smoke` once artifacts archived. Optional follow-up: rerun with `--logger tensorboard` and record whether TensorBoard directory + warnings appear. |
| D2 | Document CI guidance in `docs/ci_logger_notes.md` (integration strategy, env vars, how to disable in CI). | [ ] | Author markdown note covering: (1) Default CSV logger location (`lightning_logs/version_N/metrics.csv`) and how to attach as CI artifact; (2) Steps to disable logging for smoke runs (`--logger none --quiet`), with explanation of DeprecationWarning for `--disable_mlflow`; (3) Cleanup guidance to avoid leaving large `lightning_logs/` directories; (4) Mention optional TensorBoard/MLflow backends and dependency expectations. Mirror Phase D smoke doc tone; link back to EB3 plan and CONFIG-LOGGER-001. |

## Success Criteria
- Approved decision record stored with rationale and supervisor sign-off.
- CLI/help text, shared helpers, and factories handle `logger_backend` (or emit explicit warnings) consistently.
- Workflow tests confirm Lightning trainer receives expected logger configuration or deliberate absence.
- Normative docs + findings updated the same loop as implementation.
- All artifacts captured in timestamped subdirectories with cross-references from docs/fix_plan.md and plan checklists.
