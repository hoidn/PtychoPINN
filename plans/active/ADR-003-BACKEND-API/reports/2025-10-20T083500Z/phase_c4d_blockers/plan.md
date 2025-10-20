## Context
- Initiative: ADR-003-BACKEND-API (Standardize PyTorch backend API)
- Phase Goal: Clear Phase C4.D blockers by delivering a functional PyTorch bundle loader and restoring CLI training parity so the integration workflow selector can run end-to-end.
- Dependencies: `specs/ptychodus_api_spec.md` §4.6–§4.8 (bundle contract & CLI routing), `docs/workflows/pytorch.md` §§5–7 (persistence & inference), `docs/findings.md` (`CONFIG-001`, `BUG-TF-001`), Phase C4 artefacts (`plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/`).

### Phase A — Bundle Loader Enablement
Goal: Implement `load_torch_bundle` model reconstruction (Phase D3.C dependency) using TDD so inference CLI no longer raises `NotImplementedError`.
Prereqs: Factory payloads GREEN (C4.C), bundle persistence test `test_bundle_persistence` GREEN, Poisson parity fix landed (commit `e10395e7`).
Exit Criteria: Targeted bundle loader test goes RED→GREEN and integration selector reaches inference without `NotImplementedError`.

| ID | Task Description | State | How/Why & Guidance (including API / document / artifact / source file references) |
| --- | --- | --- | --- |
| A1 | Author failing bundle loader test | [x] | ✅ RED selector captured in `reports/2025-10-20T083500Z/phase_c4d_blockers/pytest_load_bundle_red.log` (NotImplementedError baseline). Test added at `tests/torch/test_model_manager.py::TestLoadTorchBundle::test_reconstructs_models_from_bundle`. |
| A2 | Implement `load_torch_bundle` helper | [x] | ✅ `ptycho_torch/model_manager.py` now exposes `create_torch_model_with_gridsize()` + dual-model `load_torch_bundle()` with CONFIG-001 restoration; workflow shim updated. See commit `40968c02` and loop summary. |
| A3 | GREEN verification | [x] | ✅ Selector GREEN log `pytest_load_bundle_green_final.log`; integration selector advanced to inference (log stored as `pytest_integration_phase_a.log`). Observed new runtime failure (`'dict' object has no attribute 'eval'`) during inference CLI — track follow-up under Phase B diagnostics. |

### Phase B — Training Channel Parity
Goal: Align training CLI / workflow configuration so the Lightning model receives the correct channel count for the requested `gridsize`.
Prereqs: Phase A GREEN, `manual_cli_smoke.log` reproduced (see 2025-10-20T081500Z hub).
Exit Criteria: Manual CLI smoke completes at least one Lightning training epoch without channel-count RuntimeError; targeted regression captures the fix.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Capture regression test | [x] | ✅ RED/Green cycle captured (2025-10-20T103500Z hub) — new pytest `test_lightning_training_respects_gridsize` reproduces in_channels mismatch and passes post-fix (`pytest_gridsize_{red,green}.log`). |
| B2 | Diagnose & patch gridsize mismatch | [x] | ✅ 2025-10-20 — `_build_lightning_dataloaders` now permutes `coords_relative` to `(batch, gridsize**2, 1, 2)` and marks tensors `contiguous()`. Regression `test_coords_relative_layout` + gridsize smoke selectors captured under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T104500Z/phase_c4d_coords_fix/{pytest_coords_layout_{red,green}.log,pytest_gridsize_regression.log}`. Manual CLI with `--gridsize 1` succeeded (see `manual_cli_smoke_green.log`). |
| B3 | GREEN validation | [ ] | With axis fix in place, rerun targeted selectors for parity and extend CLI smoke to `gridsize=2`. Capture logs under new hub `plans/active/ADR-003-BACKEND-API/reports/<ISO8601>/phase_c4d_at_parallel/`. Steps: (1) Re-run `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv` (store as `pytest_gridsize_green.log`); (2) Execute bundle + workflow selectors (`pytest tests/torch/test_integration_workflow_torch.py::{test_bundle_loader_returns_modules,test_run_pytorch_train_save_load_infer} -vv`) to ensure no regressions; (3) Run CPU CLI smoke with `--gridsize 2`, `--n_images 64`, `--batch_size 4`, `--max_epochs 1`, `--disable_mlflow`, `--device cpu` and store output as `manual_cli_smoke_gs2.log`. If CLI fails due to grouping limits, capture log as blocker and revert this row to `[P]`. |
| B4 | Inspect bundle module types | [x] | ✅ 2025-10-20 — Regression test + bundle introspection confirmed both `diffraction_to_obj` and `autoencoder` load as `PtychoPINN_Lightning` instances (see `reports/2025-10-20T093500Z/phase_c4d_bundle_probe/summary.md`). |

### Phase C — Integration Close-Out & Ledger Updates
Goal: Document results, update plans, and ensure ledger reflects the cleared blockers.
Prereqs: Phases A & B GREEN.
Exit Criteria: Plan tables updated, summary written, fix_plan attempt logged.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Update phase C4 plan entries | [ ] | Mark C4.D3/C4.D4 rows in `reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md` according to outcomes; reference new artifact paths under 2025-10-20T083500Z hub. |
| C2 | Append supervisor/engineer summaries | [ ] | Author `summary.md` in this directory capturing decisions, metrics (runtime, channel counts), and remaining follow-ups (e.g., docs updates for `docs/workflows/pytorch.md` §12). |
| C3 | Update docs/fix_plan.md attempts | [ ] | Log Attempt #29+ with links to new logs/tests, note whether C4 transitions to `[x]` or remains `[P]`. |

## Reporting Discipline
- Store all new artefacts under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/`.
- RED/Green logs should follow `{pytest_selector}_{red|green}.log}` naming.
- Manual CLI runs pipe through `tee` to capture stdout/stderr.
- Reference this plan from `docs/fix_plan.md` entry `[ADR-003-BACKEND-API]` during updates.
