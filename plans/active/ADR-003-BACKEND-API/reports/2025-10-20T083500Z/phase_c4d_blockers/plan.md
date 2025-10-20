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
| B1 | Capture regression test | [ ] | Add pytest in `tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining` (new method `test_lightning_training_respects_gridsize`) that builds a training payload with `gridsize=2` and asserts the first conv layer input channels match `gridsize**2`. Run RED selector `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv` → store RED log `pytest_gridsize_red.log`. |
| B2 | Diagnose & patch gridsize mismatch | [ ] | Inspect `ptycho_torch/config_factory.py` (grid_size handling), `_train_with_lightning` manual config construction, `ptycho_torch/model.py::create_model`, and `_build_lightning_dataloaders` to ensure `config.model.gridsize` coherently propagates. Update `_train_with_lightning` to reuse factory-derived `PyTorch` configs (set `C`, `C_model`, `C_forward` = `gridsize**2`) so `PtychoPINN_Lightning` first conv expects four channels when `gridsize=2`. Keep `params.cfg['gridsize']` synchronized (see `docs/findings.md#BUG-TF-001`). |
| B3 | GREEN validation | [ ] | Re-run selector from B1 capturing GREEN log `pytest_gridsize_green.log`. Execute manual smoke `CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --n_images 64 --max_epochs 1 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-4` (add `--gridsize 2` if required post-fix) and store stdout to `manual_cli_smoke_green.log`. |
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
