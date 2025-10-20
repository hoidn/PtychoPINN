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
| A1 | Author failing bundle loader test | [ ] | Create `tests/torch/test_model_manager.py::TestLoadTorchBundle::test_reconstructs_models_from_bundle` asserting that loading `wts.h5.zip` returns a dict with `['diffraction_to_obj', 'autoencoder']` and hydrated dataclass config. Use the mini bundle captured in `reports/2025-10-20T060955Z/phase_c4_cli_integration_debug/pytest_cli_train_bundle_green_final.log` (CLI run emits bundle to tmpdir). Store RED log under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/pytest_load_bundle_red.log`. Command: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_reconstructs_models_from_bundle -vv`. |
| A2 | Implement `load_torch_bundle` helper | [ ] | Add `create_torch_model_with_gridsize()` + `load_torch_bundle()` logic in `ptycho_torch/model_manager.py` leveraging `ModelConfig` from serialized metadata (see TensorFlow analogue `ptycho/model_manager.py:create_model`). Ensure CONFIG-001 bridge runs before returning. Update `ptycho_torch/workflows/components.py::load_inference_bundle_torch` call site accordingly. |
| A3 | GREEN verification | [ ] | Rerun selector from A1; capture GREEN log `pytest_load_bundle_green.log`. Then run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv` to confirm inference proceeds past bundle load (store to `pytest_integration_phase_a.log`). |

### Phase B — Training Channel Parity
Goal: Align training CLI / workflow configuration so the Lightning model receives the correct channel count for the requested `gridsize`.
Prereqs: Phase A GREEN, `manual_cli_smoke.log` reproduced (see 2025-10-20T081500Z hub).
Exit Criteria: Manual CLI smoke completes at least one Lightning training epoch without channel-count RuntimeError; targeted regression captures the fix.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Capture regression test | [ ] | Add pytest in `tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining` (new method `test_lightning_training_respects_gridsize`) that builds a training payload with `gridsize=2` and asserts the first conv layer input channels match `gridsize**2`. Run RED selector `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv` → store RED log `pytest_gridsize_red.log`. |
| B2 | Diagnose & patch gridsize mismatch | [ ] | Inspect `ptycho_torch/config_factory.py` (grid_size handling), `ptycho_torch/model.py::create_model` (input channels), and `_build_lightning_dataloaders` to ensure `config.model.gridsize` coherently propagates. Update factories/model so `diffraction_to_obj` expects `grid_size**2` channels. Ensure `params.cfg['gridsize']` remains synchronized (see `docs/findings.md#BUG-TF-001`). |
| B3 | GREEN validation | [ ] | Re-run selector from B1 capturing GREEN log `pytest_gridsize_green.log`. Execute manual smoke `CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --n_images 64 --max_epochs 1 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-4` (add `--gridsize 2` if required post-fix) and store stdout to `manual_cli_smoke_green.log`. |

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
