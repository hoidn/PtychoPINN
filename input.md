Summary: Add a failing dataloader parity test, fix the coords_relative axis order in `_build_lightning_dataloaders`, then rerun the PyTorch bundle + CLI selectors.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4.D (bundle loader unblock)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_coords_relative_layout -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T104500Z/phase_c4d_coords_fix/{summary.md,pytest_coords_layout_red.log,pytest_coords_layout_green.log,pytest_gridsize_regression.log,pytest_bundle_loader_green.log,pytest_integration_green.log,manual_cli_smoke_green.log}

Do Now:
1. ADR-003-BACKEND-API C4.D.B2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — Author pytest `TestWorkflowsComponentsTraining::test_coords_relative_layout` reproducing the current axis ordering bug by asserting the dataloader batch `coords_relative` comes back as `(batch, gridsize**2, 1, 2)`. Capture the RED run with `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_coords_relative_layout -vv | tee .../pytest_coords_layout_red.log`. tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_coords_relative_layout -vv
2. ADR-003-BACKEND-API C4.D.B2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — Fix `_build_lightning_dataloaders` (and any shared helper) so `coords_relative` is permuted/reshaped to `(batch, gridsize**2, 1, 2)` before batching; ensure tensors are `contiguous()` to keep `view` happy. Re-run the new test plus the existing gridsize regression and capture logs: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_coords_relative_layout -vv | tee .../pytest_coords_layout_green.log` and `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv | tee .../pytest_gridsize_regression.log`. tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_coords_relative_layout -vv
3. ADR-003-BACKEND-API C4.D.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — Run the PyTorch bundle + CLI smoke selectors and log results: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv | tee .../pytest_bundle_loader_green.log`; `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee .../pytest_integration_green.log`; `CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --max_epochs 1 --n_images 32 --gridsize 1 --batch_size 4 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-3 --disable_mlflow | tee .../manual_cli_smoke_green.log`. Update plan rows B2→[x], B3→[x] and add a new docs/fix_plan.md attempt linking the logs. tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv

If Blocked: Capture the failing selector output into `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T104500Z/phase_c4d_coords_fix/blocker.log`, revert plan rows B2/B3 to `[P]`, and summarize the obstruction in docs/fix_plan.md Attempts History before stopping.

Priorities & Rationale:
- specs/ptychodus_api_spec.md §4.6 expects grouping metadata channels to align with physics helpers; current axis order violates that contract and surfaces as the Translation reshape crash.
- docs/workflows/pytorch.md §§5–7 document bundle persistence tests as acceptance criteria—selectors must pass before Phase C4.D can close.
- `phase_c4d_blockers/plan.md` B2 now codifies the axis permute + regression-test deliverable; finishing B2 unlocks B3 validation.
- BUG-TF-001 in docs/findings.md flags gridsize-driven shape mismatches as historical regressions—new test ensures we don’t regress again.
- Maintaining evidence under the new timestamp keeps Attempt #33 traceable and prevents stray logs (per CLAUDE.md artifact policy).

How-To Map:
- Test RED (new regression): `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_coords_relative_layout -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T104500Z/phase_c4d_coords_fix/pytest_coords_layout_red.log`
- Implement + GREEN unit tests: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_coords_relative_layout -vv | tee .../pytest_coords_layout_green.log`
- Gridsize parity regression: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv | tee .../pytest_gridsize_regression.log`
- Bundle loader regression: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv | tee .../pytest_bundle_loader_green.log`
- Full workflow selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee .../pytest_integration_green.log`
- CLI smoke: `CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --max_epochs 1 --n_images 32 --gridsize 1 --batch_size 4 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-3 --disable_mlflow | tee .../manual_cli_smoke_green.log`
- Log summary after GREEN run under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T104500Z/phase_c4d_coords_fix/summary.md` with test runtimes + remaining follow-ups.

Pitfalls To Avoid:
- Do not rely on implicit broadcasting—ensure the permuted tensor is `.contiguous()` before calling `.view()`.
- Leave raw helper modules (e.g., `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`) untouched per CLAUDE.md.
- Keep the new pytest fixture in native pytest style; no unittest mix-ins.
- Remove or avoid ad-hoc debug prints; store evidence via logs instead.
- Don’t forget to update plan and docs/fix_plan.md states once selectors pass.
- Keep artifacts inside the timestamped directory; no stray `train_debug.log` at repo root.
- Respect CONFIG-001 ordering—factory must stay the source of truth for params.cfg sync.
- Re-run targeted unit tests before long selectors so failures stay localized.
- Mind tensor dtype/device neutrality; keep everything on CPU for reproducibility.
- Avoid reshaping without `.contiguous()` after permuting axes.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md#phase-b — axis-fix guidance
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103200Z/phase_c4d_coords_debug/summary.md — shape-analysis evidence
- specs/ptychodus_api_spec.md §4.6–§4.8 — config + persistence contract
- docs/workflows/pytorch.md §§5–7, §12 — workflow + parity requirements
- tests/torch/test_workflows_components.py — location for new regression test

Next Up:
- 1. Phase C4 documentation sweep (plan rows C1–C3) once B2/B3 are green.
