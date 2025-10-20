Summary: TDD the Lightning training path so PyTorch respects gridsize-derived channel counts before rerunning the integration workflow.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4.D (bundle loader unblock)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/{pytest_gridsize_red.log,pytest_gridsize_green.log,pytest_integration_green.log,manual_cli_smoke_green.log}

Do Now:
1. ADR-003-BACKEND-API C4.D.B1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — add `test_lightning_training_respects_gridsize` in `tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining`, assert that `_train_with_lightning` raises when the first Conv2d still expects 1 channel for `gridsize=2`, then capture the RED run via `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/pytest_gridsize_red.log`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv
2. ADR-003-BACKEND-API C4.D.B2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — refactor `_train_with_lightning` to reuse factory-derived PyTorch configs (propagate `grid_size`/`C` into `PTDataConfig`, `PTModelConfig`, `PTTrainingConfig`), ensure `PtychoPINN_Lightning` and dataloaders agree on `grid_size**2` channels, and record the GREEN rerun with `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/pytest_gridsize_green.log`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv
3. ADR-003-BACKEND-API C4.D.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — rerun the integration selector (`CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/pytest_integration_green.log`) and the manual CLI smoke (`CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --n_images 64 --max_epochs 1 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-4 --gridsize 2 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/manual_cli_smoke_green.log`); tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv

If Blocked: Capture failing selector output and stack traces under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/` (add `blocker.md`), set the relevant plan rows (B1/B2/B3) back to `[P]` with notes, and summarize the obstacle in docs/fix_plan.md Attempts.

Priorities & Rationale:
- `_train_with_lightning` at ptycho_torch/workflows/components.py:620 currently rebuilds PyTorch configs with default `C_model=1`, violating spec §4.6 dual-model parity when `gridsize>1`.
- `docs/findings.md#BUG-TF-001` warns that missing gridsize propagation causes channel mismatches; honoring CONFIG-001 requires propagating `grid_size**2` everywhere.
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md` keeps Phase C4.D blocked on B1–B3 until Lightning training reproduces TensorFlow grouping semantics.
- Integration selector (`tests/torch/test_integration_workflow_torch.py`) is the authoritative acceptance test for bundle persistence per specs/ptychodus_api_spec.md §4.6–§4.8.

How-To Map:
- `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/pytest_gridsize_red.log`
- `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/pytest_gridsize_green.log`
- `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/pytest_integration_green.log`
- `CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --n_images 64 --max_epochs 1 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-4 --gridsize 2 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/manual_cli_smoke_green.log`
- Remove or relocate any new `train_debug.log` from repo root back into the artifact directory before finishing.

Pitfalls To Avoid:
- Do not touch TensorFlow persistence code or change `load_torch_bundle` signatures—focus on `_train_with_lightning`/config propagation.
- Keep TDD ordering: add the failing test before modifying production code.
- Maintain CONFIG-001 ordering (`update_legacy_dict`) when introducing new helper calls.
- Ensure permutes remain device/dtype neutral; no `.cuda()` calls in tests.
- Avoid baking dataset-specific hacks (e.g., hard-coding channel counts); derive from config.
- Capture logs under the new timestamped directory; no artifacts at repository root.
- Do not delete existing evidence hubs (e.g., 2025-10-20T093500Z); append new material only.
- Respect pytest runtime budget (<20s) for the new regression test—use fixtures already provided.
- Keep sentinel `autoencoder` bundle behavior intact when adjusting results dictionaries.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md
- ptycho_torch/workflows/components.py:620
- ptycho_torch/config_factory.py:196
- docs/findings.md:11
- specs/ptychodus_api_spec.md:205

Next Up:
- 1. Phase C4.E documentation updates once C4.D unblocked (workflow guide + spec tables).
- 2. Phase C4.F summary + hygiene wrap-up for ADR-003 plan.
