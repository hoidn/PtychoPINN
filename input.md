Summary: Validate Phase C4.D.B3 parity by rerunning gridsize regression and executing a gridsize=2 CLI smoke to capture fresh GREEN evidence.
Mode: Parity
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4.D (bundle loader unblock)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/{pytest_gridsize_green.log,pytest_bundle_loader_green.log,pytest_integration_green.log,manual_cli_smoke_gs2.log,summary.md}

Do Now:
1. ADR-003-BACKEND-API C4.D.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — Rerun the targeted regression `TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize` to confirm the axis fix still holds; store the GREEN log as `pytest_gridsize_green.log` under the new artifact hub. tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv
2. ADR-003-BACKEND-API C4.D.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — Execute the PyTorch bundle + workflow selectors and capture logs (`pytest_bundle_loader_green.log`, `pytest_integration_green.log`) to ensure no regressions before the CLI smoke. tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv
3. ADR-003-BACKEND-API C4.D.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — Run the CPU CLI smoke with gridsize=2 and log output as `manual_cli_smoke_gs2.log`; reuse the minimal fixture and pass `--n_images 64`, `--batch_size 4`, `--max_epochs 1`, `--disable_mlflow`, `--device cpu`, `--accelerator cpu`. tests: none
4. ADR-003-BACKEND-API C4.D.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — Summarize outcomes in `summary.md`, mark plan row B3 `[x]`, and add a new docs/fix_plan.md attempt referencing the fresh logs. tests: none

If Blocked: Capture the failing selector or CLI output into `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/blocker.log`, revert plan row B3 to `[P]`, and describe the blocker in docs/fix_plan.md Attempts History before stopping.

Priorities & Rationale:
- specs/ptychodus_api_spec.md §4.6–§4.8 demand bundle + CLI parity; B3 evidence closes the outstanding Phase C4.D gate.
- docs/workflows/pytorch.md §§5–7 document persistence and CLI usage; gridsize=2 smoke ensures guidance reflects reality.
- findings CONFIG-001 / BUG-TF-001 require verifying params.cfg + gridsize alignment after the coords permute fix.
- phase_c4d_blockers/plan.md tracks the remaining checklist; closing B3 unblocks Phase C close-out rows (C1–C3).

How-To Map:
- `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel`
- `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/pytest_gridsize_green.log`
- `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/pytest_bundle_loader_green.log`
- `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/pytest_integration_green.log`
- `CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --n_images 64 --gridsize 2 --batch_size 4 --max_epochs 1 --disable_mlflow --device cpu --accelerator cpu --learning-rate 1e-3 --num-workers 0 --deterministic | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/manual_cli_smoke_gs2.log`
- Update `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md` row B3, draft `summary.md` (runtime, tensor shapes, remaining risks), and append Attempt #35 in docs/fix_plan.md with artifact links.

Pitfalls To Avoid:
- Keep `CUDA_VISIBLE_DEVICES=""` set for every command to avoid GPU variance.
- Use the timestamped artifact directory only—no logs at repository root.
- If the gridsize=2 CLI fails due to insufficient groups, document it and back out plan state instead of hacking the dataset.
- Do not modify production code; this loop is evidence-only.
- Ensure pytest selectors remain in native pytest style (no unittest harness).
- Run integration selectors after the unit regression so failures stay localized.
- Leave existing TensorFlow core modules untouched per CLAUDE.md protected assets rule.
- After CLI run, clean `/tmp/cli_smoke` if rerunning to avoid checkpoint reuse confusion.
- Double-check plan/fix_plan updates before committing to avoid ledger drift.
- Capture summary.md with key runtimes and dataset notes; omit raw logs from docs.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md#phase-b
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T104500Z/phase_c4d_coords_fix/pytest_gridsize_regression.log
- specs/ptychodus_api_spec.md#46-configuration-handshake-and-bundle-lifecycle
- docs/workflows/pytorch.md#5-running-complete-training-workflow
- docs/findings.md#CONFIG-001

Next Up:
- Phase C4 documentation updates (plan rows C1–C3) once B3 evidence is captured.
