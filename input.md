Summary: Wire neighbor_count through the PyTorch factory/CLI so Lightning training uses the right grouping, then rerun the integration evidence to go GREEN.
Mode: Parity
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4.D (bundle loader unblock)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T091341Z/phase_c4d_gridsize_debug/{summary.md,pytest_bundle_loader_failure.log,pytest_bundle_loader_green.log,pytest_integration_green.log,manual_cli_smoke_green.log}

Do Now:
1. Prepare fixture (tests: none) — Run `python scripts/tools/make_pytorch_integration_fixture.py --source datasets/Run1084_recon3_postPC_shrunk_3.npz --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` if the NPZ is missing so the integration selectors have data.
2. ADR-003-BACKEND-API C4.D.B2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — Update `ptycho_torch/train.py` (factory overrides) and any helper wiring so the CLI passes `neighbor_count` (default 4) into `create_training_payload`, ensuring `_train_with_lightning` propagates the same value when it reuses the factory. Confirm `payload.tf_training_config.neighbor_count` stays 4 when the CLI flag is omitted. tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv
3. ADR-003-BACKEND-API C4.D.B3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — Re-run the integration selectors and CLI smoke, capturing logs under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T091341Z/phase_c4d_gridsize_debug/`:
   • `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv | tee .../pytest_bundle_loader_green.log`
   • `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee .../pytest_integration_green.log`
   • `CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --max_epochs 1 --n_images 32 --gridsize 1 --batch_size 4 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-3 --disable_mlflow | tee .../manual_cli_smoke_green.log`
   Update plan row B2→[x]/B3→[x] and append a new docs/fix_plan attempt with artifact links.

If Blocked: Capture the failing selector output into the same report directory (name the file `blocker.log`), revert plan rows B2/B3 to `[P]`, and summarise the obstacle in docs/fix_plan.md Attempts.

Priorities & Rationale:
- Factory defaults neighbor_count=6 unless overridden; specs/ptychodus_api_spec.md §4.6 assumes CLI preserves canonical defaults (CONFIG-001), so we must forward the dataclass default (4).
- Failure log (`plans/.../pytest_bundle_loader_failure.log`) shows Lightning Translation reshape crash — once neighbor_count matches gridsize semantics the grouping pipeline should align with docs/workflows/pytorch.md §§5–7.
- Parity tests in tests/torch/test_workflows_components.py guard the intended channel propagation; rerunning them ensures the refactor didn’t regress.
- Integration selector is the authoritative acceptance test for bundle persistence, so both pytest selectors and manual CLI smoke must pass before plan B3 can close.
- Maintaining artifacts under the timestamped hub keeps docs/fix_plan Attempt #32 traceable and avoids more stray `train_debug.log` files.

How-To Map:
- Fixture generation: `python scripts/tools/make_pytorch_integration_fixture.py --source datasets/Run1084_recon3_postPC_shrunk_3.npz --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`
- Targeted regression: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize -vv`
- Bundle loader test: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv`
- Full workflow selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`
- Manual CLI smoke: `CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --max_epochs 1 --n_images 32 --gridsize 1 --batch_size 4 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-3 --disable_mlflow`
- Move any new `train_debug.log` into the artifact directory or delete it once logs are captured.

Pitfalls To Avoid:
- Do not leave `neighbor_count` at the factory’s default (6); explicitly pass the canonical default when CLI flags are absent.
- Keep modifications isolated to CLI/factory wiring—no changes to core physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Ensure the `.npz` fixture stays under `tests/fixtures/...`; don’t commit large data files outside the ignored paths.
- Preserve CONFIG-001 sequencing (call `update_legacy_dict` before data loading) when adjusting wiring.
- Capture all logs via `tee` into the report directory; do not leave raw logs at repo root.
- Maintain native pytest style in any updated tests; no `unittest.TestCase` mixes.
- Watch for lingering `train_debug.log` and remove or relocate before finishing.
- Re-run targeted test after code changes before jumping to full selector so failures localize quickly.
- Respect `CUDA_VISIBLE_DEVICES=""` for reproducibility; do not rely on GPU availability.
- When editing plan/docs, keep checklist IDs and `[ ]/[P]/[x]` markers accurate.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md (B2/B3 guidance)
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T091341Z/phase_c4d_gridsize_debug/summary.md
- specs/ptychodus_api_spec.md §4.6–§4.8
- docs/workflows/pytorch.md §§5–7, §12
- tests/torch/test_workflows_components.py:TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize

Next Up:
- 1. Phase C4.E documentation updates once B2/B3 are GREEN.
- 2. Begin refactoring CLI thin wrappers (Phase D1/D2) after parity evidence closes.
