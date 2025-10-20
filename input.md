Summary: TDD the PyTorch bundle loader so the integration test clears the load_torch_bundle blocker.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4.D (bundle loader unblock)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_reconstructs_models_from_bundle -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/{pytest_load_bundle_red.log,pytest_load_bundle_green.log,pytest_integration_phase_a.log}

Do Now:
1. ADR-003-BACKEND-API C4.D.A1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — author failing `TestLoadTorchBundle::test_reconstructs_models_from_bundle`, run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_reconstructs_models_from_bundle -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/pytest_load_bundle_red.log`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_reconstructs_models_from_bundle -vv
2. ADR-003-BACKEND-API C4.D.A2+A3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — implement bundle loader helpers, rerun the selector for GREEN and capture the integration log via `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_reconstructs_models_from_bundle -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/pytest_load_bundle_green.log` then `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/pytest_integration_phase_a.log`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_reconstructs_models_from_bundle -vv && CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv

If Blocked: Capture stdout/stderr to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/blocker.log`, mark A1/A2 back to `[P]`, and summarize the obstacle in docs/fix_plan.md Attempts.

Priorities & Rationale:
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md` A1–A3 define the minimum to unblock Phase C4.D3.
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/pytest_integration.log` shows the current `NotImplementedError` we must eliminate.
- `specs/ptychodus_api_spec.md` §4.6–§4.8 require wts.h5.zip loading via canonical bundle APIs.
- `docs/findings.md` `CONFIG-001` demands the loader repopulate `params.cfg` before inference runs.
- Implementation ledger (`plans/active/ADR-003-BACKEND-API/implementation.md`) now references this blocker plan; closing A1–A3 is prerequisite for Phase B gridsize work.

How-To Map:
- Mirror the TensorFlow loader contract: use bundle metadata from `tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_bundle_persistence` as a reference and call `save_torch_bundle` within the new test to generate a temporary archive under `tmp_path`.
- Seed the test bundle with lightweight `torch.nn.Identity()` modules for `diffraction_to_obj` and `autoencoder`, plus serialized `TrainingConfig` via `dataclasses.asdict`.
- New loader helper should rebuild `PtychoPINNLightningModule` (or appropriate model) using `ModelConfig.gridsize`; add `create_torch_model_with_gridsize()` next to existing factory wiring inside `ptycho_torch/model_manager.py`.
- After implementation, rerun the targeted selector followed by the integration selector, piping each through `tee` into the plan hub filenames listed above.
- Keep `CUDA_VISIBLE_DEVICES=""` in the environment and reuse `tests/fixtures/pytorch_integration/minimal_dataset_v1.json` metadata when verifying integration output paths.

Pitfalls To Avoid:
- Do not bypass the RED→GREEN cycle; capture the failing log before implementing the loader.
- Avoid writing artefacts outside the designated 2025-10-20T083500Z directory.
- Do not regress `_train_with_lightning` dual-model return contract (`{'diffraction_to_obj', 'autoencoder'}`) validated by `test_bundle_persistence`.
- Keep CONFIG-001 ordering intact—bridge params before instantiating models in the loader.
- Refrain from editing TensorFlow workflow files; changes belong in `ptycho_torch/` only for this loop.
- Do not delete or overwrite prior evidence (070610Z/081500Z hubs); add new files instead.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md
- specs/ptychodus_api_spec.md:4.6
- ptycho_torch/model_manager.py:200
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/summary.md
- docs/findings.md:CONFIG-001

Next Up:
- Phase B of the blocker plan (gridsize/channel parity) once the bundle loader is GREEN.
