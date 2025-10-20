Summary: Restore PyTorch Lightning dataloader parity so compute_loss stops crashing during the integration workflow.
Mode: Parity
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration (C4.D3)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_dataloader_tensor_dict_structure -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T080500Z/phase_c4_cli_integration_debug/{pytest_dataloader_red.log,pytest_dataloader_green.log,pytest_integration.log}
Do Now:
1. ADR-003-BACKEND-API C4.D3 dataloader RED test @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — add `test_lightning_dataloader_tensor_dict_structure` in `tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining`, then run CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_dataloader_tensor_dict_structure -vv (expect FAIL, tee to pytest_dataloader_red.log).
2. ADR-003-BACKEND-API C4.D3 dataloader implementation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — refactor `_build_lightning_dataloaders` to emit TensorDict-style batches (reuse `TensorDictDataLoader`/`Collate_Lightning` from `ptycho_torch.dataloader`) so compute_loss sees `images/coords_relative/rms_scaling_constant/physics_scaling_constant`, then rerun CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_dataloader_tensor_dict_structure -vv (tee to pytest_dataloader_green.log).
3. ADR-003-BACKEND-API C4.D3 integration validation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — rerun CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv (tee to pytest_integration.log) and confirm the workflow completes with bundle + checkpoint artifacts present.
4. ADR-003-BACKEND-API C4.D3 ledger wrap-up @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — mark C4.D3 `[x]`, update `summary.md` with dataloader parity notes, and log Attempt #26 follow-up in docs/fix_plan.md referencing the new artifact directory; tests: none.
If Blocked: capture failing selector output to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T080500Z/phase_c4_cli_integration_debug/blocker.log`, note whether batches still lack TensorDict keys or downstream persistence fails, update C4.D3 back to `[P]` with the blocker reason, then stop.
Priorities & Rationale:
- C4.D3 plan row (plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md) — integration remains red until Lightning batches mirror TensorFlow structure.
- dataloader_summary.md (plans/active/ADR-003-BACKEND-API/reports/2025-10-20T073500Z/phase_c4_cli_integration_debug/dataloader_summary.md) — documents root cause and expected batch keys.
- specs/ptychodus_api_spec.md:§4.6 — dual-model bundle contract enforced by integration test; pipeline must deliver proper inputs to Lightning before persistence.
- docs/workflows/pytorch.md:§12 — describes execution config + loader parity requirements for PyTorch backend.
- ptycho_torch/train_utils.py:217-308 — legacy DataModule reference for TensorDictDataLoader/Collate_Lightning usage.
How-To Map:
- RED run: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_dataloader_tensor_dict_structure -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T080500Z/phase_c4_cli_integration_debug/pytest_dataloader_red.log`
- GREEN run: after refactor, repeat the same command and tee to `pytest_dataloader_green.log`.
- Integration: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T080500Z/phase_c4_cli_integration_debug/pytest_integration.log`
- Use `TensorDictDataLoader` and `Collate_Lightning` from `ptycho_torch.dataloader` to build batches; ensure probe/scaling tensors propagate so compute_loss indexes `[1]` and `[2]` successfully.
Pitfalls To Avoid:
- Do not drop existing execution-config threading when editing `_build_lightning_dataloaders` — keep deterministic seeding intact.
- Preserve CONFIG-001 sequencing: `_ensure_container` must still be called before loader construction; no data access before params.cfg updated.
- Avoid reintroducing torch-optional code paths (POLICY-001). Batches should stay on CPU but must use torch tensors.
- Keep new pytest test purely in pytest style (no unittest mix).
- Do not modify bundle persistence logic introduced in Attempt #25; focus only on dataloader structure.
- Ensure new dataset helper respects gridsize>1 channel counts (use container attributes so tests cover both 1×1 and grouped cases).
- Capture all command output via tee into the designated artifact directory.
- Leave `tests/fixtures/pytorch_integration/minimal_dataset_v1.json` untouched; fixture generation lives under TEST-PYTORCH-001.
- No logging noise at repo root; keep new notes under the timestamped report directory.
Pointers:
- ptycho_torch/workflows/components.py:260-352 — current `_build_lightning_dataloaders` stub.
- ptycho_torch/dataloader.py:430-646 — reference implementation for TensorDictDataLoader + Collate_Lightning.
- ptycho_torch/model.py:1115-1140 — `compute_loss` expectations for batch layout.
- tests/torch/test_workflows_components.py:200-520 — fixture definitions and training tests to extend.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T073500Z/phase_c4_cli_integration_debug/dataloader_summary.md.
Next Up: (1) Phase C4.F documentation/ledger wrap-up once integration stays green; (2) revisit CLI manual smoke test (plan row C4.D4) if time permits.
