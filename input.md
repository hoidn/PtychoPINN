Summary: Diagnose why inference sees dicts instead of Lightning modules when loading PyTorch bundles and patch the pipeline with TDD.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4.D (bundle loader unblock)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T093500Z/phase_c4d_bundle_probe/{pytest_integration_red.log,bundle_introspection.md,pytest_integration_green.log}

Do Now:
1. ADR-003-BACKEND-API C4.D.B4 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — ensure the integration fixture exists (run `python scripts/tools/make_pytorch_integration_fixture.py --source <canonical_dataset> --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --subset-size 64` if the NPZ is missing), then reproduce the failure via `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T093500Z/phase_c4d_bundle_probe/pytest_integration_red.log`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
2. ADR-003-BACKEND-API C4.D.B4 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — author a regression test that loads the freshly written `training_outputs/wts.h5.zip` via `load_inference_bundle_torch` and asserts both `models_dict['diffraction_to_obj']` and `models_dict['autoencoder']` are `torch.nn.Module` instances supporting `.eval()`. Capture the RED run with `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T093500Z/phase_c4d_bundle_probe/pytest_bundle_loader_red.log`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv
3. ADR-003-BACKEND-API C4.D.B4 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md — implement the fix so `load_inference_bundle_torch` (and the underlying persistence path) return Lightning modules, rerun the new regression test until GREEN, and confirm the integration selector now reaches inference without raising `AttributeError`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_bundle_loader_returns_modules -vv && CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv

If Blocked: Record failing command output and bundle inspection notes in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T093500Z/phase_c4d_bundle_probe/blocker.md`, update `phase_c4d_blockers/plan.md` (row B4) back to `[P]`, and summarize the obstacle in docs/fix_plan.md Attempts.

Priorities & Rationale:
- `phase_c4d_blockers/plan.md` row B4 now tracks bundle-type verification; resolving it is prerequisite for closing C4.D.
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/pytest_integration_phase_a.log` shows the current `AttributeError: 'dict' object has no attribute 'eval'` in inference.
- `specs/ptychodus_api_spec.md` §4.6–§4.8 mandate dual-model bundle persistence and nn.Module return types.
- `docs/workflows/pytorch.md` §§5–7 document bundle expectations and CLI workflow; parity requires honoring those contracts.
- Implementation ledger (`plans/active/ADR-003-BACKEND-API/implementation.md` Phase C4 rows) references this blocker; clearing it unblocks Phase D CLI refactors.

How-To Map:
- Store all logs and introspection notes under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T093500Z/phase_c4d_bundle_probe/`; include `bundle_introspection.md` describing module types and params snapshot keys.
- After running the integration selector, copy `training_outputs/wts.h5.zip` into the artifact directory before rerunning tests.
- Use a short Python snippet (documented in `bundle_introspection.md`) to `torch.load` each `model.pth` and print `type(...)` plus available methods; cite spec §4.6 in commentary.
- When authoring the new regression test, reuse pytest fixtures (`tmp_path`, `cuda_cpu_env`) to keep runtime <20s; the test should intentionally fail prior to the fix.
- After implementing the fix, rerun the regression test and the integration selector, capturing `pytest_bundle_loader_green.log` and `pytest_integration_green.log`.

Pitfalls To Avoid:
- Do not delete existing evidence hubs (`2025-10-20T083500Z/phase_c4d_blockers`); append new logs under the new timestamp.
- Ensure fixture generation writes to `tests/fixtures/...` only once; avoid committing large NPZs if the generator already produced them.
- Keep CONFIG-001 ordering intact—`update_legacy_dict` must execute before any RawData or model reconstruction.
- Handle both `'diffraction_to_obj'` and `'autoencoder'` in the regression test; spec requires dual-model support.
- Avoid reverting or editing TensorFlow persistence code; changes belong in the PyTorch stack.
- Maintain device neutrality—run tests with `CUDA_VISIBLE_DEVICES=""` exactly as specified.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T090900Z/debug/summary.md
- specs/ptychodus_api_spec.md:4.6-4.8
- docs/workflows/pytorch.md:§5-§7
- tests/torch/test_integration_workflow_torch.py:185

Next Up:
- After the bundle loader returns proper modules, proceed with `phase_c4d_blockers` Phase B gridsize parity tasks.
