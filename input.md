Summary: Make the PyTorch training CLI emit the spec-required wts.h5.zip bundle while keeping inference CLI functional so the integration workflow passes.
Mode: Parity
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_bundle_persistence -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T060955Z/phase_c4_cli_integration_debug/{triage.md,pytest_cli_train_bundle_red.log,pytest_cli_train_bundle_green.log,pytest_integration.log,pytest_cli_inference.log}

Do Now:
1. ADR-003-BACKEND-API C4.D3 bundle TDD @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — add pytest `test_bundle_persistence` under `TestExecutionConfigCLI` that monkeypatches `ptycho_torch.train.save_torch_bundle` and asserts the new CLI path invokes it with both autoencoder/diffraction_to_obj keys; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_bundle_persistence -vv.
2. ADR-003-BACKEND-API C4.D3 workflow persistence implementation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — route the new training CLI through `run_cdi_example_torch`, update `_train_with_lightning` to emit a dual-model bundle (diffraction_to_obj module + autoencoder sentinel), and ensure `save_torch_bundle` writes `{output_dir}/wts.h5.zip`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI::test_bundle_persistence -vv.
3. ADR-003-BACKEND-API C4.D3 integration rerun @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — rerun the PyTorch integration selector and capture a fresh log showing training+inference succeed with the bundle present; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv.
4. ADR-003-BACKEND-API C4.D3 inference guard @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — keep CLI execution-config tests green while adding a temporary fallback so inference still loads `last.ckpt` if bundle loading raises NotImplementedError; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv.
5. ADR-003-BACKEND-API C4.F wrap-up @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — update plan row C4.D3, extend summary with bundle findings, and log Attempt #25 in docs/fix_plan.md referencing the new artifacts; tests: none.

If Blocked: tee the failing selector output to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T060955Z/phase_c4_cli_integration_debug/blocker.log`, note whether the CLI still calls legacy `main()` or `save_torch_bundle` rejects the models dict, set C4.D3 back to `[P]` with the blocker note, then stop.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md (row C4.D3) — exit criteria now hinge on producing the bundle while keeping regression green.
- specs/ptychodus_api_spec.md:149 — backend contract mandates `MODEL_FILE_NAME = 'wts.h5.zip'`.
- docs/workflows/pytorch.md:125 — training workflow must persist `{output_dir}/wts.h5.zip`.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T060955Z/phase_c4_cli_integration_debug/triage.md — documents the bundle gap and workflow fallback expectations.
- tests/torch/test_model_manager.py:150 — defines the dual-model requirement (`autoencoder`, `diffraction_to_obj`) enforced by `save_torch_bundle`.

How-To Map:
- RED test: monkeypatch `ptycho_torch.train.save_torch_bundle` and `ptycho_torch.train.run_cdi_example_torch` so the CLI runs fast; expect the mock to see a models dict with both keys. Capture RED log as `pytest_cli_train_bundle_red.log`.
- Implementation: in `ptycho_torch/train.py` branch the new interface to load `RawData` (use `payload.tf_training_config` & payload execution config) and call `run_cdi_example_torch(train_data, test_data, payload.tf_training_config, do_stitching=False, output_dir=output_dir)`. Ensure `_train_with_lightning` returns `{'diffraction_to_obj': model, 'autoencoder': {'_sentinel': 'autoencoder'}}` so `save_torch_bundle` accepts the payload. After the call, surface the CLI status messages and skip the legacy `main()` invocations. Store the GREEN log as `pytest_cli_train_bundle_green.log`.
- Integration: rerun the selector and tee to `pytest_integration.log`. Expect wts.h5.zip alongside checkpoints; if inference still needs the old checkpoint, catch `NotImplementedError` inside CLI and call the legacy loader path with a warning so the test completes.
- Inference CLI fallback: wrap the bundle-loading block in try/except, logging the bundle path when successful and falling back to the prior checkpoint loader (without removing spec-compliant code) when the bundle loader raises `NotImplementedError`. Capture test output in `pytest_cli_inference.log`.
- Documentation: once tests pass, mark C4.D3 `[x]` in plan, append the key decisions to `summary.md`, and add Attempt #25 to `docs/fix_plan.md` with artifact links.

Pitfalls To Avoid:
- Do not remove the factory or CONFIG-001 calls when rerouting the CLI.
- Keep `save_torch_bundle` invocation single-shot; no duplicate archives or legacy writes.
- Preserve tqdm/logging behaviour (`--disable-mlflow` should still suppress progress bars).
- Avoid importing torch-heavy modules at module scope in tests; stick to mocks.
- Maintain ASCII formatting and existing import ordering in modified files.
- Do not delete historical logs under other timestamped directories.
- Ensure fallback code only triggers on `NotImplementedError`, not real IO errors.
- Leave legacy CLI interface untouched (only adjust new interface branch).
- Update `_train_with_lightning` without breaking existing unit tests (respect execution config fields).
- Capture every test run with `tee` into the specified artifact directory.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:94
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T060955Z/phase_c4_cli_integration_debug/triage.md
- ptycho_torch/train.py:360
- ptycho_torch/workflows/components.py:470
- ptycho_torch/model_manager.py:85
- specs/ptychodus_api_spec.md:149

Next Up: 1) Implement real `load_torch_bundle` reconstruction so the inference CLI can drop the checkpoint fallback; 2) Phase C4.E documentation updates once persistence is green.
