Summary: Refactor the inference CLI into a thin wrapper and drive Phase D.C C3 tests to GREEN.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D.C (Inference CLI thin wrapper, C3)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120825Z/phase_d_cli_wrappers_inference_impl/{summary.md,pytest_cli_inference_green.log,pytest_cli_integration_green.log}

Do Now:
1. ADR-003-BACKEND-API C3 (implement thin wrapper) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — Extract `_run_inference_and_reconstruct()` from the existing inline logic, update `cli_main()` to call shared helpers (`validate_paths`, `build_execution_config_from_args(mode='inference')`, `resolve_accelerator`), and delegate RawData loading plus helper invocation before saving reconstructions; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv (expect GREEN, capture log to artifact hub).
2. ADR-003-BACKEND-API C3 (integration guard) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — Run the PyTorch integration workflow to confirm CLI changes preserve end-to-end behaviour; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv (capture log).
3. ADR-003-BACKEND-API C3 (plan + ledger sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — Mark plan row C3 `[x]`, drop `summary.md` with test outcomes, and append Attempt entry in `docs/fix_plan.md` referencing the GREEN logs; tests: none.

If Blocked: Capture failure details (stack trace, command, inputs) in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120825Z/phase_d_cli_wrappers_inference_impl/blocker.md`, keep plan C3 at `[ ]`, and document the stall in `docs/fix_plan.md` Attempts History before stopping.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — C3 exit criteria require helper extraction, shared helper delegation, and targeted pytest validation.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/inference_refactor.md#inference-orchestration-refactor — Blueprint spells out the helper structure and delegation order for C3.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T115252Z/phase_d_cli_wrappers_inference_red/summary.md — RED logs document the current failure signatures each refactor step must resolve.
- specs/ptychodus_api_spec.md:170 — CONFIG-001 ordering and bundle loading contract must remain intact after the refactor.
- docs/workflows/pytorch.md:318 — CLI documentation describes expected flags and quiet-mode semantics that shared helpers enforce.

How-To Map:
- Create artifact hub: `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120825Z/phase_d_cli_wrappers_inference_impl`.
- In `ptycho_torch/inference.py`, move the inference loop (current lines ~520-660) into a private helper `_run_inference_and_reconstruct(model, raw_data, config, execution_config, device, quiet=False)` that returns `(amplitude, phase)`; ensure helper lives alongside `save_individual_reconstructions`.
- Replace inline validation in `cli_main()` with calls to `validate_paths(train_file=None, test_file=..., output_dir=...)` and `build_execution_config_from_args(args, mode='inference')`; use `resolve_accelerator()` for `--device` compatibility and pass `quiet` flag through.
- After factory payload + bundle loader complete, delegate to the new helper and capture its return values before calling `save_individual_reconstructions`.
- Preserve legacy MLflow path (`if __name__ == '__main__'` guard) untouched.
- Run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120825Z/phase_d_cli_wrappers_inference_impl/pytest_cli_inference_green.log`.
- Follow with `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120825Z/phase_d_cli_wrappers_inference_impl/pytest_cli_integration_green.log`.
- Summarize outcomes (helper extractions, test results, any follow-up) in `summary.md` within the artifact hub before updating plan/ledger rows.

Pitfalls To Avoid:
- Do not drop quiet-mode behaviour; shared helpers must remain the single source of progress-bar toggling.
- Keep `_run_inference_and_reconstruct` importable from `ptycho_torch.inference` so tests can patch it.
- Ensure helper returns numpy arrays, not torch tensors, so `save_individual_reconstructions` keeps working.
- Maintain CONFIG-001 ordering: `validate_paths` → factory → bundle loader → RawData load → helper.
- Retain legacy MLflow CLI path logic and argument parsing untouched.
- Avoid introducing GPU-only defaults; continue honouring CPU fallback when accelerator is unavailable.
- Do not run full pytest suites—stick to mapped selectors.
- Stop if integration test fails: capture log, do not iterate blindly.
- Leave artifact logs in the timestamped directory; no output at repo root.

Pointers:
- ptycho_torch/inference.py:293
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/inference_refactor.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T115252Z/phase_d_cli_wrappers_inference_red/summary.md
- tests/torch/test_cli_inference_torch.py:203
- tests/torch/test_integration_workflow_torch.py:1
- specs/ptychodus_api_spec.md:170

Next Up: If C3 lands early, stage Phase D.C C4 (docs updates) per plan once Do Now is complete.
