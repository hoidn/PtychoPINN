Summary: Stage RED tests for the inference CLI thin-wrapper refactor (Phase D.C C2).
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D.C (Inference CLI thin wrapper, C2)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv (expected RED); CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py -k inference_mode -vv (expected RED)
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T115252Z/phase_d_cli_wrappers_inference_red/{pytest_cli_inference_thin_red.log,pytest_cli_shared_inference_red.log}

Do Now:
1. ADR-003-BACKEND-API C2 (thin-wrapper CLI tests) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:47 — Add `TestInferenceCLIThinWrapper` RED cases in `tests/torch/test_cli_inference_torch.py` covering helper delegation (`validate_paths`, factory call, `RawData.from_file`, `_run_inference_and_reconstruct`) plus quiet-flag behaviour; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv (expect FAIL, capture log to artifact hub).
2. ADR-003-BACKEND-API C2 (shared helper inference coverage) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:47 — Extend `tests/torch/test_cli_shared.py` with inference-mode cases (`test_build_execution_config_inference_mode_defaults`, `test_build_execution_config_inference_mode_custom_batch_size`, `test_build_execution_config_inference_mode_respects_quiet`) referencing blueprint §Test Strategy; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py -k "inference_mode or quiet" -vv (expect FAIL, capture log).
3. ADR-003-BACKEND-API C2 (plan + ledger sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:47 — Update plan row C2 to `[x]` with artifact links and note RED selectors; append Attempt entry in `docs/fix_plan.md` and drop summary stub in the new artifact directory; tests: none.

If Blocked: Record details in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T115252Z/phase_d_cli_wrappers_inference_red/blocker.md`, keep plan row C2 at `[ ]`, and log the stall in `docs/fix_plan.md` Attempts History before stopping.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:47 — Phase D.C C2 exit criteria require RED selectors before refactor begins.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/inference_refactor.md#test-strategy — Blueprint enumerates the exact helper delegation tests to author.
- specs/ptychodus_api_spec.md:180 — CLI contract mandates CONFIG-001 ordering; RED tests need to assert validate_paths/factory call sequence.
- docs/workflows/pytorch.md:344 — Execution-config helper behaviour must stay aligned; inference-mode tests safeguard quiet flag + accelerator routing.
- tests/torch/test_cli_inference_torch.py:1 — Existing GREEN coverage documents current behaviour; new tests should wrap mocks without breaking baseline cases.

How-To Map:
- Create artifact hub via `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T115252Z/phase_d_cli_wrappers_inference_red`.
- In `tests/torch/test_cli_inference_torch.py`, add a new `TestInferenceCLIThinWrapper` class that:
  - patches `ptycho_torch.cli.shared.validate_paths`, `ptycho_torch.config_factory.create_inference_payload`, `ptycho_torch.workflows.components.load_inference_bundle_torch`, `ptycho_torch.inference.RawData.from_file`, and the future `_run_inference_and_reconstruct`.
  - asserts the CLI calls helpers in the expected order by inspecting mock call arguments, raising `AssertionError` if `_run_inference_and_reconstruct` is missing (expected RED).
  - includes a quiet-mode test that confirms `enable_progress_bar` maps to quiet flag (mock print/log).
- Capture RED evidence: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T115252Z/phase_d_cli_wrappers_inference_red/pytest_cli_inference_thin_red.log` (failure expected on missing helper/validate_paths wiring).
- Update `tests/torch/test_cli_shared.py` with inference-mode helper tests per blueprint §Test Strategy (defaults vs custom batch size vs quiet). Ensure tests assert `enable_progress_bar` toggles and that `inference_batch_size` defaults to `None`.
- Run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_shared.py -k "inference_mode or quiet" -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T115252Z/phase_d_cli_wrappers_inference_red/pytest_cli_shared_inference_red.log` (expect new tests to FAIL until helpers updated).
- After logs captured, mark plan C2 row `[x]`, embed artifact file list, and add a short summary (`summary.md`) to the artifact hub referencing both failing logs and key assertions.
- Update `docs/fix_plan.md` Attempts History with the new RED evidence pointer, noting selectors and failure signatures.

Pitfalls To Avoid:
- Do not modify `ptycho_torch/inference.py` or production helpers this loop.
- Keep mocks focused—avoid patching modules that would hide missing helper calls.
- Maintain pytest-native style; no unittest.TestCase usage.
- Ensure RED logs capture the failure stack traces (no `-q` / `--maxfail` that truncates output).
- Do not delete or overwrite prior artifact directories; add new evidence only to the timestamped hub.
- Preserve CONFIG-001 references in assertions; do not skip validate_paths() ordering checks.
- Avoid editing existing passing tests except for necessary fixture reuse.
- Keep environment CPU-only (set `CUDA_VISIBLE_DEVICES=""`) to avoid accidental GPU dependence.
- Remember to restore any `sys.argv` monkeypatching within tests to prevent bleed-over.
- Update plan/ledger after logs are saved to keep traceability intact.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/inference_refactor.md#test-strategy
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:44
- ptycho_torch/inference.py:293
- tests/torch/test_cli_inference_torch.py:1
- tests/torch/test_cli_shared.py:1
- specs/ptychodus_api_spec.md:180
- docs/workflows/pytorch.md:344

Next Up: Consider ADR-003-BACKEND-API C3 (implement inference thin wrapper) once RED tests land.
