Summary: Align inference CLI thin-wrapper tests with the new delegation flow and restage GREEN evidence.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D.C (Inference CLI thin wrapper, C3)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup/{summary.md,pytest_cli_inference_green.log,pytest_cli_integration_green.log}

Do Now:
1. ADR-003-BACKEND-API C3 (fix thin-wrapper tests) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — Update `tests/torch/test_cli_inference_torch.py` delegation cases so they inspect `mock_validate_paths.call_args.kwargs` and seed the mocked bundle loader with `{'diffraction_to_obj': MagicMock()}`; keep assertions on call ordering/intents. tests: none.
2. ADR-003-BACKEND-API C3 (CLI selector GREEN) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — Run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv` and tee output to `pytest_cli_inference_green.log` inside the artifact hub.
3. ADR-003-BACKEND-API C3 (integration guard) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — Run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv` and capture the log as `pytest_cli_integration_green.log` in the same artifact directory.
4. ADR-003-BACKEND-API C3 (plan + ledger sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — Move `train_debug.log` into the artifact hub, refresh `phase_d_cli_wrappers/plan.md` C3 to `[x]`, append summary + Attempt entry, and record outcomes in `docs/fix_plan.md`. tests: none.

If Blocked: Capture failing selector output and a short blocker note to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup/blocker.md`, leave plan C3 at `[P]`, and document the stall in `docs/fix_plan.md` before stopping.

Priorities & Rationale:
- tests/torch/test_cli_inference_torch.py:262 — Delegation tests encode thin-wrapper contract; they must reflect keyword invocation + bundle contract for CONFIG-001 compliance.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — C3 exit criteria demand GREEN selectors and artifact hygiene before advancing to C4.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120825Z/phase_d_cli_wrappers_inference_impl/pytest_cli_inference_green.log — Existing log shows two failures; new run should supersede it in the follow-up hub.
- specs/ptychodus_api_spec.md:170 — CONFIG-001 ordering (validate_paths → factory → bundle → data load) informs the assertions you are adjusting.
- docs/workflows/pytorch.md:318 — CLI documentation commits to shared helper delegation (`build_execution_config_from_args`, `resolve_accelerator`).

How-To Map:
- mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup
- After editing tests: git diff tests/torch/test_cli_inference_torch.py
- CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup/pytest_cli_inference_green.log
- CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup/pytest_cli_integration_green.log
- mv train_debug.log plans/active/ADR-003-BACKEND-API/reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup/train_debug.log
- Update `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md` and `summary.md` with GREEN status + log references, then append Attempt #50 in docs/fix_plan.md referencing the new artifact paths.
- Commit message suggestion: `ADR-003 Phase D.C C3: align inference thin-wrapper tests` (include tests run).

Pitfalls To Avoid:
- Do not revert earlier Phase D rows or reintroduce inline validation in `ptycho_torch/inference.py`.
- Keep new tests strictly pytest-style; avoid unittest.TestCase patterns.
- Preserve CONFIG-001 ordering inside assertions; don’t mock out factory invocation order.
- Ensure the mocked bundle still surfaces `'diffraction_to_obj'` so helper path executes before you assert call order.
- Store all logs under the timestamped artifact directory; leave no outputs at repo root.
- Keep accelerator mapping assertions consistent with docs/workflows/pytorch.md §12; no new device strings.
- Skip unrelated test suites; only run the mapped selectors.
- After moving `train_debug.log`, confirm repo root is clean via `ls` before committing.
- Don’t edit production helper signatures; adjust tests to the existing keyword pattern instead.
- Capture stderr/stdout in logs; avoid redirecting to /dev/null.

Pointers:
- ptycho_torch/inference.py:510
- tests/torch/test_cli_inference_torch.py:262
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/summary.md:37
- specs/ptychodus_api_spec.md:170

Next Up: Phase D.C C4 (docs + hygiene) once C3 is GREEN.
