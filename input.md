
Summary: Author torch-optional red tests for PyTorch persistence and workflow regression (Phase D4.B)
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase D4.B regression tests
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_round_trip_returns_model_stub -vv; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_load_inference_bundle_handles_bundle -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T112849Z/{phase_d4_red_persistence.log,phase_d4_red_workflows.log,phase_d4_red_summary.md}
Do Now:
1. INTEGRATE-PYTORCH-001 — D4.B1 persistence red tests @ plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:38 — tests: pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_round_trip_returns_model_stub -vv — Extend TestLoadTorchBundle (tests/torch/test_model_manager.py:327) with a torch-optional round-trip test that saves sentinel models via save_torch_bundle and expects load_torch_bundle to return the saved sentinel/params without raising NotImplementedError; let the current NotImplementedError failure stand and capture the log to phase_d4_red_persistence.log.
2. INTEGRATE-PYTORCH-001 — D4.B2 orchestration red tests @ plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:39 — tests: pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_load_inference_bundle_handles_bundle -vv — Add new tests under TestWorkflowsComponentsRun (tests/torch/test_workflows_components.py:340) that monkeypatch save_torch_bundle / load_torch_bundle to sentinel stubs, assert run_cdi_example_torch persists models when config.output_dir is set, and that load_inference_bundle_torch delegates to the model_manager shim; both should currently fail because orchestration lacks persistence wiring—record their pytest output to phase_d4_red_workflows.log.
3. INTEGRATE-PYTORCH-001 — D4.B3 red-phase summary @ plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:40 — tests: none — Write phase_d4_red_summary.md in the artifact directory summarizing the failing assertions, selectors, environment variables used, and follow-up actions; update phase_d4_regression.md to set D4.B1/D4.B2 state to [P] with guidance notes and log the artifact path, then add the loop attempt to docs/fix_plan.md Attempts History.
If Blocked: Document the blocker in phase_d4_red_summary.md, keep D4.B rows at [ ] with rationale, and note the issue in docs/fix_plan.md; capture any error output under the same artifact timestamp for traceability.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:31 — Phase D4.B explicitly calls for torch-optional failing regression tests before implementation begins.
- specs/ptychodus_api_spec.md:192 — Model persistence contract requires loader/saver parity that the new red tests must encode.
- docs/findings.md:9 — CONFIG-001 mandates params.cfg restoration, so regression tests need to assert this behavior during persistence/orchestration flows.
- tests/torch/test_model_manager.py:327 — Existing load bundle tests stop at params restoration; new tests extend coverage to full round-trip expectations.
- tests/torch/test_workflows_components.py:340 — Current workflow tests cover update_legacy_dict and training delegation but lack persistence/regression guarantees.
How-To Map:
- Export CUDA_VISIBLE_DEVICES="" and MLFLOW_TRACKING_URI=memory before running selectors to keep runs CPU-only and side-effect free.
- Use existing fixtures (dummy_torch_models, minimal_training_config) and sentinel dicts so tests execute without torch; keep imports guarded via TORCH_AVAILABLE checks.
- Capture logs with tee, e.g., `pytest ... | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T112849Z/phase_d4_red_persistence.log` and repeat for workflows.
- When authoring assertions, give clear failure messages (e.g., "load_torch_bundle should return saved sentinel model instead of raising NotImplementedError") to document desired behavior.
- After tests fail, update phase_d4_regression.md D4.B rows with status [P] and note the new test names/selectors; reference the log files in the How/Why column.
- Append Attempt #55 in docs/fix_plan.md for INTEGRATE-PYTORCH-001 noting the red tests, selectors, and artifact directory.
Pitfalls To Avoid:
- Do not touch production implementations (`ptycho_torch/model_manager.py`, `ptycho_torch/workflows/components.py`) in this loop—tests only.
- Keep new tests torch-optional; avoid hard importing torch or relying on GPU-only behavior.
- Ensure artifact directory matches the timestamp exactly; no stray files at repo root.
- Don’t mark D4.B tasks complete; they stay red-phase until implementations land.
- Avoid conflating TEST-PYTORCH-001 integration scope—document blockers rather than building subprocess harnesses now.
- Leave MLflow disabled (`MLFLOW_TRACKING_URI=memory`) to prevent network calls during pytest execution.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:31
- specs/ptychodus_api_spec.md:192
- docs/findings.md:9
- tests/torch/test_model_manager.py:327
- tests/torch/test_workflows_components.py:340
Next Up: Turn the new regression tests green (Phase D4.C) once persistence and orchestration wiring is implemented.
