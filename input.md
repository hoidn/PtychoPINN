Summary: Turn Phase D4 regression tests green by wiring PyTorch persistence + loader shims
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase D4.C regression green
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_load_inference_bundle_handles_bundle -vv; pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_round_trip_returns_model_stub -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T121930Z/{phase_d4_green_persistence.log,phase_d4_green_workflows.log,phase_d4_handoff.md}
Do Now:
1. INTEGRATE-PYTORCH-001 — D4.C1 persistence fixes @ plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:66 — tests: pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv — Update `ptycho_torch/workflows/components.py` to import `save_torch_bundle`, call it inside `run_cdi_example_torch` when `config.output_dir` is set, and ensure the call receives the dual-model dict from `train_results`; capture pytest output to `phase_d4_green_persistence.log`.
2. INTEGRATE-PYTORCH-001 — D4.C2 loader delegation @ plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:72 — tests: pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_load_inference_bundle_handles_bundle -vv — Implement `load_inference_bundle_torch` so it imports/delegates to `load_torch_bundle`, restores params.cfg (CONFIG-001), and returns `(models_dict, params_dict)`; record the test run in `phase_d4_green_workflows.log`.
3. INTEGRATE-PYTORCH-001 — D4.C3 regression sweep & handoff @ plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:78 — tests: pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_round_trip_returns_model_stub -vv pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_load_inference_bundle_handles_bundle -vv — Finish `load_torch_bundle` model reconstruction logic (or document why the XFAIL remains), rerun all three selectors, and write `phase_d4_handoff.md` summarizing results, remaining gaps (if any), and TEST-PYTORCH-001 activation guidance.
If Blocked: Capture failing logs in the same artifact directory, update D4.C rows with `[P]` + blocker notes, and log the issue + evidence path in docs/fix_plan.md before exiting.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:192 — Reconstructor must persist dual-model bundles and reload them transparently; D4.C implements that contract for PyTorch.
- docs/findings.md:9 — CONFIG-001 enforces params.cfg restoration, so loader wiring must update params before returning.
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:70 — Phase D4 is now the active blocker; A/B done, C outstanding.
- plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:63 — Checklist enumerates C1–C3 deliverables that turn the red tests green.
- tests/torch/test_workflows_components.py:512 — Regression tests already encode desired orchestration behavior; wiring code must satisfy them.
- tests/torch/test_model_manager.py:620 — Loader test documents round-trip expectations and will remain XFAIL until reconstruction logic lands or is explicitly deferred.
How-To Map:
- Export `CUDA_VISIBLE_DEVICES=""` and `MLFLOW_TRACKING_URI=memory` before running selectors; use `pytest … | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T121930Z/phase_d4_green_*.log` for traceability.
- In `run_cdi_example_torch`, only call `save_torch_bundle` when `train_results` exposes a `models` dict and `config.output_dir` is truthy; build archive path via `Path(config.output_dir) / "wts.h5"`.
- Implement `load_inference_bundle_torch(bundle_dir)` by importing `load_torch_bundle`, calling it with `Path(bundle_dir) / "wts.h5"`, and returning the tuple; keep torch-optional fallbacks intact.
- Extend `load_torch_bundle` to detect sentinel dictionaries (dill vs torch.save), load state_dicts when torch available, and return `(model_or_sentinel, params_dict)` after params restoration; document any partial implementation in the summary if full model reconstruction remains deferred.
- Store green-phase summary + any diffs (e.g., archive tree listing) in `phase_d4_handoff.md` with explicit references to updated plan rows and remaining risks.
Pitfalls To Avoid:
- Do not bypass CONFIG-001; params.cfg must be updated before any model reconstruction returns.
- Keep modules torch-optional—guard imports and support sentinel models when torch is missing.
- Avoid duplicating archive filenames; reuse `wts.h5` convention to match TensorFlow parity.
- Ensure new logic writes no artifacts outside `config.output_dir`; tests expect pure function behavior.
- Preserve existing decision notes in plan docs; append state changes instead of overwriting history.
- If loader implementation stays partial, mark `TestLoadTorchBundle::test_load_round_trip_returns_model_stub` with a clear justification in the handoff summary.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:63
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:70
- specs/ptychodus_api_spec.md:192
- docs/findings.md:9
- tests/torch/test_workflows_components.py:512
- ptycho_torch/workflows/components.py:150
- ptycho_torch/model_manager.py:187
Next Up: Once D4.C is green, prep TEST-PYTORCH-001 activation (Phase E1–E2) with subprocess harness.
