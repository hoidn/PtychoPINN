Summary: Complete the backend dispatcher so config.backend actually routes between TensorFlow and PyTorch while keeping parity guarantees.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 Phase E1.C
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_backend_selection.py -vv; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T190900Z/{phase_e_backend_green.md,pytest_backend_selection.log,pytest_workflows_backend.log}
Do Now:
- INTEGRATE-PYTORCH-001 Phase E1.C — E1.C3+E1.C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_e_backend_design.md (tests: none): add a torch-optional backend dispatcher (new module under ptycho/workflows or reconstructor shim) that inspects Training/InferenceConfig.backend, calls update_legacy_dict(params.cfg, config), imports the correct workflow (`ptycho.workflows.components.run_cdi_example` vs `ptycho_torch.workflows.components.run_cdi_example_torch`), records the active backend in results, and raises a RuntimeError with installation guidance when PyTorch is unavailable; update tests/torch/test_backend_selection.py to remove xfails and verify dispatcher, import guarding, and CONFIG-001 behavior.
- INTEGRATE-PYTORCH-001 Phase E1.C — E1.C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_e_backend_design.md (tests: pytest tests/torch/test_backend_selection.py -vv; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv): capture logs to the new report directory, then summarize the implementation decisions, remaining risks, and any open follow-ups in phase_e_backend_green.md alongside the command transcripts.
If Blocked: Archive failing pytest output and a short diagnostic note in phase_e_backend_green.md, flag the blocker in docs/fix_plan.md Attempts History, and leave TODO markers in the dispatcher indicating the unmet dependency.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:127 — reconstructor contract requires dual-backend selection with CONFIG-001 enforced before dispatch.
- plans/active/INTEGRATE-PYTORCH-001/phase_e_backend_design.md:45 — defines acceptance criteria for E1.C3/E1.C4 (dispatcher, error handling, logging, results metadata).
- tests/torch/test_backend_selection.py:128 — remaining xfail cases encode dispatcher and import-guard requirements that must now go green.
- ptycho/workflows/components.py:705 — TensorFlow path already calls update_legacy_dict; dispatcher must preserve parity and instrumentation when routing there.
- ptycho_torch/workflows/components.py:94 — PyTorch orchestration entry to invoke once backend flag selects torch.
How-To Map:
- Implement dispatcher: add `ptycho/workflows/backend_selector.py` (or comparable module) exposing `run_cdi_example_with_backend`, `train_cdi_model_with_backend`, and `load_inference_bundle_with_backend`; keep imports local and guarded, call `update_legacy_dict` before selecting backend, and populate `results['backend']` prior to return.
- Error handling: wrap PyTorch import with try/except ImportError and raise RuntimeError mentioning "PyTorch backend selected" and "pip install .[torch]" when unavailable.
- Tests: adjust backend selection tests to import the new dispatcher, mock workflow functions/torch availability, assert update_legacy_dict spy/count, assert RuntimeError message, and ensure configs remain torch-optional; rerun `pytest tests/torch/test_backend_selection.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T190900Z/pytest_backend_selection.log`.
- Regression guard: verify persistence wiring remains intact by running `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T190900Z/pytest_workflows_backend.log`.
- Documentation: record the dispatcher design, logging decisions, and any follow-ups in `phase_e_backend_green.md`, noting how Ptychodus should import the new helper.
Pitfalls To Avoid:
- Do not add unconditional `import torch`; guard PyTorch imports inside dispatcher branches.
- Keep default backend `'tensorflow'`; do not flip defaults or mutate configs in-place.
- Ensure `update_legacy_dict` still runs before any backend-specific work to satisfy CONFIG-001.
- Avoid modifying protected core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Preserve existing torch-optional skip logic in tests by staying within the whitelist in tests/conftest.py.
- Make dispatcher logging deterministic; no environment-dependent strings in assertions.
- Return the same `(amp, phase, results)` tuple structure that callers expect; include backend metadata without altering keys that tests rely on.
- When mocking imports in tests, restore sys.modules to prevent leakage into later tests.
- Capture artifact logs exactly under the specified report directory; no stray files at repo root.
- Document any deferred PyTorch functionality instead of silently passing.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_e_backend_design.md
- specs/ptychodus_api_spec.md:127
- ptycho/workflows/components.py:705
- ptycho_torch/workflows/components.py:94
- tests/torch/test_backend_selection.py:128
- tests/torch/test_workflows_components.py:487
Next Up: Phase E2.A fixture alignment once backend selection is green.
