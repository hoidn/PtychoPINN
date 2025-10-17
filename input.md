Summary: Implement the Phase E1 backend selector using the new blueprint so PyTorch can be chosen without breaking TensorFlow defaults.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 Phase E1.C
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_backend_selection.py -vv; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T181500Z/{phase_e_backend_impl.md,pytest_backend_selection.log,pytest_workflows_backend.log}
Do Now:
- INTEGRATE-PYTORCH-001 Phase E1.C — E1.C1+E1.C2 @ plans/active/INTEGRATE-PYTORCH-001/phase_e_backend_design.md (tests: none): add `backend` default to Training/Inference dataclasses, propagate through `dataclass_to_legacy_dict`, and make `config_bridge` return `backend='pytorch'` when translating.
- INTEGRATE-PYTORCH-001 Phase E1.C — E1.C3 @ plans/active/INTEGRATE-PYTORCH-001/phase_e_backend_design.md (tests: none): implement Ptychodus dispatcher/import guard so config.backend selects the correct workflow, log the backend, and embed it in the results dict.
- INTEGRATE-PYTORCH-001 Phase E1.C — E1.C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_e_backend_design.md (tests: pytest tests/torch/test_backend_selection.py -vv; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv): capture passing logs + brief `phase_e_backend_impl.md` summary under the new report directory.
If Blocked: Document the failure in `phase_e_backend_impl.md` with stack traces and config details, archive pytest output even if red, and flag the blocker in docs/fix_plan.md Attempt history.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:127 — reconstructor contract requires dual-backend parity and mandates `update_legacy_dict` before dispatch.
- plans/active/INTEGRATE-PYTORCH-001/phase_e_backend_design.md — defines acceptance for E1.C1–E1.C4 and lists guardrails for dispatcher/import logic.
- tests/torch/test_backend_selection.py:1 — red tests encode required behavior; turning them green proves backend toggle works.
- ptycho/config/config.py:24 — dataclasses are the canonical configuration surface; backend flag must default to TensorFlow to protect backwards compatibility.
- ptycho/workflows/components.py:700 — TensorFlow entry still enforces CONFIG-001; PyTorch path must mirror this ordering.
How-To Map:
- Update dataclasses and config bridge first; rerun `python -m compileall ptycho/config/config.py ptycho_torch/config_bridge.py` to catch syntax errors before tests.
- Dispatcher implementation: wrap backend import inside a helper in the Ptychodus reconstructor (`PtychoPINNReconstructorLibrary`) so CLI/UI code paths stay untouched; raise `RuntimeError` when PyTorch modules cannot be imported.
- Testing: run `pytest tests/torch/test_backend_selection.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T181500Z/pytest_backend_selection.log` then `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_persists_models -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T181500Z/pytest_workflows_backend.log`.
- Summarize code changes, backend logging behavior, and any remaining skips/XFAILs in `phase_e_backend_impl.md` within the same directory.
Pitfalls To Avoid:
- Do not import torch at module scope; keep PyTorch optional with guarded imports.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` — they are protected assets.
- Keep default backend as TensorFlow; changing defaults breaks existing workflows.
- Ensure `update_legacy_dict` still runs before dispatch for both backends.
- Avoid adding tests under unittest harness; stay with pytest style already used.
- Capture artifacts under the specified timestamped directory; no loose logs at repo root.
- Update docs/fix_plan.md Attempt history after the loop; do not skip ledger work.
- Remove `xfail` markers in backend selection tests only after the implementation is ready.
- Maintain device/dtype neutrality; no hard-coded CUDA device selection.
- Do not swallow ImportError silently—surfacing actionable error messages is required.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_e_backend_design.md
- specs/ptychodus_api_spec.md:133
- tests/torch/test_backend_selection.py:59
- ptycho/config/config.py:24
- ptycho_torch/config_bridge.py:1
- phase_d4_alignment.md @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T111700Z/
Next Up: Phase E2.A fixture alignment with TEST-PYTORCH-001 once backend selector lands.
