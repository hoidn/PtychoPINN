Summary: Implement torch-optional `run_cdi_example_torch` inference/stitching path with red→green pytest parity.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 — Prepare for PyTorch Backend Integration with Ptychodus
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_invokes_training -vv ; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T101500Z/{phase_d2c_red.md,phase_d2c_green.md,pytest_red.log,pytest_green.log}
Do Now:
- INTEGRATE-PYTORCH-001 D2.C @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — author red-phase parity test covering `run_cdi_example_torch` + optional stitching; document design + params in phase_d2c_red.md; tests: none.
- INTEGRATE-PYTORCH-001 D2.C @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — implement torch-optional inference/stitching helpers to satisfy new test; capture notes in phase_d2c_green.md; run `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_invokes_training -vv`.
- INTEGRATE-PYTORCH-001 D2.C @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — rerun training parity selector `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning -vv`; update docs/fix_plan Attempts with artifacts + pytest logs (tests: executed).
If Blocked: If torch runtime or Lightning deps missing, capture the import/ModuleNotFoundError output in pytest_red.log, note blockers in phase_d2c_red.md, and stop after updating docs/fix_plan.md with failure context.
Priorities & Rationale:
- Align with reconstructor contract (`specs/ptychodus_api_spec.md:180`) so PyTorch backend mirrors TF `run_cdi_example` lifecycle.
- Follows D2.C guidance in `plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md` (table row now awaiting implementation).
- Builds on existing training stub tests in `tests/torch/test_workflows_components.py:170` to preserve parity.
- Uses TF baseline `ptycho/workflows/components.py:535` as authoritative behaviour for inference + stitching.
- Supports downstream persistence plan (Phase D3) by returning results dict with reconstructed outputs.
How-To Map:
- Author pytest in `tests/torch/test_workflows_components.py` (new class `TestWorkflowsComponentsRun`) that monkeypatches `train_cdi_model_torch`, `_ensure_container`, and a new `_reassemble_with_tensorflow`/`reassemble_cdi_image_torch` helper; include `pytest.importorskip("torch", reason="torch backend required")` guarded by TORCH_AVAILABLE to keep torch-optional behaviour.
- Reference spec expectations for returned tuple `(recon_amp, recon_phase, results)`; ensure test asserts placeholders when stitching disabled/enabled and that `train_cdi_model_torch` invoked once.
- Implement `run_cdi_example_torch` by delegating to existing training stub, wiring optional stitching via placeholder helper returning `None` outputs until Phase D2.C fleshed out; keep import guards identical to training path and respect CONFIG-001 by leaving `update_legacy_dict` call untouched.
- Add helper (e.g., `_reassemble_with_torch`) that currently raises NotImplementedError when stitching requested; update test to xfail or expect NotImplementedError until implemented, but ensure behaviour documented in phase_d2c_green.md.
- After implementation, run targeted pytest selector then rerun training selector to confirm no regression; store logs under specified artifact directory.
Pitfalls To Avoid:
- Do not require actual PyTorch tensors or Lightning runs; keep helpers torch-optional and use monkeypatchable stubs.
- Preserve existing `update_legacy_dict` call order; do not duplicate calls inside helpers.
- Do not import torch-lightning at module scope; wrap inside TORCH_AVAILABLE check.
- Ensure new pytest uses native pytest style (no unittest.TestCase) and respects existing skip logic in `tests/conftest.py`.
- Avoid mutating global `params.cfg` in tests without restoring via fixture (reuse existing pattern).
- Keep placeholder returns consistent with spec signature (tuple of amplitude, phase, results).
- Document any TODOs in artifacts instead of leaving ambiguous comments in code.
Pointers:
- specs/ptychodus_api_spec.md:180 — inference lifecycle contract.
- ptycho/workflows/components.py:535 — TensorFlow run/stitch baseline.
- ptycho_torch/workflows/components.py:1 — current PyTorch scaffold needing D2.C logic.
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — Phase D2 checklist + updated guidance.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T095250Z/phase_d2b_review.md — hand-off notes for D2.C.
Next Up: Consider replacing `_train_with_lightning` stub with real Lightning orchestration once torch runtime available.
