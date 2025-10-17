Summary: Stage PyTorch training orchestration (D2.B) via TDD before implementation.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase D2.B – Implement training path
Branch: feature/torchapi
Mapped tests: none — author targeted pytest first
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T094500Z/{phase_d2_training.md,pytest_red.log,pytest_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001 D2.B @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — Author red-phase pytest (`TestWorkflowsComponentsTraining`) covering `_ensure_container` + Lightning delegation skeleton; tests: `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning -vv`.
2. INTEGRATE-PYTORCH-001 D2.B @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T093500Z/phase_d2_training_analysis.md — Implement `_ensure_container` helper + `train_cdi_model_torch` Lightning path to turn test green; tests: `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning -vv`.
3. INTEGRATE-PYTORCH-001 D2.B @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — Capture training summary + pytest logs in `phase_d2_training.md`; tests: none.

If Blocked: Document the obstacle and captured pytest output in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T094500Z/blocked.md`, then update docs/fix_plan.md Attempts with failure details before pausing.

Priorities & Rationale:
- `specs/ptychodus_api_spec.md:186-189` mandates backend parity for train/infer orchestration; D2.B delivers the training half.
- `plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md` lists D2.B as next dependency before persistence (D3).
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T093500Z/phase_d2_training_analysis.md` catalogs TF baseline + PyTorch gaps; following it keeps CONFIG-001/ DATA-001 protections intact.

How-To Map:
- Red phase: `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning -vv > plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T094500Z/pytest_red.log 2>&1`
- Green phase: rerun the same selector, save to `pytest_green.log`; append summary + decision notes to `phase_d2_training.md`.
- Keep artifacts torch-optional: stub Lightning trainer/model inside the test using monkeypatch, per analysis doc §TDD.

Pitfalls To Avoid:
- Do not import real Lightning components in tests without guarding for `TORCH_AVAILABLE`; use monkeypatch stubs.
- Keep `_ensure_container` torch-optional—rely on Phase C adapters, no direct torch.tensor creation.
- No full training runs or subprocess calls; D2.B unit test should remain fast and CPU-only.
- Do not bypass `update_legacy_dict`; ensure test keeps existing parity guard intact.
- Avoid writing artifacts outside the timestamped reports directory.
- Preserve existing exports in `ptycho_torch/workflows/__init__.py`; update cautiously.
- Keep MLflow disabled in tests to prevent network side effects.
- Leave `NotImplementedError` in inference path (D2.C) untouched this loop.

Pointers:
- docs/fix_plan.md (INTEGRATE-PYTORCH-001 attempts #43-44)
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md (D2 checklist)
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T093500Z/phase_d2_training_analysis.md
- ptycho/workflows/components.py:535-666 (TensorFlow reference)
- ptycho_torch/workflows/components.py:1-213 (Scaffold to extend)

Next Up: D2.C inference/stitching path once training orchestration is green.
