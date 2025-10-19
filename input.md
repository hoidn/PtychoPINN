Summary: Enforce float32 tensors in PyTorch inference so Lightning convolution accepts checkpoint-loaded batches without dtype mismatch
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-STUBS — Phase D1d PyTorch inference dtype mismatch
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchFloat32 -vv; pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/{pytest_dtype_red.log,pytest_dtype_green.log,pytest_integration_dtype_green.log,summary.md,dtype_triage.md,train_debug.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS D1d @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Add pytest coverage asserting `_build_inference_dataloader`/`_reassemble_cdi_image_torch` keep batches float32; run `pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchFloat32::test_batches_remain_float32 -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/pytest_dtype_red.log` (tests: targeted selector)
2. INTEGRATE-PYTORCH-001-STUBS D1d @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Implement dtype enforcement (cast tensors to `torch.float32` before Lightning forward) and rerun `pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchFloat32::test_batches_remain_float32 -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/pytest_dtype_green.log` (tests: targeted selector)
3. INTEGRATE-PYTORCH-001-STUBS D1d @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Re-run `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/pytest_integration_dtype_green.log` to confirm dtype fix holds end-to-end (tests: targeted selector)
4. INTEGRATE-PYTORCH-001-STUBS D1d @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Update `summary.md`, flip D1d row to `[x]`, record docs/fix_plan.md Attempt (paths above), and note any follow-up risks (tests: none)

If Blocked: Capture the failing selector output into the artifact directory (e.g., `tee .../pytest_dtype_blocked.log`), keep D1d `[P]`, and document the blocker in `dtype_triage.md` plus docs/fix_plan.md before pausing.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md: current D1d row gates parity work until float32 enforcement is proven.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/dtype_triage.md: documents failure stack trace and hypotheses; follow it for evidence collection.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T134500Z/phase_d2_completion/pytest_integration_checkpoint_green.log: shows Lightning load succeeding followed by float64 crash.
- specs/data_contracts.md §1: mandates diffraction tensors remain float32; regression test should encode this contract.
- docs/workflows/pytorch.md §§5–7: orchestrator expectations, including deterministic CPU loaders and dtype parity with TensorFlow.

How-To Map:
- Tests: Add `TestReassembleCdiImageTorchFloat32` in `tests/torch/test_workflows_components.py` (pytest style). Build fixtures that reuse Phase C scaffolding (`MockLightningModule` etc.) and assert `X_batch.dtype is torch.float32` while running `_reassemble_cdi_image_torch` via helper. Expect current code to raise `RuntimeError`; structure the test so the failure is a pytest assertion (not `pytest.raises`).
- Instrumentation: Within the test, inspect `dtype` of `infer_loader.dataset.tensors[0]` and capture in assertion messages so log output proves regression.
- Implementation guidance: In `_build_inference_dataloader`, call `infer_X = infer_X.to(torch.float32, copy=False)` before wrapping in `TensorDataset`. Inside `_reassemble_cdi_image_torch`, ensure `X_batch = X_batch.to(torch.float32)` prior to model invocation. Avoid altering coordinates/offset dtypes (they remain float64 per contract).
- Integration: After unit tests pass, re-run the targeted integration selector with `tee` into the artifact path to confirm the dtype fix resolves the original failure.
- Artifact hygiene: Keep all logs under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/`; do not recreate `train_debug.log` at repo root.

Pitfalls To Avoid:
- Do not downcast complex tensors (`Y`, `probe`)—only the diffraction amplitude batches should be float32.
- Avoid introducing torch dependencies at module import time; guard new torch calls within existing try/except blocks.
- Preserve deterministic loader settings (respect `config.sequential_sampling` and seed).
- Ensure pytest tests stay in native pytest style (no `unittest.TestCase`).
- Keep artifact logs small (use `-vv` only on targeted selectors) and delete temporary checkpoints after tests.
- Do not touch TensorFlow baseline modules (`ptycho/model.py`, `ptycho/tf_helper.py`).
- Make sure integration rerun uses the same dataset path defined in plan; no ad-hoc fixtures.
- Capture both red and green logs to show TDD progression.
- Update plan + fix_plan in the same loop; leaving D1d `[ ]` without rationale is non-compliant.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md#L70
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/dtype_triage.md
- specs/data_contracts.md:34
- docs/workflows/pytorch.md:74
- tests/torch/test_workflows_components.py:1076

Next Up: D2 parity summary refresh once dtype enforcement is green.
