# PtychoPINN Fix Plan Ledger

**Last Updated:** 2025-10-16
**Active Focus:** Restore pytest signal; PyTorch integration follows once green baseline returns.


---

## [TEST-SUITE-TRIAGE] Restore pytest signal and triage failures
- Spec/AT: `docs/TESTING_GUIDE.md`, `docs/debugging/debugging.md`, `specs/data_contracts.md`
- Plan: `plans/active/TEST-SUITE-TRIAGE/plan.md`
- Priority: Critical (Do Now)
- Status: planning
- Owner/Date: Galph Supervisor/2025-10-16
- Reproduction: `pytest tests/ -vv`
- Attempts History:
  * [2025-10-16] Attempt #0 — Planning: Established multi-phase triage plan, awaiting execution.
- Exit Criteria:
  - Pytest sweep captured in timestamped report with failure ledger per Phase A.
  - Each failure classified (bug vs deprecated/flake) with references in plan summary (Phase B).
  - docs/fix_plan.md updated with delegated fix items per failure cluster and input.md prepared for execution (Phase C).


---

## [TEST-PYTORCH-001] Build Minimal Test Suite for PyTorch Backend
- Spec/AT: Corresponds to existing TensorFlow integration test `tests/test_integration_workflow.py` and guidance in `plans/pytorch_integration_test_plan.md`.
- Priority: Critical
- Status: pending
- Owner/Date: Codex Agent/2025-10-16
- Reproduction: N/A (new feature)
- Attempts History:
  * [2025-10-16] Attempt #0 — Planning: Initial task creation.
- Exit Criteria:
  - A new test file `tests/torch/test_integration_workflow.py` exists.
  - The test successfully runs a minimal train -> save -> load -> infer cycle using the PyTorch backend.
  - The test passes, confirming the basic viability of the PyTorch persistence layer.

## [INTEGRATE-PYTORCH-001] Prepare for PyTorch Backend Integration with Ptychodus
- Spec/AT: `specs/ptychodus_api_spec.md` and `plans/ptychodus_pytorch_integration_plan.md`.
- Priority: High
- Status: pending
- Owner/Date: Codex Agent/2025-10-16
- Reproduction: N/A (new feature)
- Attempts History:
  * [2025-10-16] Attempt #0 — Planning: Initial task creation.
- Exit Criteria:
  - All gaps identified in the "TensorFlow ↔ PyTorch Parity Map" within `plans/ptychodus_pytorch_integration_plan.md` are addressed with a concrete implementation plan.
  - A `RawDataTorch` shim and `PtychoDataContainerTorch` class are implemented.
  - Configuration parity (Phase 1 of the integration plan) is complete and tested.
