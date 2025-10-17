# PtychoPINN Fix Plan Ledger

**Last Updated:** 2025-10-17
**Active Focus:** Stand up PyTorch backend parity (integration + minimal test harness).


---

## [INTEGRATE-PYTORCH-000] Pre-refresh Planning for PyTorch Backend Integration
- Spec/AT: `plans/ptychodus_pytorch_integration_plan.md`, commit bfc22e7 (PyTorch tree rebase), and `specs/ptychodus_api_spec.md` for contractual alignment.
- Priority: High
- Status: in_progress
- Owner/Date: Codex Agent/2025-10-17
- Reproduction: N/A (planning + documentation refresh)
- Working Plan: plans/active/INTEGRATE-PYTORCH-000/implementation.md
- Attempts History:
  * [2025-10-17] Attempt #0 — Planning: Authored phased rebaseline plan (`plans/active/INTEGRATE-PYTORCH-000/implementation.md`).
  * [2025-10-17] Attempt #1 — Phase A.A1 Evidence Capture: Generated module inventory and delta analysis. Artifacts: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/{module_inventory.md,delta_log.md}`. Key findings: 5 critical deltas identified — (1) Config schema mismatch between PyTorch and TensorFlow spec (blocker); (2) New `api/` layer not in legacy plan; (3) `datagen/` package addition; (4) Reassembly module suite divergence; (5) Lightning+MLflow orchestration vs TensorFlow workflows. No code changes; evidence-only loop. Next: Phase B.B1 architectural decisions + plan redline drafting.
- Exit Criteria:
  - Phase A reports capture current `ptycho_torch/` module inventory and delta analysis (see plan for artifact paths). ✅
  - `plans/ptychodus_pytorch_integration_plan.md` updated to reflect rebased PyTorch stack. [Pending Phase B]
  - docs/fix_plan.md and downstream initiatives link to refreshed plan with current action state noted. [In progress]


## [TEST-PYTORCH-001] Build Minimal Test Suite for PyTorch Backend
- Spec/AT: Corresponds to existing TensorFlow integration test `tests/test_integration_workflow.py` and guidance in `plans/pytorch_integration_test_plan.md`.
- Priority: Critical
- Status: pending
- Owner/Date: Codex Agent/2025-10-16
- Reproduction: N/A (new feature)
- Linked Plan: plans/pytorch_integration_test_plan.md (needs activation under `plans/active/TEST-PYTORCH-001`).
- Attempts History:
  * [2025-10-16] Attempt #0 — Planning: Initial task creation.
- Exit Criteria:
  - A new test file `tests/torch/test_integration_workflow.py` exists.
  - The test successfully runs a minimal train -> save -> load -> infer cycle using the PyTorch backend.
  - The test passes, confirming the basic viability of the PyTorch persistence layer.

## [INTEGRATE-PYTORCH-001] Prepare for PyTorch Backend Integration with Ptychodus
- Spec/AT: `specs/ptychodus_api_spec.md` and `plans/ptychodus_pytorch_integration_plan.md`.
- Priority: High
- Status: in_progress
- Owner/Date: Codex Agent/2025-10-16
- Reproduction: N/A (new feature)
- Working Plan: plans/active/INTEGRATE-PYTORCH-001/implementation.md
- Attempts History:
  * [2025-10-16] Attempt #0 — Planning: Initial task creation.
  * [2025-10-17] Attempt #1 — Planning: Authored phased implementation plan (`plans/active/INTEGRATE-PYTORCH-001/implementation.md`).
  * [2025-10-17] Attempt #2 — Evidence (Phase A): Completed parity baseline inventory. Artifacts: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/{parity_map.md,summary.md}` and `plans/active/INTEGRATE-PYTORCH-001/glossary_and_ownership.md`. Key findings: 7 critical gaps identified across configuration, data pipeline, workflows, and persistence. Configuration dataclass bridge is the #1 blocker (Phase B). All Phase A tasks (A1-A3) completed. Evidence-only loop; no code changes. Next: Phase B — Configuration & Legacy Bridge Alignment.
  * [2025-10-17] Attempt #3 — Review: Marked Phase A checklist complete in `plans/active/INTEGRATE-PYTORCH-001/implementation.md`, updated B1 guidance with config bridge pointers, and prepared next-loop directive (no new artifacts).
- Exit Criteria:
  - All gaps identified in the "TensorFlow ↔ PyTorch Parity Map" within `plans/ptychodus_pytorch_integration_plan.md` are addressed with a concrete implementation plan.
  - A `RawDataTorch` shim and `PtychoDataContainerTorch` class are implemented.
  - Configuration parity (Phase 1 of the integration plan) is complete and tested.
