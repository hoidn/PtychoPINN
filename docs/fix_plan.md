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
  * [2025-10-17] Attempt #2 — Phase B.B1 Planning: Drafted redline outline and summary to guide canonical plan edits. Artifacts: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025633Z/{plan_redline.md,summary.md}`. Captured decision inventory (API surface, config schema harmonization, persistence format, Lightning dependency policy) and sequenced editing order for `plans/ptychodus_pytorch_integration_plan.md`. Next: Phase B.B2 apply redline updates, then B.B3 brief stakeholders.
  * [2025-10-17] Attempt #3 — Phase B.B2 Canonical Plan Update: Applied redline edits to `plans/ptychodus_pytorch_integration_plan.md`. Implemented all 5 revision items: (1) Updated Section 1 scope with dual backend surface and config bridge priority; (2) Overhauled Phase 0-5 sections with API layer decision gate, config schema harmonization details, datagen package notes, barycentric reassembly documentation, and Lightning/MLflow persistence strategy; (3) Added Phase 8 "Spec & Ledger Synchronization" subsection; (4) Refreshed deliverables list with PyTorch-specific items (memory-mapped shim, persistence adapter, parity tests); (5) Appended 4 new risk entries (API layer drift, config schema divergence, Lightning/MLflow dependencies, reassembly parity complexity). Artifacts referenced: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025633Z/{plan_redline.md,delta_log.md}`. No code changes (docs-only loop). Next: Phase B.B3 stakeholder brief; Phase C governance sync.
  * [2025-10-17] Attempt #4 — Phase B.B3 Prep & Governance Kickoff: Marked Phase B.B2 complete in `plans/active/INTEGRATE-PYTORCH-000/implementation.md`, created stakeholder brief outline at `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/brief_outline.md`, and rewrote `input.md` to direct B.B3 execution plus Phase C sync. Next: Draft the full stakeholder brief and log cross-initiative follow-ups (Phase C tasks).
  * [2025-10-17] Attempt #5 — Phase C.C2 Stakeholder Brief: Authored comprehensive stakeholder brief at `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/stakeholder_brief.md` summarizing 5 critical deltas from canonical plan updates. Key deliverables: (1) Delta-by-delta breakdown with execution roadmaps mapped to INTEGRATE-PYTORCH-001 phases and TEST-PYTORCH-001 fixtures; (2) 8 open governance questions logged in `open_questions.md` with decision forums and blocking relationships; (3) Configuration bridge (Delta 1) confirmed as #1 blocker per CONFIG-001 finding; (4) Immediate next steps defined for INTEGRATE-PYTORCH-001 Phase B (config schema audit, failing test, harmonization implementation). Artifacts: `{stakeholder_brief.md,open_questions.md}`. No code changes (docs-only loop). Next: Phase C.C3 — update downstream plans (`plans/active/INTEGRATE-PYTORCH-001/implementation.md`) with brief references and coordinate test fixture requirements with TEST-PYTORCH-001.
- Exit Criteria:
  - Phase A reports capture current `ptycho_torch/` module inventory and delta analysis (see plan for artifact paths). ✅
  - `plans/ptychodus_pytorch_integration_plan.md` updated to reflect rebased PyTorch stack. ✅ [Phase B.B2 complete — all redline items applied]
  - docs/fix_plan.md and downstream initiatives link to refreshed plan with current action state noted. ✅ [Phase C.C1 complete — attempt #5 logged; C.C2 complete — stakeholder brief authored; C.C3 pending — downstream plan updates]


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
