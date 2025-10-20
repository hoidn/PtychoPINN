# Phase C3 Workflow Integration — Supervisor Planning Notes (2025-10-20T025643Z)

## Objective
Prepare execution config integration through PyTorch workflows so trainer/inference helpers honour Lightning runtime knobs introduced in Phases C1–C2.

## Key Decisions This Loop
- Authored detailed C3 plan with sub-phase checklists covering Trainer wiring, inference integration, deterministic validation, and reporting hygiene.
- Flagged regression: `ptycho/config/config.py` lost the `__all__` export block during Phase C2; schedule restoration under task C3.A1 before mutating workflows.
- Identified hygiene gap: root-level `train_debug.log` resurrected by C2 full-suite run; mandate relocation/removal in C3.D3.

## Next Steps for Engineer
- Follow `phase_c3_workflow_integration/plan.md` tasks (C3.A1 → C3.C3) using TDD. Capture RED/ GREEN pytest logs in this directory.
- Update workflow tests to assert trainer kwargs + deterministic mode, then document evidence here before updating fix ledger.

## Artifact Map
- Plan: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/plan.md`
- RED Log (expected): `.../pytest_workflows_execution_red.log`
- GREEN Log (expected): `.../pytest_workflows_execution_green.log`
- Summary Updates: append implementation notes + outstanding questions upon engineer completion.

## Outstanding Questions
- How should scheduler/accumulation knobs degrade on CPU-only systems? Document decision after tests.
- Do we need additional mocks for Lightning trainer to assert deterministic state without GPU context?
