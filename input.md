Summary: Capture Phase D4 alignment narrative and selector map to unlock PyTorch regression tests
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase D4.A planning alignment
Branch: feature/torchapi
Mapped tests: none — planning
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T111700Z/{phase_d4_alignment.md,phase_d4_selector_map.md}
Do Now:
1. INTEGRATE-PYTORCH-001 — D4.A1 alignment narrative @ plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:25 — tests: none — Summarize TEST-PYTORCH-001 activation strategy and ownership in phase_d4_alignment.md (cite fix_plan attempts + stakeholder brief).
2. INTEGRATE-PYTORCH-001 — D4.A3 selector map @ plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:27 — tests: none — Build phase_d4_selector_map.md listing authoritative pytest selectors, env overrides (torch optional, MLflow toggle), artifact storage expectations.
If Blocked: Note blockers in phase_d4_alignment.md, update docs/fix_plan.md Attempts History with the blocker description, and ping supervisor next loop.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:18 — Phase D4 requires alignment artifacts before code changes.
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:50 — Workflow checklist now delegates D4.A work to the new plan.
- specs/ptychodus_api_spec.md:194 — Persistence loader contract informs selector coverage and handoff notes.
- plans/pytorch_integration_test_plan.md:13 — Integration harness expectations drive selector and ownership mapping.
- docs/fix_plan.md:57 — Current focus item mandates Phase D4 progress before TEST-PYTORCH-001 can start.
How-To Map:
- Create artifact directory `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T111700Z` before writing markdown files.
- For D4.A1, pull context from `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/stakeholder_brief.md` and docs/fix_plan attempts; capture ownership table (e.g., who drives TEST-PYTORCH-001) and dependency gating (D4.B readiness).
- For D4.A3, enumerate selectors covering persistence (`pytest tests/torch/test_model_manager.py::TestLoadTorchBundle -vv`), workflows (`pytest tests/torch/test_workflows_components.py -k torch -vv`), and upcoming integration tests; document skip behavior via `tests/conftest.py` torch guards and env requirements like `CUDA_VISIBLE_DEVICES=""` and `MLFLOW_TRACKING_URI=memory`.
- Reference docs/workflows/pytorch.md:1-102 for Lightning/MLflow knobs when describing overrides.
- Keep both documents concise (<2 pages each) and link back to the plan + ledger entries at the top.
Pitfalls To Avoid:
- Do not edit production code or existing tests in this loop.
- Don’t invent new artifact locations; stick to the provided timestamped directory.
- Avoid duplicating specs content; cite relevant files instead.
- Keep selectors torch-optional—note skip markers rather than assuming torch availability.
- Ensure narrative captures dependency on TEST-PYTORCH-001 without promising work outside D4 scope.
- No ad-hoc scripts outside repo tools; use python one-liners if you must inspect data.
- Record any open questions in the alignment doc under a separate heading.
- Maintain ASCII-only content in plan artifacts.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d4_regression.md:1
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:50
- specs/ptychodus_api_spec.md:180
- plans/pytorch_integration_test_plan.md:13
- docs/TESTING_GUIDE.md:45
Next Up: D4.B1 failing persistence regression tests once alignment + selector map are approved.
