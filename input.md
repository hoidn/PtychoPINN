Summary: Define CI integration strategy for the PyTorch regression (Phase D3).
Mode: Docs
Focus: [TEST-PYTORCH-001] Author PyTorch integration workflow regression — Phase D3 CI integration
Branch: feature/torchapi
Mapped tests: none — documentation
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T232500Z/phase_d_hardening/{ci_notes.md,summary.md}

Do Now:
1. TEST-PYTORCH-001 D3.A @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Inspect docs/development/TEST_SUITE_INDEX.md plus `.github/workflows/` configs to inventory where the torch integration selector should run; capture notes in ci_notes.md; tests: none.
2. TEST-PYTORCH-001 D3.B @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Decide on execution command/markers (`pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`), skip conditions, and runtime guardrails; document in ci_notes.md with explicit rationale; tests: none.
3. TEST-PYTORCH-001 D3.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Log any follow-up automation tasks in ci_notes.md, update plans/active/TEST-PYTORCH-001/implementation.md D3 row to [x], and append a docs/fix_plan.md Attempt summarizing Phase D3; tests: none.

If Blocked: Record unanswered questions in ci_notes.md, leave D3 rows as [P], note blockers in docs/fix_plan.md Attempt, and alert supervisor via galph_memory.md before closing loop.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/implementation.md:73 — D3 checklist defines CI integration deliverables.
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md:92 — Detailed D3.A–D3.C task guidance to follow precisely.
- docs/fix_plan.md:118 — Ledger entry expects CI strategy evidence before closing initiative.
- docs/TESTING_GUIDE.md:24 — Integration tier policies inform runtime expectations and gating rules.

How-To Map:
- mkdir -p plans/active/TEST-PYTORCH-001/reports/2025-10-19T232500Z/phase_d_hardening
- ls .github/workflows && cat .github/workflows/*.yml | rg "tests/torch" -n to check existing coverage.
- rg "integration workflow" docs/TESTING_GUIDE.md -n to align with integration tier guidance.
- Summarize findings, commands, and recommended CI hook in ci_notes.md; capture decisions + open questions in summary.md.
- After documentation, edit plans/active/TEST-PYTORCH-001/implementation.md and docs/fix_plan.md with artifact links and state updates.

Pitfalls To Avoid:
- Do not modify CI configs or run pytest in this loop (documentation-only).
- Keep all new artifacts under the specified 2025-10-19T232500Z directory.
- Cite authoritative sources (runtime_profile.md, spec §4.8) instead of recollection.
- Document skip conditions explicitly (e.g., requires torch>=2.2, canonical dataset).
- When updating plan/ledger, include precise artifact paths and completion notes.

Pointers:
- plans/active/TEST-PYTORCH-001/implementation.md:63
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md:70
- docs/TESTING_GUIDE.md:20
- docs/fix_plan.md:107
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md:12

Next Up: 1. TEST-PYTORCH-001 Phase D3 implementation of CI automation if decision requires follow-on changes.
