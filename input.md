Summary: Capture the minimal fixture runtime outcomes in project docs and close Phase B3 paperwork.
Mode: Docs
Focus: [TEST-PYTORCH-001] Author PyTorch integration workflow regression — Phase B3 fixture wiring
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T224546Z/phase_b_fixture/{workflow_updates.md,summary.md}

Do Now:
1. TEST-PYTORCH-001 B3.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Refresh `docs/workflows/pytorch.md` §11 so it references the minimal fixture (`tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`), 3.82s smoke runtime, 14.53s integration runtime, and updated CI guardrails; document the edits in `plans/active/TEST-PYTORCH-001/reports/2025-10-19T224546Z/phase_b_fixture/workflow_updates.md`; tests: none.
2. TEST-PYTORCH-001 B3.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — After the doc refresh, mark `plans/active/TEST-PYTORCH-001/implementation.md` B3 `[x]`, append a docs/fix_plan Attempt citing the new artifact hub, and capture loop notes in `plans/active/TEST-PYTORCH-001/reports/2025-10-19T224546Z/phase_b_fixture/summary.md`; tests: none.

If Blocked: Capture the conflicting data (old vs new runtimes, dataset references) in `workflow_updates.md`, leave B3.C `[P]`, and log the blocker in docs/fix_plan.md with the supporting evidence path.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T233500Z/phase_b_fixture/summary.md — Documents the 14.53s runtime that now needs to live in the workflow guide.
- docs/workflows/pytorch.md:246 — Section 11 still reports the 35.9s baseline from Phase D1; it must reflect the minimal fixture evidence.
- plans/active/TEST-PYTORCH-001/implementation.md:47 — B3 row is `[P]` until documentation is updated.
- docs/fix_plan.md:187 — Attempts history must record B3.C completion for traceability.

How-To Map:
- `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before quoting commands or runtime budgets.
- Edit `docs/workflows/pytorch.md` Section 11 to swap in the new dataset path, 3.82s smoke runtime, 14.53s integration runtime, and clarify CI thresholds (<45s fixture target, 90s CI cap, 60s warning).
- Run `git diff docs/workflows/pytorch.md` to verify only the intended section changed; capture the narrative of changes in `workflow_updates.md`.
- Update `implementation.md` B3 row and `docs/fix_plan.md` within the same loop so the ledger, plan, and docs stay in sync.

Pitfalls To Avoid:
- Do not regress the documented requirement to call `CUDA_VISIBLE_DEVICES=""` in the regression selector.
- Keep canonical dataset references elsewhere intact; only swap the default regression fixture narrative.
- Avoid introducing new runtime numbers without citing the 2025-10-19T233500Z logs.
- Do not delete or relocate the existing 2025-10-19T233500Z artifact directory.
- Preserve ASCII formatting and existing `<doc-ref>` tags.
- Ensure the new artifact hub `2025-10-19T224546Z` includes both `workflow_updates.md` and `summary.md`.
- Don’t alter `data/memmap/meta.json` further unless documenting the rationale.

Pointers:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md:56
- docs/workflows/pytorch.md:246
- plans/active/TEST-PYTORCH-001/implementation.md:47
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T233500Z/phase_b_fixture/summary.md:5
- docs/fix_plan.md:187

Next Up: 1. TEST-PYTORCH-001 Phase D variance sweeps once documentation close-out is green.
