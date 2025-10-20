Summary: Close Phase C4 by shipping the summary, ledger entry, and hygiene checks for documentation work.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4.F (close-out)
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123500Z/phase_c4f_closeout/{summary.md,ledger_update.md}

Do Now:
1. ADR-003-BACKEND-API C4.F1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:126 — Author Phase C4 summary covering doc updates, test evidence, deferred knobs; tests: none
2. ADR-003-BACKEND-API C4.F2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:127 — Append new Attempt entry to docs/fix_plan.md (C4 close-out) with artifact links + exit criteria; tests: none
3. ADR-003-BACKEND-API C4.F3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:128 — Capture Phase D prep notes in the new summary (deferred knobs, owners, prerequisites); tests: none
4. ADR-003-BACKEND-API C4.F4 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:129 — Run hygiene check (git status, stray files) and document outcome in ledger_update.md; tests: none

If Blocked: Log the blocker in plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123500Z/phase_c4f_closeout/blocker.md, revert checklist states in plan.md back to [P], and note the issue in docs/fix_plan.md before stopping.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:120 — C4.F checklist is the remaining gate before Phase D work can begin.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120500Z/phase_c4_docs_update/summary.md:1 — Documentation artifacts must be reflected in the closing summary.
- docs/fix_plan.md:143 — Ledger needs a fresh Attempt entry capturing C4.F deliverables.
- plans/active/ADR-003-BACKEND-API/implementation.md:40 — Implementation plan expects C4 finalized prior to Phase D.

How-To Map:
- Create the artifact directory before writing (`mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123500Z/phase_c4f_closeout`).
- Draft `summary.md` capturing: CLI flag exposure scope, key evidence (tests/logs), documentation deltas (with section numbers), exit-criteria checklist, and explicit list of deferred knobs for Phase D.
- Update `docs/fix_plan.md` Attempts History with a new entry summarizing Phase C4.F (timestamp, tasks, artifact paths, confirmation of hygiene check).
- Record hygiene verification steps (commands run, findings) in `ledger_update.md`; include `git status` output snapshot after cleanup.
- After edits, ensure plan checklist C4.F rows are marked `[x]` where appropriate and reference the new artifacts.

Pitfalls To Avoid:
- Do not modify production Python modules or tests in this loop.
- Keep all new artifacts inside the specified timestamped directory (no root-level logs).
- Reference existing evidence (e.g., phase_c4d_at_parallel logs) rather than re-running tests.
- Preserve numbering/formatting in docs/fix_plan.md (no renumbering of prior attempts).
- Ensure summary cites exact filenames and paths for cross-reference.
- Run hygiene checks after documentation work to capture final repository state.
- Avoid duplicating content already captured in the Phase C4.E summary; link instead.
- Keep narrative concise—focus on evidence, exit criteria, and next steps.
- When quoting plan items, use their IDs exactly (C4.F1–C4.F4).
- Confirm no temporary files remain in `tmp/` or repository root before finishing.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:120
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/summary.md:180
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120500Z/phase_c4_docs_update/summary.md:1
- docs/fix_plan.md:140
- plans/active/ADR-003-BACKEND-API/implementation.md:40

Next Up:
- Phase D1 (PyTorch training CLI refactor) once C4.F artifacts are merged.
