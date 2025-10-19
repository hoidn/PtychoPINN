Summary: Align Phase D documentation with runtime evidence and fix_plan ledger.
Mode: Docs
Focus: [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T201900Z/phase_d_hardening/{doc_alignment_notes.md,summary.md}

Do Now:
1. TEST-PYTORCH-001 D2.A @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Update `plans/active/TEST-PYTORCH-001/implementation.md` Phase D table (mark D1 `[x]`, document D2 completion references) and capture a short summary in `plans/active/TEST-PYTORCH-001/reports/2025-10-19T201900Z/phase_d_hardening/doc_alignment_notes.md` (tests: none).
2. TEST-PYTORCH-001 D2.B @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Append a new Attempt entry to `docs/fix_plan.md` citing D2 documentation updates and the 2025-10-19T201900Z artifact hub (tests: none).
3. TEST-PYTORCH-001 D2.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Refresh `docs/workflows/pytorch.md` testing section with the pytest selector, 36s±5s baseline, ≤90s CI guardrail, and POLICY-001/FORMAT-001 reminders; record the edits plus any supplemental notes in `plans/active/TEST-PYTORCH-001/reports/2025-10-19T201900Z/phase_d_hardening/summary.md` (tests: none).

If Blocked: If documentation scope expands beyond Phase D2 (e.g., spec conflicts or missing parity evidence), capture the issue in `plans/active/TEST-PYTORCH-001/reports/2025-10-19T201900Z/phase_d_hardening/doc_alignment_notes.md`, update `plans/active/TEST-PYTORCH-001/implementation.md` D2 row to `[P]` with the blocker description, and log the impediment in `docs/fix_plan.md` Attempts before stopping.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Phase D checklist calls for D2 documentation before CI follow-up.
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md — Source for selector command, runtime mean (35.92s), and guardrails.
- plans/active/TEST-PYTORCH-001/implementation.md#L55 — D1 row still `[ ]`; needs to reference D1 artifacts before D3 work commences.
- docs/workflows/pytorch.md#L151 — Testing guidance lacks the modern pytest selector and runtime expectations established in Phase C/D1.
- docs/findings.md#POLICY-001 — Mandatory PyTorch dependency must be cited in updated workflow docs.

How-To Map:
- Create artifact hub: `mkdir -p plans/active/TEST-PYTORCH-001/reports/2025-10-19T201900Z/phase_d_hardening`
- Plan update: edit `plans/active/TEST-PYTORCH-001/implementation.md` D1/D2 rows; cite `runtime_profile.md`, `doc_alignment_notes.md`, and updated workflow section.
- Notes capture: write `doc_alignment_notes.md` summarizing which files changed, key references, and outstanding risks; update `summary.md` in the new hub with exit-criteria confirmation for D2.
- Fix plan: add Attempt #11 (Phase D2 documentation alignment) under `[INTEGRATE-PYTORCH-001-STUBS]`, including artifact paths and highlights of documentation changes.
- Workflow doc: add a new subsection (e.g., "## 11. Regression Test & Runtime Expectations") covering the pytest selector, runtime baselines, CI guardrails, artifact discipline, and POLICY-001/FORMAT-001 reminders; ensure links to `runtime_profile.md` and the new hub.

Pitfalls To Avoid:
- Do not reuse the 2025-10-19T193425Z directory for new notes—create and reference the 2025-10-19T201900Z hub.
- Keep `plans/active/TEST-PYTORCH-001/implementation.md` checklist IDs intact; update only the State and guidance columns.
- When editing `docs/workflows/pytorch.md`, stay within documentation scope—no code changes or new promises about unimplemented features.
- Preserve prior runtime data (35.86s/35.98s/35.92s); cite them accurately instead of rounding aggressively.
- Reference POLICY-001 and FORMAT-001 explicitly; avoid inventing new policy IDs.
- Use Markdown tables/lists consistently; do not introduce HTML snippets.
- Ensure fix_plan Attempt references both the runtime_profile and the new docs edits.
- Avoid deleting historical summaries in the 2025-10-19T193425Z hub; only add new context in the fresh hub.
- Run no tests; this loop is docs-only per Mode.
- Keep artifact filenames lowercase with underscores to match repository conventions.

Pointers:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md
- plans/active/TEST-PYTORCH-001/implementation.md#L55
- docs/workflows/pytorch.md#L151
- docs/fix_plan.md#L118
- docs/findings.md#POLICY-001

Next Up: TEST-PYTORCH-001 D3 CI integration follow-up once documentation is aligned.
