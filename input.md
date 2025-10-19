Summary: Extend the PyTorch handoff brief with monitoring cadence and escalation triggers for TEST-PYTORCH-001 Phase D3 guidance.
Mode: Docs
Focus: [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2 — Phase E3.D follow-ups
Branch: feature/torchapi
Mapped tests: none — documentation
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T223500Z/phase_e3_docs_handoff/{monitoring_update.md}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS D3.A–D3.B @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Update `handoff_brief.md` (2025-10-19T215800Z folder) with monitoring cadence (per-PR/nightly/weekly) and explicit escalation triggers (runtime >90s, POLICY-001/CONFIG-001 violations), then capture a `monitoring_update.md` summary under 2025-10-19T223500Z/phase_e3_docs_handoff/ (tests: none).
2. INTEGRATE-PYTORCH-001-STUBS D3.A–D3.B @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Mark D3 rows `[x]` with artifact pointers and append docs/fix_plan.md Attempt summarizing monitoring guidance + escalation matrix (tests: none).

If Blocked: Document outstanding questions in `monitoring_update.md`, flag D3 row `[P]`, update docs/fix_plan.md Attempts, and notify supervisor before exiting.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/handoff_brief.md — Base document to extend with monitoring directives.
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md — Source for runtime guardrails and thresholds.
- specs/ptychodus_api_spec.md:224 — Normative backend dispatch guarantees to reiterate when defining escalation triggers.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Checklist authority for D3.A/D3.B tasks.
- docs/findings.md#POLICY-001 — Reinforces PyTorch dependency requirement in escalation criteria.

How-To Map:
- mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T223500Z/phase_e3_docs_handoff
- Update `handoff_brief.md` §2 (selectors) with cadence bullets (per-PR/nightly/weekly) and §3 (artifact expectations) with explicit escalation triggers (runtime budget breaches, checkpoint load failures, POLICY-001 violations).
- Author `monitoring_update.md` summarizing new cadence table, trigger thresholds, notification workflow, and any open questions; link back to updated sections.
- After edits, set D3 rows to `[x]` in `phase_e3_docs_plan.md` with inline reference to `monitoring_update.md`, then append Attempt entry to `docs/fix_plan.md` capturing Mode=Docs, artifacts, and remaining next steps.

Pitfalls To Avoid:
- Do not invent new policy IDs; leverage POLICY-001/CONFIG-001/FORMAT-001 references already established.
- Keep runtime thresholds aligned with runtime_profile.md (≤90s max, 60s warning, 36s±5s baseline, <20s incomplete) and cite source.
- Maintain ASCII formatting and heading style in `handoff_brief.md`; no TODO placeholders.
- Record any unresolved monitoring questions in `monitoring_update.md` instead of leaving implicit.
- Ensure new summary lives under the 2025-10-19T223500Z timestamp; no stray files at repo root.
- Skip test execution; this is a documentation-only loop.
- Note exact selectors and commands rather than paraphrasing (quote them in backticks).
- Update plan and ledger only after documentation edits are complete to keep traceability clean.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/handoff_brief.md:45
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md:12
- specs/ptychodus_api_spec.md:224
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md:94
- docs/fix_plan.md:6

Next Up: Draft closure recommendation for INTEGRATE-PYTORCH-001 once monitoring cadence is accepted.
