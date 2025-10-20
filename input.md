Summary: Author the ADR-003 governance addendum so Phase E can proceed with clear acceptance criteria.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase E (Governance dossier E.A1)
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T134500Z/phase_e_governance_adr_addendum/{adr_addendum.md,summary.md}

Do Now:
1. ADR-003-BACKEND-API E.A1 (ADR addendum) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md:E.A1 — draft `adr_addendum.md` capturing Phases A–D evidence, open issues, and acceptance rationale; tests: none.

If Blocked: Capture the blocker details in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T134500Z/phase_e_governance_adr_addendum/blocker.md`, keep E.A1 `[P]`, and log the issue in docs/fix_plan before stopping.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/implementation.md: Phase E hinges on governance dossier before any code edits.
- specs/ptychodus_api_spec.md §4.8: Acceptance doc must codify backend routing + execution config commitments.
- docs/workflows/pytorch.md §§11–13: Provide runtime + helper context cited in the addendum.
- docs/findings.md#POLICY-001 & #CONFIG-001: Addendum must reaffirm mandatory torch dependency and params bridge ordering.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/handoff_summary.md: Use Phase D evidence to justify acceptance.

How-To Map:
- Prep: `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T134500Z/phase_e_governance_adr_addendum`.
- Source material: `phase_e_governance/plan.md` (checklist), `phase_d_cli_wrappers_smoke/{smoke_summary.md,handoff_summary.md}`, `factory_design.md`, `override_matrix.md`, `phase_c4_cli_integration/` closings.
- Deliverables: `adr_addendum.md` (context, decision, evidence, outstanding work), `summary.md` (1–2 paragraph synopsis + next steps), both saved under the new artifact hub.
- Update plan: mark E.A1 `[x]` in `phase_e_governance/plan.md` with artifact links once docs are written; leave E.A2+ untouched.
- Ledger: Append docs/fix_plan Attempt entry summarising E.A1 completion after writing docs; note Mode Docs, tests none.

Pitfalls To Avoid:
- Don’t mix new implementation ideas into addendum—focus on documenting decisions/evidence.
- Keep artifact paths under the timestamped directory; no files at repo root.
- Reference authoritative docs (specs, workflow guide) with accurate section cites.
- Maintain factual parity; avoid promising execution knobs until tracked in E.B tasks.
- Do not edit production code or tests in this loop.
- Ensure addendum records outstanding backlog (execution knobs, tests) rather than resolving them.
- Preserve plan table formatting when flipping E.A1 `[x]`.
- Keep summary concise (<=1 page) but link to supporting docs.
- Use ISO timestamps in new artifacts if additional files created inside the hub.
- Commit artifacts and plan/ledger updates together after review.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/handoff_summary.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md
- specs/ptychodus_api_spec.md:312

Next Up: If time remains after E.A1, queue E.A2 (spec redline) for the following loop.
