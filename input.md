Summary: Draft TEST-PYTORCH-001 handoff brief capturing backend contract, selectors, and artifact expectations (Phase E3.D1).
Mode: Docs
Focus: [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2 — Phase E3.D handoff
Branch: feature/torchapi
Mapped tests: none — documentation
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/{plan.md,summary.md,handoff_brief.md}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS D1.A–D1.C @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/plan.md — Author `handoff_brief.md` covering backend literals/CONFIG-001, regression selectors + runtime guardrails, artifact expectations & ownership matrix (tests: none).

If Blocked: Capture findings in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/summary.md`, flag affected checklist row(s) to `[P]`, update docs/fix_plan.md Attempts history, and alert supervisor before exiting.

Priorities & Rationale:
- specs/ptychodus_api_spec.md:224-235 — Backend dispatch guarantees must appear in the handoff.
- docs/workflows/pytorch.md:297 — Provides runtime baseline (~36s) and regression selector context.
- plans/active/TEST-PYTORCH-001/implementation.md:64 — Phase D runtime hardening needs explicit instructions from the handoff.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/plan.md — Defines D1 subtasks and artifact expectations.
- docs/findings.md#POLICY-001 — Reinforce torch>=2.2 requirement and fail-fast messaging.

How-To Map:
- Create `handoff_brief.md` in the artifact directory with sections for (1) Backend selection contract (include literals, CONFIG-001 reminder, RuntimeError wording), (2) Required pytest selectors with cadence (integration workflow, backend selection suite, parity checks), (3) Artifact/log expectations (checkpoint, recon PNGs, runtime_profile reference), (4) Ownership & escalation matrix.
- Cite runtime guardrails (≤90s CI max, 60s warning, 36s±5s baseline) from `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`.
- Reference doc/spec anchors explicitly (spec §4.8, workflow guide §12) when describing configuration steps.
- Note assumptions about CI environment (CPU default, CUDA optional) and document open questions in the brief if unresolved.

Pitfalls To Avoid:
- Do not invent new policy IDs; rely on POLICY-001/FORMAT-001.
- Keep literals `'tensorflow'`/`'pytorch'` quoted and lowercase.
- Store the brief in the designated artifact directory; no files at repo root.
- Avoid duplicating plan content verbatim; synthesize guidance tailored for TEST-PYTORCH-001 owners.
- Maintain ASCII formatting and existing Markdown style; no TODO placeholders.
- Stay in Docs mode—skip pytest execution.
- Record any open risks or assumptions in the brief so they can be closed in later phases.

Pointers:
- specs/ptychodus_api_spec.md:224
- docs/workflows/pytorch.md:297
- plans/active/TEST-PYTORCH-001/implementation.md:64
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/plan.md
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md

Next Up: D2.A–D2.B plan/ledger updates once the handoff brief is published.
