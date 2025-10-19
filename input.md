Summary: Update CLAUDE.md and README.md with PyTorch backend selection guidance, then verify docs cross-links (Phase E3.B2/B3).
Mode: Docs
Focus: [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2 — Phase E3.B2 dual-backend messaging
Branch: feature/torchapi
Mapped tests: none — documentation
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T213900Z/phase_e3_docs_b2/{guidance.md,summary.md,rg_notimplemented.log}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS B2 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Apply CLAUDE.md + README.md backend messaging updates per guidance.md (tests: none).
2. INTEGRATE-PYTORCH-001-STUBS B3 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Run backend-doc verification command and capture rg_notimplemented.log (tests: none).

If Blocked: Document the blocker in plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T213900Z/phase_e3_docs_b2/summary.md, set the affected plan rows to [P] with rationale, update docs/fix_plan.md attempts, and notify supervisor before exiting.

Priorities & Rationale:
- CLAUDE.md:53 — Needs explicit PyTorch backend selection + CONFIG-001 reminder (per spec §4.8, docs/workflows/pytorch.md §12).
- README.md:12 — Feature list lacks dual-backend architecture summary; onboarding requires PyTorch parity messaging.
- specs/ptychodus_api_spec.md:224 — Newly normative backend selection requirements must be cited in docs.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T213900Z/phase_e3_docs_b2/guidance.md — Contains detailed steps, command expectations, and artifact paths.
- docs/findings.md#POLICY-001 — Reinforce PyTorch mandatory policy in updated messaging.

How-To Map:
- `CLAUDE.md`: Insert a short paragraph under Section 4 (parameter initialization) noting that PyTorch backend selection uses `TrainingConfig.backend='pytorch'`, requires `update_legacy_dict`, and references docs/workflows/pytorch.md §12 + spec §4.8; mention runtime_profile artifact for parity evidence.
- `README.md`: Add `### Dual-Backend Architecture` after the Features list covering TensorFlow default, PyTorch backend availability, workflow doc pointer, and Phase D runtime evidence link (file path only).
- After edits, update `guidance.md` checklist if anything deviates and append execution notes to `summary.md` (artifact hub above).
- Run `rg "NotImplementedError" docs/workflows/pytorch.md docs/architecture.md | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T213900Z/phase_e3_docs_b2/rg_notimplemented.log`; confirm output empty; cite command + result in summary.md.
- Update `phase_e3_docs_plan.md` B2/B3 rows to `[x]` with brief completion notes and artifact references; append docs/fix_plan.md Attempts entry if new evidence is produced.

Pitfalls To Avoid:
- Preserve directive XML tags and numbering in CLAUDE.md; keep text ASCII.
- Do not alter mermaid diagrams or existing section numbering beyond adding the new README subsection.
- Reference internal docs via relative paths; avoid external URLs besides existing ones.
- Keep PyTorch requirement phrasing aligned with POLICY-001 (torch>=2.2, actionable RuntimeError on missing torch).
- Ensure guidance cites docs/workflows/pytorch.md §12 and specs/ptychodus_api_spec.md §4.8 exactly; double-check line anchors in diff_notes.
- Store command output in the provided artifact directory; do not leave logs at repo root.
- Maintain consistent `'tensorflow'` / `'pytorch'` literals in documentation (single quotes).
- Skip tests per Mode: Docs; do not run pytest.
- Avoid introducing TODO headings; convert gaps to plan rows if new issues emerge.
- Update docs/fix_plan.md only once per loop with consolidated summary.

Pointers:
- CLAUDE.md:53
- README.md:12
- docs/workflows/pytorch.md:297
- specs/ptychodus_api_spec.md:224
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md:34

Next Up: Prepare Phase E3.D handoff brief once B2/B3 documentation tasks close.
