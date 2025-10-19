Summary: Document backend selection guidance in workflow + architecture docs for Phase E3.B.
Mode: Docs
Focus: [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2 — Phase E3.B1 backend documentation refresh
Branch: feature/torchapi
Mapped tests: none — documentation
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T210000Z/phase_e3_docs_update/{diff_notes.md,summary.md}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS B1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Update docs/workflows/pytorch.md with a backend selection section that references spec §4.8, dispatcher CLI usage, and test selectors (tests: none).
2. INTEGRATE-PYTORCH-001-STUBS B1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Add backend selector notes to docs/architecture.md (diagram caption + text block) ensuring TensorFlow/PyTorch paths are described (tests: none).
3. INTEGRATE-PYTORCH-001-STUBS B1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Capture edits and rationale in plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T210000Z/phase_e3_docs_update/{diff_notes.md,summary.md}; update phase_e3_docs_plan.md B1 row with completion notes (tests: none).

If Blocked: Record blocker details in phase_e3_docs_update.md, set B1 row to [P] with the reason, update summary.md, and add the note to docs/fix_plan.md attempts before exiting.

Priorities & Rationale:
- docs/workflows/pytorch.md:246 — Needs backend selection guidance for Ptychodus users (E3 inventory high gap).
- docs/architecture.md:40 — Component diagram should call out backend selector and PyTorch branch.
- specs/ptychodus_api_spec.md:224 — Newly added §4.8 provides authoritative requirements to cite.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — B1 checklist defines deliverables and artifact expectations.

How-To Map:
- Add a dedicated subsection after §11 in docs/workflows/pytorch.md summarizing backend selection steps; cite spec §4.8 and list the pytest selectors (`tests/torch/test_backend_selection.py`) for verification.
- Update docs/architecture.md component diagram caption or accompanying text to note backend selector routing to TensorFlow vs PyTorch stacks; include CONFIG-001 reminder.
- Record concrete edits (file + line anchors) in diff_notes.md and summarize key messaging changes + open questions in summary.md.
- Ensure artifact files live under plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T210000Z/phase_e3_docs_update/ and reference them from phase_e3_docs_plan.md B1 row.

Pitfalls To Avoid:
- Do not modify backend_selector code—documentation only.
- Keep mermaid diagram syntax valid when editing architecture.md.
- Preserve ASCII formatting; avoid smart quotes beyond existing usage.
- Reference spec/tests with accurate line numbers.
- Maintain consistent numbering in docs/workflows/pytorch.md sections.
- Retain existing Phase D2 regression guidance; append rather than overwrite.
- Note CONFIG-001 requirement explicitly—don’t assume implicit knowledge.
- Keep artifact filenames as instructed to avoid orphaned reports.
- Do not add new TODO headings; convert gaps into plan entries instead.

Pointers:
- docs/workflows/pytorch.md:246
- docs/architecture.md:40
- specs/ptychodus_api_spec.md:224
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md#L32
- tests/torch/test_backend_selection.py:59

Next Up: Phase E3.B2 (CLAUDE.md + README dual-backend messaging) once workflow/architecture docs are updated.
