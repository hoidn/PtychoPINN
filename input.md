Summary: Apply backend selection spec updates and capture supporting docs per Phase E3 plan.
Mode: Docs
Focus: [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2 — Phase E3.C spec sync
Branch: feature/torchapi
Mapped tests: none — documentation
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/{phase_e3_spec_patch.md,summary_phase_e3_spec_planning.md}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS C1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Update specs/ptychodus_api_spec.md with §4.8 backend selection text based on phase_e3_spec_patch.md (tests: none).
2. INTEGRATE-PYTORCH-001-STUBS C2 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Review docs/findings.md for backend selection policy need; add POLICY-002 entry if change is normative (tests: none).

If Blocked: Document blocker in phase_e3_spec_patch.md, set C1 or C2 to [P] with note, append summary_phase_e3_spec_planning.md, and record the issue in docs/fix_plan.md before exiting.

Priorities & Rationale:
- specs/ptychodus_api_spec.md:143 — Spec currently lacks backend dispatch contract; needs update for PyTorch parity.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_spec_patch.md — Draft language approved for implementation this loop.
- docs/findings.md:8 — POLICY-001 enforces torch requirement; new backend policy must align if introduced.
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md:68 — Phase E3 dependencies call for spec + knowledge base sync before handoff.
- tests/torch/test_backend_selection.py:59 — Provides acceptance criteria to mirror in spec update.

How-To Map:
- Copy §4.8 text from phase_e3_spec_patch.md into specs/ptychodus_api_spec.md after §4.7, preserving markdown heading levels.
- Verify references to code paths (`ptycho/workflows/backend_selector.py`) use inline code formatting and update table of contents if present.
- If the spec change introduces a new policy, append a row to docs/findings.md with ID POLICY-002, description, evidence link (phase_e3_spec_patch.md), and status Active; otherwise record “no new policy required” note in phase_e3_spec_patch.md.
- Record change summary in a new note (e.g., append to phase_e3_spec_patch.md or create phase_e3_spec_update.md) citing line anchors for modified files.
- All artifacts/logs stay under plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/; update summary_phase_e3_spec_planning.md with outcomes.

Pitfalls To Avoid:
- Do not edit backend selector code; this loop is documentation-only.
- Keep markdown anchors and numbering stable when inserting §4.8.
- Ensure CONFIG-001 language matches existing spec terminology.
- Avoid duplicating POLICY-001 details; reference instead.
- Maintain ASCII characters and existing formatting conventions.
- Do not run tests or touch generated artifacts.
- Preserve prior summary.md content; append new sections rather than overwriting history.
- Reference files with accurate line anchors when updating plan or findings.

Pointers:
- specs/ptychodus_api_spec.md:130
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_spec_patch.md:1
- docs/findings.md:8
- tests/torch/test_backend_selection.py:59
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md:68

Next Up: Phase E3.B documentation updates (pytorch workflow guide, architecture diagram, CLAUDE.md) once spec is synchronized.
