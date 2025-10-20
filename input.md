Summary: Update PyTorch CLI documentation (Phase C4.E) so the new execution-config flags and validation evidence are captured across spec, workflow guide, and plan.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4.E (documentation updates)
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120500Z/phase_c4_docs_update/{summary.md,docs_diff.txt}

Do Now:
1. ADR-003-BACKEND-API C4.E1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — Revise docs/workflows/pytorch.md §12–§13 to document the new CLI execution config flags (accelerator, deterministic, num-workers, learning-rate, inference-batch-size) with UPDATED examples reflecting the gridsize=2 smoke; include link to phase_c4d_at_parallel/summary.md; tests: none
2. ADR-003-BACKEND-API C4.E2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — Extend specs/ptychodus_api_spec.md (CLI tables) to add training/inference execution-config flag mappings, citing PyTorchExecutionConfig fields and CONFIG-001 ordering; tests: none
3. ADR-003-BACKEND-API C4.E3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — Refresh CLAUDE.md “Key Commands”/PyTorch sections with a concise CLI example that shows deterministic + accelerator usage and points to the new docs; tests: none
4. ADR-003-BACKEND-API C4.E4 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — Update implementation.md (Phase C4 rows) to reference the new documentation artifacts and log a short summary in plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120500Z/phase_c4_docs_update/summary.md; tests: none

If Blocked: Capture the blocker in plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120500Z/phase_c4_docs_update/blocker.md, note which document could not be updated, revert plan rows C4.E* to [P], and record the reason in docs/fix_plan.md before stopping.

Priorities & Rationale:
- specs/ptychodus_api_spec.md §4.8 and plan C4.E mandate documenting the CLI knobs for governance.
- docs/workflows/pytorch.md needs parity evidence so users can run gridsize=2 confidently.
- CLAUDE.md quick commands are authoritative for agents; they must reflect the new flag vocabulary.
- implementation.md Phase C4 checklist must stay authoritative for downstream planning.

How-To Map:
- Edit docs/workflows/pytorch.md to add a subsection summarizing the five execution-config flags, include the exact CLI command from manual_cli_smoke_gs2.log, and reference summary.md in the new artifact directory.
- Update specs/ptychodus_api_spec.md by inserting a table under §4 or new §7 mapping each flag → dataclass field → factory override precedence.
- In CLAUDE.md, refresh the PyTorch command snippet (Section 5) to show `--accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-3 --gridsize 2` and mention CONFIG-001 bridge.
- After documentation edits, append a new paragraph in plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120500Z/phase_c4_docs_update/summary.md capturing what changed, with bullet links to each file and spec references.

Pitfalls To Avoid:
- Do not modify production code or tests; this loop is documentation-only.
- Keep all new artifacts inside the timestamped reports/2025-10-20T120500Z directory.
- Maintain CONFIG-001 language consistency (bridge before data/model construction).
- Preserve existing Markdown cross-reference tags (<doc-ref>, etc.).
- Avoid removing historical context from docs; append or clearly replace outdated snippets.
- Double-check spelling of flags (`--num-workers`, not `--num_workers`).
- Reference artifact paths relative to repo root (no absolute paths).
- Do not delete prior logs or summaries when updating plan/summary files—append updates.
- Ensure summary.md lists pending Phase C4.F tasks to avoid premature closure.
- Run spellcheck or read-through manually—no automated tooling required.

Pointers:
- docs/workflows/pytorch.md
- specs/ptychodus_api_spec.md
- CLAUDE.md
- plans/active/ADR-003-BACKEND-API/implementation.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/summary.md

Next Up:
- Phase C4.F close-out (comprehensive summary + fix_plan handoff) once documentation is refreshed.
