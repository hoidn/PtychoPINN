Summary: Align developer guidance with the new torch-required policy and capture documentation changes for Phase F4.1.
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 Phase F4 — Documentation/spec sync & handoff
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T203640Z/{doc_updates.md,spec_sync.md,handoff_notes.md}

Do Now:
1. INTEGRATE-PYTORCH-001 — Complete F4.1.A+B+C @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md; migrate all torch-optional language in CLAUDE.md/docs/workflows/pytorch.md/README and log changes to doc_updates.md (tests: none).
2. INTEGRATE-PYTORCH-001 — Update phase_f_torch_mandatory.md F4 row and docs/fix_plan.md history once F4.1 artifacts are saved (tests: none).

If Blocked: Capture findings in doc_updates.md (even if no edits land) and note the blocker in docs/fix_plan.md Attempts History before exiting the loop.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md:1 — new checklist defines required deliverables for Phase F4.
- CLAUDE.md:57 — current directive still mandates torch-optional parity; must be rewritten to reflect governance decision.
- docs/workflows/pytorch.md:19 — prerequisites omit the torch-required policy framing and need to match setup.py dependency list.
- docs/fix_plan.md:1 — ledger must cite the new documentation artifacts for traceability.

How-To Map:
- Run `rg "torch-optional" docs/ README.md` and record matches in plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T203640Z/doc_updates.md before editing.
- Update CLAUDE.md directive (line 57) to state PyTorch is required, referencing Phase F governance + migration evidence.
- Refresh docs/workflows/pytorch.md prerequisites/sections to emphasize mandatory PyTorch install; adjust README language accordingly.
- Append a short changelog in doc_updates.md summarizing each file touched, anchor location, and rationale.
- After saving artifacts, edit phase_f_torch_mandatory.md F4.1 row to `[x]` if complete (or `[P]` if partial) and add Attempt entry to docs/fix_plan.md with artifact path.

Pitfalls To Avoid:
- Do not leave doc_updates.md empty; inventory is required even if edits are pending.
- Avoid introducing new torch-optional phrasing elsewhere in docs.
- Preserve existing directive structure and XML-like tags in CLAUDE.md when rewriting text.
- Keep README edits limited to PyTorch policy updates—no drive-by formatting.
- Do not run tests; this is a documentation-only loop.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md:1
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:58
- CLAUDE.md:57
- docs/workflows/pytorch.md:17
- docs/fix_plan.md:5

Next Up: Phase F4.2 — Update specs/ptychodus_api_spec.md and docs/findings.md per plan once F4.1 artifacts are committed.
