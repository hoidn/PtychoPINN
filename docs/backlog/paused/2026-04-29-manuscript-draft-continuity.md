---
priority: 55
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    draft = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex")
    figure = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_synthetic_line_pattern_amp_phase.png")
    missing = [str(p) for p in (draft, figure) if not p.exists()]
    if missing:
        raise SystemExit(f"missing manuscript continuity inputs: {missing}")
    print("manuscript continuity inputs present")
    PY
prerequisites:
  - 2026-04-29-paper-facing-evidence-index
related_roadmap_phases:
  - phase-5-paper-facing-evidence-bundle
signals_for_selection:
  - This is intentionally paused until manuscript drafting is explicitly authorized after the Phase 5 evidence index exists.
  - The current TeX draft is reusable manuscript material, not just scratch context.
---

# Backlog Item: Preserve Manuscript Draft Continuity

## Objective

- Use `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  as the starting point for manuscript drafting so future paper-writing tasks
  reuse the existing introduction, context, related-work framing, mathematical
  methods description, benchmark introductions, result framing, and figure/table
  structure.

## Scope

- Read the existing TeX draft before creating or rewriting any manuscript
  section.
- Reuse or revise the draft's paper-facing prose where it remains consistent
  with the locked evidence package.
- Preserve the descriptive benchmark language already introduced in the draft,
  including "synthetic line-pattern CDI benchmark" instead of internal run
  nicknames.
- Reuse the associated CDI ground-truth/reconstruction figure at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_synthetic_line_pattern_amp_phase.png`
  unless a newer locked figure bundle supersedes it.
- Update the draft only to reflect later locked CDI/CNS evidence, reviewer
  feedback, venue formatting, or stronger paper organization.

## Out of Scope

- Do not rerun experiments or change evidence status in this item.
- Do not invent new result claims beyond the locked evidence package.
- Do not replace the draft with a fresh outline that ignores the existing
  introduction, methods, benchmark framing, and available result tables.

## Notes for Reviewer

- This item is a continuity guard for manuscript agents, not an evidence
  generation task.
- Keep it paused while `docs/backlog/roadmap_gate.json` disallows `phase-5-*`.
- When activated, the implementation should first diff the current manuscript
  target against the TeX draft and explicitly state which draft material was
  reused, revised, or superseded by newer evidence.
