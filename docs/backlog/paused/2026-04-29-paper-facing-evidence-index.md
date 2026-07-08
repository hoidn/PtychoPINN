---
priority: 50
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    root = Path("/home/ollie/Documents/neurips")
    print(f"paper evidence root target: {root}")
    PY
prerequisites:
  - 2026-04-29-paper-evidence-package-audit
related_roadmap_phases:
  - phase-5-paper-facing-evidence-bundle
signals_for_selection:
  - This is intentionally paused until the roadmap reaches Phase 5 because it creates the paper-facing evidence map outside the PtychoPINN repo.
---

# Backlog Item: Build Paper-Facing Evidence Index

## Objective

- Create `/home/ollie/Documents/neurips/index.md` and the paper-facing evidence
  checklist after the repo-local CDI/CNS evidence package is locked.

## Scope

- Create `/home/ollie/Documents/neurips/index.md` as the top-level evidence
  map for the manuscript.
- Link CDI and CNS result tables, figure manifests, source data, run
  provenance, metric contracts, baseline definitions, ablation summaries,
  dataset/split descriptions, and failed or pivoted experiment notes.
- Generate or copy table source artifacts into
  `/home/ollie/Documents/neurips/tables/` only from locked repo-local sources.
- Write `/home/ollie/Documents/neurips/evidence_checklist.md` with the final
  paper-grade versus decision-support status of every result family.

## Notes for Reviewer

- Keep this paused until Phase 5 is active.
- Do not write manuscript prose unless separately requested.
- Do not include stale or untraceable artifact links.
