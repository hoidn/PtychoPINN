---
priority: 40
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing evidence package inputs: {missing}")
    print("evidence package inputs present")
    PY
prerequisites:
  - 2026-04-29-cdi-lines128-minimum-paper-table
  - 2026-04-29-cns-paper-table-figure-bundle
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - After CDI and CNS bundles exist, the project needs a repo-local manifest and drafting gate before paper-facing artifact assembly.
  - This item verifies claim boundaries without creating the Phase 5 `/home/ollie/Documents/neurips/` evidence map.
---

# Backlog Item: Audit Paper Evidence Package

## Objective

- Create a repo-local paper evidence manifest and completeness audit that tells
  the manuscript what can be claimed now, what is decision-support only, and
  what remains blocked.

## Scope

- Consume the locked CDI and CNS table/figure bundles and their provenance
  manifests.
- For the CNS pillar, consume the selected bounded capped `history_len=2`
  contract from
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
  and preserve its explicit claim boundary instead of rewriting CNS as a
  full-training benchmark.
- Emit `paper_evidence_manifest.json` or an equivalent structured manifest
  under the NeurIPS project plan tree.
- Record each table row, figure bundle, source-array path, metric schema,
  provenance root, row status, and claim boundary.
- Write a durable audit summary under
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/` with explicit labels for
  `paper_grade`, `full_training`, `capped_decision_support`,
  `decision_support`, `blocked`, and `not_protocol_compatible`.
- Identify which manuscript sections are draftable and which result claims must
  remain placeholders.

## Notes for Reviewer

- Do not create or populate `/home/ollie/Documents/neurips/` in this item; that
  belongs to the later paper-facing evidence-bundle phase.
- Do not mark missing provenance as acceptable because metrics exist.
- Do not collapse CDI and CNS claim boundaries into one generic success label.
