---
priority: 36
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-paper-table-figure-bundle/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/preflight_metadata.json"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing WaveBench preflight summary: {missing}")
    print("wavebench paper-bundle inputs present")
    PY
prerequisites:
  - 2026-04-29-wavebench-native-baseline-reproduction
  - 2026-04-29-wavebench-shared-encoder-supervised-benchmark
related_roadmap_phases:
  - candidate-wavebench-inverse-source-extension
signals_for_selection:
  - Select only after WaveBench has been approved as an additional manuscript evidence lane and at least the supervised shared-encoder results exist.
  - Steering on 2026-04-30 moved WaveBench ahead of remaining optional U-NO table-extension work, but this bundle remains gated on evidence-lane approval.
---

# Backlog Item: Assemble WaveBench Paper Table And Figure Bundle

## Objective

- Convert completed WaveBench inverse-source results into paper-ready tables,
  figures, and provenance manifests after WaveBench is approved as an
  additional manuscript evidence lane.

## Scope

- Consume native WaveBench baseline results and shared-encoder supervised
  results.
- Optionally consume hybrid physics-informed rows if they passed the
  forward-model validity gate.
- Emit:
  - numeric metrics table with source-reconstruction metrics
  - physics-consistency table only for valid physics rows
  - fixed-sample reconstruction figure
  - boundary-data residual figure for physics rows
  - machine-readable provenance manifest
  - claim-boundary summary
- Keep WaveBench separate from the CDI and CNS tables unless a manuscript
  revision explicitly chooses a combined evidence layout.

## Notes for Reviewer

- Do not assemble this bundle before a roadmap/evidence-package amendment adds
  WaveBench as an additional lane.
- Do not mix native WaveBench reference rows and shared-encoder rows without
  clear table labels.
- Do not include physics-informed claims unless the forward-model validation
  item passed.
- Do not let WaveBench table work delay required CDI/CNS evidence closure.
