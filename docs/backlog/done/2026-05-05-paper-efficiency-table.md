---
priority: 34
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-paper-efficiency-table/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing paper efficiency table inputs: {missing}")
    print("paper efficiency table inputs present")
    PY
  - python -m compileall -q scripts/studies ptycho_torch
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
  - 2026-04-30-cdi-lines128-uno-table-extension
  - 2026-05-04-cns-matched-condition-table-refresh
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - Parameter counts already exist across the CDI and CNS source artifacts, but there is no single paper-facing efficiency table that compiles and labels them consistently.
  - Training runtime exists in heterogeneous fields across source manifests and table JSON; it needs extraction, normalization, and caveat labeling before manuscript use.
  - True inference throughput may be absent for some rows; this item should run only lightweight missing probes, not rerun training.
---

# Backlog Item: Compile Paper Efficiency Table

## Objective

- Build a repo-local efficiency table for the NeurIPS evidence package by
  compiling parameter counts, training runtime, hardware metadata, and any
  available inference-throughput fields from existing CDI and CNS artifacts.

## Scope

- Consume existing source artifacts for the current paper authorities:
  - CDI `lines128` complete table plus U-NO extension;
  - CNS matched-condition table;
  - any candidate-lane rows that are already explicitly approved for additive
    paper-evidence use by their own evidence gate.
- Extract and normalize, per row where available:
  - model label and row id;
  - dataset / benchmark lane;
  - parameter count;
  - training runtime field and source field name
    (`runtime_sec`, `command_wall_time_sec`, `train_wall_time_sec`, etc.);
  - hardware/device string and CUDA/PyTorch provenance when present;
  - inference latency or throughput if already recorded;
  - source artifact path and claim boundary.
- Emit machine-readable and manuscript-ready outputs under
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/`, including JSON, CSV, and
  TeX payloads plus a short durable summary.
- If inference throughput is missing for a row that is otherwise part of the
  headline table, run only a lightweight deterministic inference probe under the
  existing dataset/row contract. Do not train or tune any model.
- Label heterogeneous runtime evidence honestly. Training runtime from old
  launches is acceptable as provenance-backed context, but must not be presented
  as a strictly normalized speed benchmark unless the same measurement protocol
  was used for all compared rows.

## Required Interpretation

- Parameter counts can be compiled from existing evidence; no new model training
  is authorized for parameter counting.
- Training runtime is a provenance field unless the item proves a common
  measurement protocol.
- Inference throughput should be marked `measured`, `missing`, or `not
  comparable` per row.
- Do not combine CDI, CNS, BRDT, or WaveBench rows into a single model-ranking
  table. The efficiency table may be cross-lane, but it must group rows by
  benchmark and preserve each benchmark's claim boundary.

## Outputs

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`

## Notes For Reviewer

- Reject implementations that rerun training to obtain efficiency numbers.
- Reject implementations that silently compare runtime fields collected under
  different contracts as if they were normalized throughput.
- Reject tables that omit source artifact paths or source field names for
  parameter/runtime values.
- Reject candidate-lane rows unless their own completed evidence item explicitly
  authorizes paper-evidence inclusion.
- Do not write to `/home/ollie/Documents/neurips`; that remains Phase 5
  paper-facing bundle work.
