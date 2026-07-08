---
priority: 17
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-cns-inference-throughput-probes/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json"),
        Path("scripts/studies/paper_efficiency_table.py"),
        Path("scripts/studies/paper_results_refresh.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing inference-throughput probe inputs: {missing}")
    print("inference-throughput probe inputs present")
    PY
  - python -m compileall -q scripts/studies ptycho_torch
prerequisites:
  - 2026-05-05-paper-efficiency-table
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The current paper efficiency table has parameter counts for CDI Lines128 and PDEBench CNS, but their inference-throughput fields are explicitly `missing`.
  - BRDT throughput is already measured from existing 40-epoch evaluation runtime; this item should not rerun or reinterpret BRDT.
  - Adding throughput columns to CDI/CNS paper tables requires a common inference-only protocol, not heterogeneous training runtime fields.
---

# Backlog Item: Measure CDI And CNS Inference Throughput For Paper Efficiency Tables

## Objective

- Run lightweight, deterministic inference-only probes for the CDI Lines128 and
  PDEBench CNS rows that currently have missing throughput in the paper
  efficiency table, then regenerate the repo-local efficiency assets with
  measured throughput fields.

## Scope

- Consume the existing paper-authority rows:
  - CDI Lines128 rows from the complete table plus U-NO extension;
  - PDEBench CNS matched-condition rows from the `history_len=5`,
    `512 / 64 / 64`, 40-epoch capped table.
- Load existing checkpoints, dataset contracts, model configs, and split
  manifests by lineage. Do not train, tune, resume training, or alter row
  selection.
- Define one benchmark-local inference timing protocol per benchmark:
  - fixed warmup count;
  - fixed timed iteration count or fixed full-test-pass count;
  - batch size and device recorded;
  - CUDA synchronization before and after timed regions when CUDA is used;
  - wall-clock timing source recorded;
  - sample count, batch count, and seconds recorded.
- Emit item-local probe artifacts under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-cns-inference-throughput-probes/`,
  including JSON/CSV timing rows, invocation/provenance metadata, and any
  skipped-row reasons.
- Extend `scripts/studies/paper_efficiency_table.py` so regenerated
  `paper_efficiency_table.{json,csv,tex}` consumes these measured CDI/CNS
  throughput fields.
- Update the durable summary and evidence indexes to state which rows have
  measured inference throughput and which, if any, remain missing or
  not-comparable.

## Required Interpretation

- This is an inference-only measurement item. No new training is authorized.
- Training runtime fields such as `runtime_sec` or `command_wall_time_sec` must
  not be converted into throughput.
- CDI and CNS throughput may be reported within their own benchmark groups, but
  must not be used to create a cross-benchmark model ranking.
- If a checkpoint or dataset lineage is missing for a row, record that row as
  `not_comparable` or `missing_checkpoint` rather than substituting another row.
- BRDT rows should be carried through from the existing approved secondary
  evidence unchanged.

## Outputs

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-cns-inference-throughput-probes/throughput_probe_results.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-cns-inference-throughput-probes/throughput_probe_results.csv`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-cns-inference-throughput-probes/probe_provenance.json`
- Regenerated `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
- Regenerated `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv`
- Regenerated `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`
- Updated `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md`
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`

## Completion Gate

- The regenerated efficiency table must show `inference_throughput_status =
  measured` for every CDI/CNS row whose checkpoint and dataset lineage can be
  recovered under the current paper authority.
- Any non-measured CDI/CNS row must include a concrete skipped-row reason in the
  item-local JSON artifact.
- The summary must explicitly say whether CDI and CNS headline tables are ready
  to receive throughput columns, or whether throughput should remain confined to
  the standalone efficiency table.

## Notes For Reviewer

- Reject implementations that train, tune, or change row selection to obtain
  throughput numbers.
- Reject implementations that time data generation, checkpoint loading, or
  visualization as inference throughput.
- Reject implementations that omit CUDA synchronization around timed inference
  on GPU.
- Reject implementations that overwrite existing CDI/CNS metric authorities
  rather than adding item-local throughput evidence and regenerating the
  efficiency table by lineage.
