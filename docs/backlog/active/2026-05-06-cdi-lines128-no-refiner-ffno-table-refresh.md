---
priority: 3
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/execution_plan.md
check_commands:
  - python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json >/tmp/model_variant_index.json.valid
  - python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json >/tmp/ablation_index.json.valid
  - python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json >/tmp/cdi_lines128_metrics_extended.json.valid
prerequisites:
  - 2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun
  - 2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - Existing paper tables still contain historical CDI FFNO-local-refiner proxy rows.
  - Corrected no-refiner FFNO rows must be promoted by lineage before manuscript tables can call them pure FFNO.
  - This item regenerates tables and indexes only; it does not train new models.
---

# Backlog Item: Refresh CDI Lines128 Tables With Corrected No-Refiner FFNO Rows

## Objective

- Regenerate CDI Lines128 paper-facing metrics, model-configuration, efficiency,
  evidence-index, and manuscript-consumption assets after corrected no-refiner
  FFNO rows land.
- Replace historical `FFNO-local proxy` rows in paper-facing tables by lineage
  with corrected no-refiner rows.
- Preserve old `fno_cnn_blocks=2` rows in evidence indexes as caveated
  historical context.

## Scope

- Consume corrected rows from:
  - `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`;
  - `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`.
- Reuse all non-FFNO Lines128 rows by lineage from the completed U-NO extension
  and completed primary table.
- Regenerate:
  - `tables/cdi_lines128_metrics_extended.{json,csv,tex}`;
  - objective-control TeX table;
  - `tables/model_config_by_benchmark.{json,csv,tex}`;
  - `tables/paper_efficiency_table.{json,csv,tex}` if parameter/runtime sources
    change;
  - `model_variant_index.json`, `ablation_index.json`, `evidence_matrix.md`,
    `paper_evidence_index.md`, and `docs/studies/index.md`.

## Completion Gate

- Any manuscript-facing CDI row labeled `FFNO + PINN` or `FFNO + supervised`
  must point to a corrected no-refiner artifact root.
- Historical `fno_cnn_blocks=2` rows must remain labeled `FFNO-local proxy` or
  equivalent caveated context.
- The refresh must not rerun or mutate non-FFNO rows.
