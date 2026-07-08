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
  - Corrected four-block no-refiner FFNO rows must be promoted by lineage before near-term manuscript figures, images, and metrics can stop relying on the historical local-refiner proxy.
  - This is the cheap first refresh wave: it regenerates tables, figures, and indexes from four-block no-refiner rows only and does not train new models.
  - Depth-24 no-refiner rows remain a later full-results wave and must not block this refresh.
---

# Backlog Item: Refresh CDI Lines128 Tables With Four-Block No-Refiner FFNO Rows

## Objective

- Regenerate CDI Lines128 paper-facing metrics, figure/image inputs,
  model-configuration, efficiency, evidence-index, and manuscript-consumption
  assets after corrected four-block no-refiner FFNO rows land.
- Replace historical `FFNO-local proxy` rows in paper-facing tables by lineage
  with corrected `fno_blocks=4`, `fno_cnn_blocks=0` rows.
- Preserve old `fno_cnn_blocks=2` rows in evidence indexes as caveated
  historical context.
- Treat this as the interim paper-asset refresh. The final paper-results refresh
  after depth-24 rows land is owned by
  `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`.

## Scope

- Consume corrected rows from:
  - `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`;
  - `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`.
- Both consumed rows must remain the cheap default CDI FFNO depth:
  `fno_blocks=4`, `fno_modes=12`, `fno_width=32`, `fno_cnn_blocks=0`.
- Reuse all non-FFNO Lines128 rows by lineage from the completed U-NO extension
  and completed primary table.
- Regenerate:
  - `tables/cdi_lines128_metrics_extended.{json,csv,tex}`;
  - objective-control TeX table;
  - figure/image source manifests or panels that currently consume the
    historical FFNO-local-refiner proxy row;
  - `tables/model_config_by_benchmark.{json,csv,tex}`;
  - `tables/paper_efficiency_table.{json,csv,tex}` if parameter/runtime sources
    change;
  - `model_variant_index.json`, `ablation_index.json`, `evidence_matrix.md`,
    `paper_evidence_index.md`, and `docs/studies/index.md`.

## Completion Gate

- Any manuscript-facing CDI row labeled `FFNO + PINN` or `FFNO + supervised`
  in this interim refresh must point to a corrected four-block no-refiner
  artifact root.
- Historical `fno_cnn_blocks=2` rows must remain labeled `FFNO-local proxy` or
  equivalent caveated context.
- Depth-24 rows must not be required or silently substituted in this item.
- The refresh must not rerun or mutate non-FFNO rows.
