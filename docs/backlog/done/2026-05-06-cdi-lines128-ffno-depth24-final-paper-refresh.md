---
priority: 9
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/execution_plan.md
check_commands:
  - python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json >/tmp/model_variant_index.json.valid
  - python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json >/tmp/ablation_index.json.valid
  - python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json >/tmp/cdi_lines128_metrics_extended.json.valid
prerequisites:
  - 2026-05-06-cdi-lines128-ffno-depth24-ablation
  - 2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun
  - 2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The four-block no-refiner refresh is the interim paper-asset update; depth-24 final paper updates should wait until after that cheaper wave lands.
  - Final depth-24 promotion requires both PINN and supervised depth-24 rows under the same no-refiner Lines128 contract.
  - This item regenerates final paper-facing CDI FFNO tables, figures, and indexes only; it does not train new models.
---

# Backlog Item: Refresh Final CDI Lines128 Paper Assets With Depth-24 No-Refiner FFNO Rows

## Objective

- Regenerate final CDI Lines128 paper-facing metrics, figures/images,
  model-configuration, efficiency, evidence-index, and manuscript-consumption
  assets after depth-24 no-refiner FFNO rows land.
- Decide explicitly whether depth-24 no-refiner rows replace the interim
  four-block no-refiner rows in final paper tables, or remain append-only
  ablation evidence if they do not improve the relevant metrics.

## Scope

- Consume:
  - `2026-05-06-cdi-lines128-ffno-depth24-ablation`;
  - `2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun`;
  - interim four-block refresh
    `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh`.
- Reuse all non-FFNO Lines128 rows by lineage. Do not rerun non-FFNO rows.
- Regenerate final CDI FFNO paper assets only after verifying:
  - depth-24 PINN row has `fno_blocks=24`, `fno_cnn_blocks=0`;
  - depth-24 supervised row has `fno_blocks=24`, `fno_cnn_blocks=0`;
  - four-block no-refiner interim rows remain discoverable as comparison
    context.

## Outputs

- Final depth-24-aware versions of CDI Lines128 metrics/table/figure assets,
  with filenames chosen to avoid overwriting interim assets without provenance.
- Updated:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`;
  - `docs/studies/index.md`.
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`.

## Completion Gate

- Any final manuscript-facing CDI row labeled `FFNO + PINN` or
  `FFNO + supervised` must point to the chosen final no-refiner artifact root
  and must state whether it is four-block or depth-24.
- Historical `fno_cnn_blocks=2` rows must remain labeled `FFNO-local proxy`.
- Four-block no-refiner rows must remain discoverable as interim evidence and
  as the direct refiner-effect comparison.
- The refresh must not rerun or mutate non-FFNO rows.

## Notes For Reviewer

- Reject implementations that silently replace four-block rows with depth-24
  rows without a metric-based promotion decision.
- Reject implementations that make final paper figures/images consume
  historical `fno_cnn_blocks=2` proxy roots as canonical FFNO.
- Reject implementations that rerun completed non-FFNO Lines128 rows.
