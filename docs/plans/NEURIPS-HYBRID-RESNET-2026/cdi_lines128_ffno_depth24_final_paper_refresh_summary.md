# CDI Lines128 FFNO Depth-24 Final Paper Refresh Summary

- Date: `2026-05-07`
- Backlog item: `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/execution_plan.md`
- State: `completed`
- Claim boundary: `complete_lines128_cdi_benchmark_plus_uno_extension_with_final_four_block_no_refiner_ffno_pair`
- Artifact root:
  `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/`

## Objective

Decide whether the completed depth-24 no-refiner FFNO pair should replace the
corrected four-block no-refiner pair in the repo-local final CDI `lines128`
paper assets, then regenerate the final FFNO-consuming tables, figures,
model-config packaging, efficiency packaging, and discovery surfaces without
rerunning or mutating any non-FFNO `lines128` row.

## Consumed Authorities

- Prior repo-local CDI FFNO packaging authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`
- Corrected four-block pure-FFNO PINN source row:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`
- Corrected four-block pure-FFNO supervised source row:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md`
- PINN depth-24 append-only ablation authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
- Supervised depth-24 append-only companion authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_depth24_no_refiner_summary.md`

## Promotion Decision

- Chosen final pair: `four_block_no_refiner`
- Final output stem: `ffno_final_depth4pair`
- Active repo-local paper rows remain:
  - `pinn_ffno` from
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`
  - `supervised_ffno` from
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z`

Decision rationale:

- The depth-24 PINN row is stronger on every tracked scalar reconstruction
  metric, but it is much more expensive (`124,968 -> 701,628` parameters,
  `873.742 -> 4,754.923` train seconds, `1.231 -> 8.505` inference seconds).
- The supervised depth-24 companion is mixed rather than clearly better:
  phase-side metrics improve modestly, but amplitude SSIM/FRC regress,
  validation loss worsens, and runtime still expands by more than `5x`.
- The manuscript-facing FFNO objective-control pair must remain same-depth.
  Because the supervised half does not justify pair-level promotion, the final
  repo-local CDI FFNO package keeps the corrected four-block pair canonical and
  retains the depth-24 family as append-only evidence.

## Preserved Lineage

- Historical `fno_cnn_blocks=2` proxy lineage remains preserved only as
  explicit `FFNO-local proxy` context:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`
- Depth-24 append-only evidence remains discoverable but not promoted:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/runs/supervised_ffno_depth24_20260507T192840Z`
- Non-FFNO rows remain reused strictly by lineage from:
  - immutable six-row CDI base:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
  - append-only U-NO extension:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z`

## Regenerated Assets

Canonical compatibility outputs:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.{json,csv,tex}`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_pinn_metrics.tex`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.tex`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet.png`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet_per_panel_scaled.png`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.{json,csv,tex}`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.{json,csv,tex}`

Versioned provenance-safe copies:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended_ffno_final_depth4pair.{json,csv,tex}`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_pinn_metrics_ffno_final_depth4pair.tex`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison_ffno_final_depth4pair.tex`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet_ffno_final_depth4pair.png`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet_per_panel_scaled_ffno_final_depth4pair.png`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark_ffno_final_depth4pair.{json,csv,tex}`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table_ffno_final_depth4pair.{json,csv,tex}`

## Verification

Archived under:
`artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/`

- Refresh command:
  `paper_results_refresh_ffno_final_depth4pair.log`
- Focused pytest batches (preflight + postfix after row-level provenance fix):
  - `pytest_preflight.log` / `pytest_postfix.log`
  - `pytest_model_config_preflight.log` / `pytest_model_config_postfix.log`
  - `pytest_efficiency_preflight.log` / `pytest_efficiency_postfix.log`
- Collect-only proof for the changed test module:
  - `pytest_collect.log`
- Compile smokes:
  - `compileall_preflight.log` / `compileall_postfix.log`
- Canonical JSON pretty-print proofs:
  - `json_paper_evidence_manifest.log`,
    `json_model_variant_index.log`,
    `json_ablation_index.log`,
    `json_cdi_lines128_metrics_extended.log`,
    `json_model_config_by_benchmark.log`,
    `json_paper_efficiency_table.log`
- Machine-readable checks ledger:
  `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`
  (16 commands)

The regenerated `tables/cdi_lines128_metrics_extended.json` carries per-row
provenance on the active FFNO rows: `pinn_ffno` and `supervised_ffno` each
record `final_ffno_pair_key`, `final_ffno_depth_label`, `claim_boundary`,
`source_root`, and `source_metrics_json`; `supervised_ffno` additionally
records `historical_proxy_lineage` pointing at the historical
`fno_cnn_blocks=2` FFNO-local proxy metrics. The same row-level provenance is
mirrored in the versioned
`tables/cdi_lines128_metrics_extended_ffno_final_depth4pair.json`.

Deterministic source-comparison standard consumed from the depth studies:

- config and invocation fields: exact equality on the listed locked-contract
  fields
- dataset payloads: exact equality for every NPZ key after dropping only
  `_metadata.creation_info.timestamp`

## Residual Risks

- The complete six-row `lines128` CDI bundle remains the primary paper-grade
  headline authority. This final refresh only governs the repo-local FFNO
  paper assets layered on top of that pillar.
- The depth-24 result is still a single-seed, two-image-per-split directional
  study. It remains useful evidence for architecture directionality, not for a
  broad robustness claim.
- Manuscript prose and TeX comments are intentionally out of scope here; this
  refresh updates repo-local assets and discoverability surfaces only.
