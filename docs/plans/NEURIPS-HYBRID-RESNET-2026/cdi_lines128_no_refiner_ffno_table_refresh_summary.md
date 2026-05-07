# CDI Lines128 No-Refiner FFNO Table Refresh Summary

- Date: `2026-05-06`
- Backlog item: `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh`
- State: `completed`
- Claim boundary: `complete_lines128_cdi_benchmark_plus_uno_extension_with_corrected_ffno_objective_control_pair`
- Artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/`

## Objective

Refresh the manuscript-facing `lines128` CDI tables, figure inputs, model-config
packaging, efficiency packaging, and discovery surfaces so the active
`FFNO + PINN` and `FFNO + supervised` paper rows consume the corrected
four-block no-refiner reruns rather than the historical `fno_cnn_blocks=2`
local-refiner proxy lineage.

## Source Authorities

Corrected active FFNO roots:

- corrected `pinn_ffno`:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`
- corrected `supervised_ffno`:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z`

Reused non-FFNO lineage authorities:

- immutable six-row base summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  with authoritative root
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- append-only U-NO extension summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
  with authoritative root
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z`
- preserved historical supervised/local-refiner proxy context:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`

## Completed In This Pass

- Patched `scripts/studies/paper_results_refresh.py` so:
  - active CDI FFNO table rows render as `FFNO`, not `FFNO-local proxy`
  - `pinn_ffno` metrics come from the corrected no-refiner rerun
  - `supervised_ffno` metrics come from the corrected no-refiner rerun
  - the default CDI phase-zoom figure input for `pinn_ffno` now reads the
    corrected no-refiner reconstruction root
  - the extended CDI metrics JSON now states the honest mixed-lineage table
    claim boundary and records both corrected FFNO source paths plus the
    preserved historical supervised proxy source path
  - `render_cdi_objective_comparison_table()` now restricts the rendered
    paper-facing objective-control table to the corrected FFNO pair only
    (`CDI_OBJECTIVE_CONTROL_ACTIVE_MODELS = ("FFNO",)`); CNN, FNO, U-NO, and
    SRU-Net rows are no longer emitted into `cdi_lines128_objective_comparison.tex`,
    and the renderer raises `ValueError` if the active FFNO pair is missing.
- Patched `scripts/studies/paper_model_config_table.py` so the active Synthetic
  CDI FFNO config rows consume the corrected no-refiner roots and surface the
  no-refiner factorized-Fourier contract in the table payload.
- Patched `scripts/studies/paper_efficiency_table.py` so the active Synthetic
  CDI FFNO runtime rows consume the corrected manifests, yielding the real
  `train_wall_time_sec` values from the reruns instead of the stale proxy
  runtime provenance.
- Added focused regressions covering:
  - corrected CDI FFNO labels in the extended metrics table payload
  - corrected default phase-zoom `pinn_ffno` recon source
  - corrected CDI model-config root selection for both FFNO rows
  - corrected CDI efficiency parameter counts and model labels
- Regenerated paper-local CDI assets:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.{json,csv,tex}`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.tex`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_pinn_metrics.tex`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet.png`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet_per_panel_scaled.png`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.{json,csv,tex}`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.{json,csv,tex}`
- Updated the durable discovery surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`

## Active Packaging Outcome

- The current paper-local CDI FFNO rows are now:
  - `pinn_ffno` -> `FFNO + PINN` from the corrected no-refiner rerun
  - `supervised_ffno` -> `FFNO + supervised` from the corrected no-refiner rerun
- Historical `fno_cnn_blocks=2` FFNO rows remain preserved only as explicit
  proxy lineage for provenance, not as active manuscript-facing CDI rows.
- No depth-24 FFNO row was consumed.
- The table-level claim boundary changed from the append-only U-NO extension
  boundary to the honest mixed-lineage paper-refresh boundary
  `complete_lines128_cdi_benchmark_plus_uno_extension_with_corrected_ffno_objective_control_pair`.

## Verification

- Refresh command:
  - `python scripts/studies/paper_results_refresh.py --write-cdi-extended-assets --write-cdi-phase-zoom-figure --write-cdi-phase-zoom-per-panel-figure --write-model-config-table --write-efficiency-table`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/verification/paper_results_refresh.log`
- Focused generator tests:
  - `pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom'`
  - `pytest -q tests/studies/test_paper_model_config_table.py`
  - `pytest -q tests/studies/test_paper_efficiency_table.py`
- Compile gate:
  - `python -m compileall -q scripts/studies`
- Verification and audit artifacts:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/verification/stale_consumer_audit.md`

## Residual Risks

- The manuscript draft prose and embedded figure-metadata comments in
  `hybrid_resnet_neurips_first_draft.tex` still use historical
  `FFNO-local refiner` wording and old figure-source comments. This refresh
  intentionally stopped at the paper-local asset and discoverability layer; a
  separate manuscript-text pass is still required before paper prose can claim
  the corrected FFNO rows directly.
- The model-config and efficiency tables intentionally use the deduped effective
  prediction-model count from the saved state dict (`124,966`) while preserving
  the raw recorded row-manifest count (`124,968`) in
  `model_config_by_benchmark.json` for the corrected `pinn_ffno` row.
