# PDEBench CNS Matched-Condition Table Refresh Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-05-04-cns-matched-condition-table-refresh`
- Date: `2026-05-04`
- Outcome: `h5_lane_selected_as_matched_condition_headline`
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-matched-condition-table-refresh/execution_plan.md`
- Item-local refresh root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-matched-condition-table-refresh/`
- Paper-local generated assets:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.tex`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.csv`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`

This refresh switches the manuscript headline CNS table from the
earlier "best observed by family" mixed-condition rendering to one
matched-condition ranking. The selector emitted a deterministic
decision payload and refreshed table assets from the completed
`history_len=5`, `40`-epoch capped CNS evidence. The capped CNS claim
boundary is unchanged.

## Selected Headline Condition

- Selected lane: `h5_512_64_64_40ep`
- Fixed contract:
  - `history_len = 5`
  - `split_counts = train=512, val=64, test=64`
  - `max_windows_per_trajectory = 8`
  - `epochs = 40`
  - `batch_size = 4`
  - `training_loss = mse`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`,
    `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
  - claim boundary: `bounded_capped_decision_support_only`
- Selection reason: the four required headline rows
  (`author_ffno_cns_base`, `spectral_resnet_bottleneck_base`,
  `fno_base`, `unet_strong`) are all complete and internally
  consistent under this single contract.
- Rejected candidate: none. The `h5` lane passed the completeness and
  consistency audit, so the locked `h2 / 2048 / 256 / 256` lane was not
  treated as a fallback.

The four headline rows on the matched `history_len=5`, `512 / 64 / 64`,
`40`-epoch slice (best aggregate first):

| Manuscript row | Repo row | err_nRMSE | fRMSE_low | fRMSE_mid | fRMSE_high |
| --- | --- | ---: | ---: | ---: | ---: |
| FFNO | `author_ffno_cns_base` | 0.0198 | 1.130 | 0.042 | 0.102 |
| SRU-Net$^*$ | `spectral_resnet_bottleneck_base` | 0.0331 | 1.854 | 0.171 | 0.262 |
| FNO | `fno_base` | 0.0384 | 2.134 | 0.122 | 0.433 |
| U-Net | `unet_strong` | 0.5386 | 30.988 | 0.645 | 1.743 |

## Source Authorities Consumed

- `history_len=5` matched-row contract and metrics:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history5_comparator_gap_fill_summary.md`
  (compare sidecar
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/compare_40ep_against_existing.json`).
- Larger-cap `history_len=2` authority preserved as bounded context:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`
  (bundle root
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/`).
- Contract authority lineage:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`.

## Capped Claim Boundary

The headline CNS table is now matched-condition, but the capped CNS
claim boundary remains `bounded_capped_decision_support_only`. The
manuscript selection moved from the `h2 / 2048` lane to the `h5 / 512`
lane; no row is relabelled `paper_grade` or `full_training`, and no
new full-training benchmark gate was opened by this refresh.

## Manuscript Figure Decision

Same-condition fixed-sample visuals are not available for the four
selected `history_len=5` rows in a single bundle. The `h5` lane lives
in three separate run roots (authored FFNO, spectral, and the gap-fill
FNO/U-Net root), and the cross-run gallery covers only the FNO and
U-Net rows. The decision payload records
`same_condition_visuals_available = false`.

The manuscript continues to display the existing
`figures/pdebench_cns_sample000_predictions.png` figure from the
earlier `h2 / 512 / 64 / 64`, 2-input-time-step bundle, but its
caption and metadata now label the figure as
`adjacent_context_only`. The figure is no longer presented as the
same-condition fixed-sample illustration of the headline ranking.

## Emitted Item-Local Refresh Root

- `matched_condition_decision.json`
- `cns_paper_table_rows.json`
- `cns_paper_table_rows.csv`
- `cns_paper_table_rows.tex`
- `source_lineage.json`
- `figure_selection.json`

## Verification

- `pytest -q tests/studies/test_paper_results_refresh.py
   tests/studies/test_pdebench_cfd_cns_metrics.py`
- `python -m compileall -q scripts/studies`
- Final-gate matched-condition input check (see plan §Required
  Deterministic Checks).

The refresh script (`scripts/studies/paper_results_refresh.py`) was
extended with a deterministic CNS matched-condition selector, an
audit-only CLI flag (`--audit-cns-matched-condition`), an item-local
asset writer (`--write-cns-matched-condition-assets`), and a
paper-local table-asset writer
(`--write-cns-matched-condition-paper-assets`). Tests cover the
happy path, missing-row fallback, inconsistent-row fallback, and the
emitted payload structure.
