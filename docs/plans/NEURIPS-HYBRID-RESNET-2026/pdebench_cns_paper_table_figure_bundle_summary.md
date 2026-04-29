# PDEBench CNS Paper Table/Figure Bundle Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-cns-paper-table-figure-bundle`
- Date: `2026-04-29`
- Outcome: `fallback_512_bundle_used`
- Bundle root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/`
- Contract authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Locked-row authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- Authoritative manifest consumed:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`

This summary records the durable CNS paper bundle assembly. It does not widen
the paper contract, does not create `/home/ollie/Documents/neurips/` outputs,
and does not relabel capped rows as `paper_grade` or `full_training`.

## 1024 Audit Outcome

The bundle audited the preferred same-contract `1024 / 128 / 128`,
`history_len=2`, `40`-epoch lane before table assembly and wrote:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/1024_same_cap_audit.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/1024_same_cap_audit.md`

Audit result:

- outcome: `fallback_to_512_required`
- same-contract `1024` row recovered:
  `spectral_resnet_bottleneck_base`
- same-contract `1024` headline rows still missing:
  `fno_base`, `unet_strong`, `author_ffno_cns_base`
- rerun commands were emitted for those missing rows, but the checked-in bundle
  remains on the existing authoritative `512 / 64 / 64` lock until a widened
  contract and row-lock are actually checked in

## Final Bundle Contract

- selected bundle lane: `512 / 64 / 64` trajectories
- history lane: `history_len=2`
- epoch budget: `40`
- window cap: `max_windows_per_trajectory=8`
- training loss: `mse`
- batch size: `4`
- claim boundary: `capped_decision_support_only`

Headline roster:

- `spectral_resnet_bottleneck_base`
- `fno_base`
- `unet_strong`
- `author_ffno_cns_base`

Continuity handling:

- `hybrid_resnet_cns` stays separated from the headline table as a
  continuity/support row
- because the fallback bundle used the same `512 / 64 / 64` contract for every
  included figure row, the continuity row is allowed in the shared-scale visual
  bundle

## Fixed Samples And Shared Scales

- fixed sample ids:
  `0`
- field order:
  `density`, `Vx`, `Vy`, `pressure`
- shared value scales:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/shared_field_scales.json`
- shared absolute-error scales:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/shared_error_scales.json`
- fixed-sample manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/fixed_sample_manifest.json`

Scale policy:

- one shared value scale per field across the common target plus every included
  prediction for that field
- one shared zero-based absolute-error scale per field across every included
  row for that field
- signed velocity fields keep symmetric limits; scalar fields use the combined
  min/max range

## Emitted Bundle Artifacts

Tables:

- JSON:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/cns_paper_table_rows.json`
- CSV:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/cns_paper_table_rows.csv`
- TeX:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/cns_paper_table_rows.tex`

Figures and source arrays:

- copied source arrays:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/figure_sources/`
- rendered panels:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/figures/`
- figure manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/figure_manifest.json`

Validation:

- bundle validation:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/bundle_validation.json`
- bundle input manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/bundle_input_manifest.json`

## Provenance Boundary

The bundle preserves the existing capped-evidence provenance gaps noted by the
row lock:

- no standalone repo git SHA artifact in the reused run roots
- no standalone dirty-state artifact in the reused run roots
- no standalone run log artifact in the reused run roots
- no standalone exit-code artifact in the reused run roots
- no precise accelerator string in the reused runtime artifacts, so the table
  records an explicit `artifact_missing_precise_accelerator` note instead of
  guessing hardware

These gaps do not block bounded capped table/figure assembly, but they still
prevent any `paper_grade` or full-training claim.
