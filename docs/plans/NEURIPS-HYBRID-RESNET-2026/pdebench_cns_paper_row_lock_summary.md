# PDEBench CNS Paper Row Lock Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-cns-paper-benchmark-rows`
- Date: `2026-04-29`
- Status: locked bounded capped CNS row bundle complete
- Contract authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Audit artifacts:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_row_lock_audit.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_row_lock_audit.json`
- Locked-row manifest:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`

This summary records the durable row authority for the current CNS paper lane.
It does not promote the capped lane to a same-protocol full-training benchmark
claim, and it does not create `/home/ollie/Documents/neurips/` outputs.

## Locked Contract

- Selected contract: `bounded_capped_decision_support`
- Selected history lane: `history_len=2`, `40` epochs, `512 / 64 / 64`
  trajectories, `max_windows_per_trajectory=8`, emitted windows
  `4096 / 512 / 512`
- Selected normalization contract: train-only per-field normalization fit on
  the `512` training trajectories, reused across all history slots and target
  channels, with evaluation reported in denormalized target space.
- Selected training recipe contract: keep the CNS task-local `mse` override
  relative to the design's generic `mae` baseline; use `Adam` with learning
  rate `2e-4`; use `ReduceLROnPlateau` with factor `0.5`, patience `2`,
  threshold `0.0`, and `min_lr=1e-5`; keep batch size `4`; keep the metric
  family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`.

## Locked Rows

Headline rows now locked for downstream CNS table/figure assembly:

- `spectral_resnet_bottleneck_base`
  - run root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
  - `relative_l2=0.0615620054`
  - `fRMSE_high=0.4349334538`
- `fno_base`
  - run root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
  - `relative_l2=0.0740992129`
  - `fRMSE_high=0.6717720628`
- `unet_strong`
  - run root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
  - `relative_l2=0.6757976413`
  - `fRMSE_high=1.3326253891`
- `author_ffno_cns_base`
  - run root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
  - `relative_l2=0.0281477310`
  - `fRMSE_high=0.1210141182`

Continuity/support row locked in the same bounded contract family:

- `hybrid_resnet_cns`
  - run root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
  - `relative_l2=0.0644183308`
  - `fRMSE_high=0.3683068156`
  - role: audited continuity/support only, not a required headline-table row

## Audit Result

The accepted run roots all matched the selected `history_len=2` capped contract
for official file, split counts, window cap, training loss, epoch budget, batch
size, and metric family. No missing field required a code patch or rerun in
this pass.

Each accepted root exposes:

- `invocation.json` and `invocation.sh`
- `dataset_manifest.json`
- `split_manifest.json`
- `normalization_stats_state.json`
- per-row `model_profile_*.json`
- per-row `metrics_*.json`
- sample `.npz` and `.png` artifacts

Known provenance gaps preserved by the lock:

- no repo git SHA or dirty-state artifact in the reused run roots
- no standalone run-log artifact in the reused run roots
- no standalone exit-code artifact in the reused run roots

These gaps do not block bounded capped row locking, but they do block any
attempt to relabel the reused rows as `paper_grade` or `full_training`
evidence.

## Excluded Adjacent Context

The lock keeps the following outside the headline bundle:

- `history_len=3` capped pilots
- `history_len=1` capped pilots
- GNOT rows
- repo-local `ffno_bottleneck_base`
- repo-local `ffno_bottleneck_localconv_base`

The reasons are recorded in the audit and derive from the contract-decision
authority: either the temporal contract diverges, the authored-FFNO cutoff was
not met under that alternate lane, or the row is explicitly a protocol-divergent
or proxy-only context row.

## Downstream Handoff

Downstream CNS table/figure and package-audit work should consume:

- the contract authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- the durable row-lock summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- the machine-readable manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`

Those consumers must preserve the current claim boundary verbatim:

- every locked row is `capped_decision_support`
- the bundle is coherent and same-contract
- the bundle is not a same-protocol full-training benchmark claim
