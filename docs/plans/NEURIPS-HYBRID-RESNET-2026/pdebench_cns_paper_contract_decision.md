# PDEBench CNS Paper Contract Decision

## Context And Authority

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-cns-paper-contract-decision`
- Date: `2026-04-29`
- Status: approved decision
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/execution_plan.md`
- Governing design/package docs:
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Supporting audit artifacts:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_full_training_cost_estimate.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_full_training_cost_estimate.json`

This document decides the authoritative CNS paper-evidence contract before the
row-lock and table/figure backlog items run.

## Audited Contract Lanes

- Lane A is the only complete same-contract headline candidate today:
  - official file:
    `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - split counts: `512 / 64 / 64` trajectories
  - history contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
  - cap: `max_windows_per_trajectory=8`
  - emitted windows: `4096 / 512 / 512`
  - normalization: train-only per-field normalization fit on the training split,
    reused across history slots and target fields, with metrics reported in
    denormalized target space
  - recipe: task-local `mse` override, `Adam`, learning rate `2e-4`,
    `ReduceLROnPlateau` with factor `0.5`, patience `2`, threshold `0.0`,
    `min_lr=1e-5`, batch size `4`
  - completed same-contract rows:
    `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`,
    `unet_strong`, `author_ffno_cns_base`
- Lane B is promising but incomplete for the paper headline bundle:
  - same official file, cap family, normalization contract, metric family, and
    recipe family as Lane A
  - history contract changes to `history_len=3`
  - completed rows:
    `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`,
    `unet_strong`
  - missing same-contract authored FFNO row at the paper cutoff
- Adjacent but non-headline context:
  - `history_len=1` pilot rows: contract-divergent temporal context only
  - repo-local `ffno_bottleneck_*` rows: proxy FFNO-family context only, not
    authored FFNO substitutes
  - GNOT rows: optional protocol-divergent context only because they depend on
    `ptycho311_2`, `AdamW`, `OneCycleLR`, and a separate paper-default recipe

Selected contract: `bounded_capped_decision_support`
Selected history lane: `history_len=2`, `40` epochs, `512 / 64 / 64` trajectories, `max_windows_per_trajectory=8`, emitted windows `4096 / 512 / 512`
Selected normalization contract: train-only per-field normalization fit on the `512` training trajectories, reused across all history slots and target channels, with evaluation reported in denormalized target space.
Selected training recipe contract: keep the CNS task-local `mse` override relative to the design's generic `mae` baseline; use `Adam` with learning rate `2e-4`; use `ReduceLROnPlateau` with factor `0.5`, patience `2`, threshold `0.0`, and `min_lr=1e-5`; keep batch size `4`; keep the metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`.

## Compute And Deadline Rationale

The selected `history_len=2` capped lane is the only lane that already has the
required same-contract model families for a coherent paper table:

- best current Hybrid-family / Hybrid-spectral row
- local FNO baseline
- local U-Net baseline
- authored FFNO row

Selecting `history_len=3` now would force the paper contract either to omit
authored FFNO or to reopen the queue with another external-baseline run before
the table can lock.

The full-training cost estimate derived from the completed `40`-epoch capped
roots scales the current `512 / 64 / 64`, `8`-window lane to the full
`8000 / 1000 / 1000` trajectory split and uncapped raw `history_len=2`
window count (`19` windows per trajectory). That estimate implies:

- scale factor from capped train windows to full-train windows: `37.109375x`
- core four-row full-training wall time:
  `~93.72` GPU-hours (`~3.90` days) for
  `spectral_resnet_bottleneck_base + fno_base + unet_strong + author_ffno_cns_base`
- five-row wall time including a same-budget `hybrid_resnet_cns` hedge:
  `~102.85` GPU-hours (`~4.29` days)

That estimate excludes failed-run recovery, row-lock packaging, table/figure
assembly, and any follow-up needed if the full split changes row ordering. On
one RTX 3090, it is not credible to imply immediate full-training CNS benchmark
promotion before the 2026-05-04 AOE abstract and 2026-05-06 AOE paper
deadlines.

## Required CNS Rows

- Headline row status for this contract: `capped_decision_support`
- Required headline rows:
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
- Continuity/support row only:
  - `hybrid_resnet_cns`
    - run root:
      `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
    - `relative_l2=0.0644183308`
    - `fRMSE_high=0.3683068156`
- Explicitly outside the headline bundle:
  - `history_len=3` pilot rows
  - `history_len=1` pilot rows
  - repo-local `ffno_bottleneck_base`
  - repo-local `ffno_bottleneck_localconv_base`
  - GNOT rows

Authored FFNO cutoff: include authored FFNO only from the already completed same-contract `history_len=2`, `40`-epoch row recorded in this decision; do not wait for or imply any `history_len=3` authored rerun for the current headline table.
Authored FFNO status: `completed`; accepted row id `author_ffno_cns_base`; accepted run root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`.

## Claim Boundary

The paper may use CNS as bounded evidence that Hybrid-family / spectral-family
behavior transfers to PDEBench CNS under one fixed local capped contract.

Allowed CNS claim surface under this decision:

- the headline table is a coherent same-contract capped comparison
- the selected row roster is auditable and ready for row locking
- `history_len=3` may be discussed only as adjacent capped context

Forbidden claims under this decision:

- same-protocol full-training CNS benchmark competitiveness
- that `history_len=3` is the locked headline contract
- that GNOT or repo-local FFNO proxies belong to the same headline row family
  as the selected authored FFNO row

## Stop / Failure Criteria

- Stop this contract lane instead of silently widening it if any later item
  needs to change the official file, split counts, history length,
  normalization contract, training recipe contract, or authored-FFNO cutoff.
- Treat missing same-contract authored FFNO evidence outside the accepted run
  root as a row-level blocker for any alternative headline lane, not as a
  reason to substitute repo-local FFNO proxies.
- Treat any attempt to relabel the capped lane as a full-training benchmark
  without a checked-in contract amendment as a failure of the paper contract.
- Keep GNOT outside the required headline bundle unless a later approved
  decision explicitly widens the contract and records its protocol caveats.

## Downstream Handoff

- For `docs/backlog/active/2026-04-29-cns-paper-benchmark-rows.md`:
  - reuse the exact selected contract, history lane, normalization contract,
    and training recipe contract lines above verbatim
  - lock the bounded headline table to
    `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`,
    `author_ffno_cns_base`
  - keep `hybrid_resnet_cns` as continuity/support only
  - do not rerun rows merely to chase the stronger but incomplete
    `history_len=3` lane
- For `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
  and `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`:
  - treat this document as the discoverable authority for the full-vs-capped
    choice, selected history lane, normalization contract, training recipe
    contract, and authored-FFNO cutoff/status
- For `docs/backlog/active/2026-04-29-paper-evidence-package-audit.md`:
  - preserve the CNS pillar as `capped_decision_support`, not `full_training`
  - preserve the selected row roster and claim boundary verbatim
- For later contract changes:
  - require a checked-in amendment or a new decision document rather than
    silent table drift
