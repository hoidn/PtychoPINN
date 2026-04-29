# PDEBench CNS Paper Contract Decision

## Status And Authority

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
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_full_training_cost_estimate.md`

This document decides the authoritative CNS paper-evidence contract before the
row-lock and table/figure backlog items run.

## Decision Summary

Selected contract: `bounded_capped_decision_support`

Selected headline lane:

- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- task: `2d_cfd_cns`
- history contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- split counts: `512 / 64 / 64` trajectories
- cap rule: `max_windows_per_trajectory=8`
- emitted windows: `4096 / 512 / 512`
- training loss: `mse`
- batch size: `4`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`
- evaluation space: denormalized target space
- row status label for the headline table: `capped_decision_support`

Selected headline row roster:

- `spectral_resnet_bottleneck_base` as the best current Hybrid-family /
  Hybrid-spectral row
- `fno_base` as the required local FNO baseline
- `unet_strong` as the required local CNN/U-Net-style baseline
- `author_ffno_cns_base` as the authored external FFNO row

Carry-forward continuity row:

- `hybrid_resnet_cns` remains an audited same-contract continuity row and may
  appear in sidecars or discussion, but it is not required in the locked
  four-row headline table under this decision.

Rows explicitly outside the headline contract:

- `history_len=3` pilot rows: adjacent capped context only
- `history_len=1` pilot rows: adjacent capped context only
- repo-local `ffno_bottleneck_*` proxy rows: not authored FFNO substitutes
- GNOT rows: optional protocol-divergent context only

## Why This Contract Was Chosen

### 1. It is the only complete same-contract lane today

The selected `history_len=2` capped lane already has completed same-contract
rows for the exact model families the paper needs:

- best Hybrid-family row
- local FNO
- local U-Net
- authored FFNO

The stronger local `history_len=3` `40`-epoch pilots do not yet have an authored
FFNO row under the same contract, so selecting `history_len=3` now would force
the paper contract to mix incomplete external-baseline coverage into the
headline table.

### 2. It keeps the fairness boundary explicit

The selected lane fixes:

- same official file
- same split counts
- same `history_len`
- same loss
- same metric family
- same capped status

This satisfies the steering requirement not to silently relax fairness rules.

### 3. Full-training is not credible on the current deadline/hardware budget

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

That estimate excludes failed-run recovery, manifest/table work, and any
follow-up needed if the full split changes row ordering. On one RTX 3090, this
is too large to imply as an immediate paper benchmark before the 2026-05-04 AOE
abstract and 2026-05-06 AOE paper deadlines.

## Locked Headline Contract

### Contract Fields

- evidence class: `bounded_capped_decision_support`
- task id: `2d_cfd_cns`
- profile family authority date: `2026-04-29`
- dataset file:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- field order: `density`, `Vx`, `Vy`, `pressure`
- history length: `2`
- sample contract: `concat u[t-2:t] -> u[t]`
- input channels: `8`
- target channels: `4`
- split counts: `train=512`, `val=64`, `test=64` trajectories
- cap: `max_windows_per_trajectory=8`
- emitted windows: `train=4096`, `val=512`, `test=512`
- batch size: `4`
- epochs for locked row reuse: `40`
- training loss: `mse`
- optimizer family: `Adam`
- scheduler family: `ReduceLROnPlateau`
- plateau floor ceiling: `min_lr <= 1e-5`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`
- claim boundary:
  `decision_support_not_benchmark_performance`

### Locked Row Roots

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

Continuity-only same-contract row:

- `hybrid_resnet_cns`
  - run root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
  - `relative_l2=0.0644183308`
  - `fRMSE_high=0.3683068156`

## Authored-FFNO Cutoff And Status

Authored FFNO decision:

- inclusion cutoff: the row-lock item may include authored FFNO only from the
  already completed same-contract `history_len=2`, `40`-epoch row recorded in
  this decision
- current status at cutoff: `completed`
- accepted authored row id: `author_ffno_cns_base`
- accepted authored row root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`

Implications:

- The headline CNS table may include authored FFNO now because the selected
  contract matches the completed author lane.
- The row-lock item must not wait for or imply a `history_len=3` authored FFNO
  rerun.
- Repo-local `ffno_bottleneck_base` and `ffno_bottleneck_localconv_base` remain
  decision-support proxy rows only and must not be substituted for the authored
  FFNO row.

## Rejected Alternatives

### Rejected: `full_training_paper_benchmark`

Reason:

- no same-contract full-training CNS row bundle exists today
- estimated sequential GPU time is `~3.90` to `~4.29` days on one RTX 3090
  before recovery risk
- this would push the paper into a deadline-sensitive compute gamble rather than
  a clean bounded-evidence claim

### Rejected: headline `history_len=3`

Reason:

- local `40`-epoch results are promising, especially for
  `spectral_resnet_bottleneck_base`
- but there is no authored FFNO row under the same `history_len=3` contract
- the selected contract would therefore either omit authored FFNO or reopen the
  queue with another external-baseline run before the table can lock

Use of `history_len=3` after this decision:

- keep it as adjacent capped context in discussion and follow-up summaries
- do not silently mix it into the headline table
- if later promoted, require a checked-in amendment or a new contract decision

## Downstream Instructions

### For `2026-04-29-cns-paper-benchmark-rows`

- Freeze the bounded headline table to the `history_len=2`, `40`-epoch capped
  lane defined here.
- Use these headline rows:
  - `spectral_resnet_bottleneck_base`
  - `fno_base`
  - `unet_strong`
  - `author_ffno_cns_base`
- Keep `hybrid_resnet_cns` as an optional continuity/support row only.
- Do not rerun rows merely to chase the stronger but incomplete `history_len=3`
  lane in this pass.

### For `2026-04-29-cns-paper-table-figure-bundle`

- Label the entire CNS bundle as `bounded_capped_decision_support`.
- State explicitly that the headline lane is capped and not a full-training
  benchmark.
- If discussion references `history_len=3`, label it as adjacent capped context
  rather than as part of the headline table.

### For `2026-04-29-paper-evidence-package-audit`

- Record the CNS pillar as `capped_decision_support`, not `full_training`.
- Preserve the selected row roster and claim boundary verbatim.
- Do not collapse the capped CNS decision into the CDI paper-grade labels.

## Comparison Standards Used In This Decision

- headline-lane selection standard:
  same dataset, same split counts, same `history_len`, same normalization,
  same training loss, same metric family, same capped/full-training status
- cross-history sidecar standard from the existing longer-context artifacts:
  `Only history_len and its derived sample/input-channel contract may differ.`
- cross-run gallery alignment standard where applicable in the audited payloads:
  `np.allclose(..., atol=1e-6, rtol=1e-6)`

## Claim Boundary

The paper may use CNS as bounded evidence that the Hybrid-family / spectral
family behavior transfers to PDEBench CNS under one fixed local capped
contract.

The paper may not claim:

- same-protocol full-training CNS benchmark competitiveness
- that `history_len=3` is the locked paper table contract
- that GNOT or repo-local FFNO proxies belong to the same headline row family
  as the selected authored FFNO row
