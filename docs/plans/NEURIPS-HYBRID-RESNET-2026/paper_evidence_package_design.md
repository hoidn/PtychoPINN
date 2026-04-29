# NeurIPS Hybrid ResNet Paper Evidence Package Design

## Context And Authority

- Status: draft design
- Date: 2026-04-29
- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Consumed brief: define what evidence is needed before drafting a credible
  Hybrid ResNet paper, starting from a CDI `lines128` benchmark and a PDEBench
  CNS benchmark, with numeric metrics, visual comparisons, and explicit
  paper-grade provenance.
- Governing docs:
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `docs/studies/index.md`
  - `docs/model_baselines.md`

## Problem And Scope

The project needs a small, paper-grade evidence package before result claims can
be drafted. Current evidence is strong enough to shape the paper, but not yet
strong enough to support final claims:

- CDI `lines128` has a strong historical Hybrid ResNet row, but it is
  decision-support only because complete paper-grade provenance was not
  recovered.
- PDEBench CNS has many useful capped comparisons, but most are explicitly
  decision-support rather than full-training benchmark evidence.
- The paper needs both numeric tables and visual comparisons under fixed,
  externally auditable contracts.

In scope:

- define the required CDI and CNS result tables
- define required figures and source artifacts
- define paper-grade provenance gates
- define claim boundaries for capped, decision-support, and paper-grade rows
- define a backlog decomposition that can be executed by the active drain
  workflow or by manual backlog items

Out of scope:

- manuscript prose
- creating `/home/ollie/Documents/neurips/` paper-facing artifacts immediately
- selecting final winning models before the locked tables exist
- broad architecture sweeps beyond rows needed for a credible paper table
- treating historical incomplete artifacts as paper-grade evidence

## Decision Summary

Use a two-pillar evidence package:

1. CDI `lines128` reconstruction benchmark.
2. PDEBench `2d_cfd_cns` forward-prediction benchmark.

The CDI pillar is the paper anchor. The CNS pillar is the generalization /
physics-modeling evidence. Each pillar must produce:

- one locked numeric table
- one locked visual comparison bundle
- one machine-readable manifest
- one durable summary with claim boundaries

Drafting may begin before every result is complete, but result claims must stay
as placeholders until the corresponding table and figure bundle are locked.

The package should prefer the best available Hybrid-family row for each pillar
when writing claim language, but row selection must not narrow the benchmark
tables themselves:

- CDI: the complete `lines128` benchmark table must include the rows required by
  `lines128_paper_benchmark_design.md`; the paper may separately identify the
  best Hybrid-family row after that table is locked.
- CNS: the best bounded Hybrid/Hybrid-spectral row under the selected CNS
  contract, with explicit labeling if it is capped decision-support evidence.

## Evidence Pillar A: CDI `lines128`

The detailed benchmark design is
`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`.
This package design treats that document as the CDI table authority.

Complete CDI benchmark table rows:

- `hybrid_resnet`
- `spectral_resnet_bottleneck_net`
- FNO comparator, selected before launch as either `fno` or `fno_vanilla`
- FFNO row, after FFNO satisfies the CDI/grid-lines generator contract
- CNN/PINN row as the local non-spectral neural baseline required by this
  package-level paper-evidence design
- optional classical CDI row, preferably HIO/ER/PyNX, if it can be made
  protocol-compatible without changing the task contract

Minimum draftable CDI claim rows:

- `hybrid_resnet`
- CNN/PINN row
- selected FNO comparator

This minimum subset can unblock manuscript table shells and bounded preliminary
claim drafting, but it is not the complete `lines128` benchmark. Do not mark the
CDI benchmark table complete until `spectral_resnet_bottleneck_net` and FFNO
are present, blocked with explicit row-level reasons, or removed by a checked-in
amendment to the detailed `lines128` design.

Required CDI metrics:

- amplitude MAE, MSE, PSNR, SSIM, MS-SSIM, FRC50
- phase MAE, MSE, PSNR, SSIM, MS-SSIM, FRC50
- final train loss
- validation loss when the training contract emits a real validation series
- parameter count
- runtime and hardware
- row status: `paper_grade`, `decision_support`, `blocked`, or
  `not_protocol_compatible`

Required CDI visuals:

- shared test sample IDs across every model row
- ground-truth amplitude and phase
- reconstructed amplitude and phase for every row
- amplitude and phase absolute-error panels for every row
- shared color scales per quantity
- FRC curves
- source arrays sufficient to regenerate figures

CDI paper-grade gate:

- every row in the headline table must be regenerated or recovered with complete
  invocation/config/git/environment/dataset/split/metric/visual provenance
- historical rows can appear only as sanity context unless they satisfy that bar
- no FFNO row may count unless it uses the same CDI dataset, split, stitching,
  metric, and generator-output contract as the Hybrid row

## Evidence Pillar B: PDEBench CNS

Selected contract as of `2026-04-29`:

- authoritative decision:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- durable row-lock summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- machine-readable locked-row manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`
- Selected contract: `bounded_capped_decision_support`
- Selected history lane: `history_len=2`, `40` epochs, `512 / 64 / 64` trajectories, `max_windows_per_trajectory=8`, emitted windows `4096 / 512 / 512`
- Selected normalization contract: train-only per-field normalization fit on the `512` training trajectories, reused across all history slots and target channels, with evaluation reported in denormalized target space.
- Selected training recipe contract: keep the CNS task-local `mse` override relative to the design's generic `mae` baseline; use `Adam` with learning rate `2e-4`; use `ReduceLROnPlateau` with factor `0.5`, patience `2`, threshold `0.0`, and `min_lr=1e-5`; keep batch size `4`; keep the metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`.
- locked headline rows:
  `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`,
  `author_ffno_cns_base`
- audited continuity row:
  `hybrid_resnet_cns`
- adjacent-only context:
  `history_len=3` capped pilots, GNOT, and repo-local FFNO proxy rows

The package no longer leaves the CNS contract open-ended. Future readers should
not recover normalization or training recipe rules from older summaries; the
decision document above is now the authority surface for both.

The row-lock pass also freezes the accepted run roots for downstream
table/figure assembly. That lock is intentionally narrower than the package's
paper-grade provenance target: the accepted capped roots expose invocation,
dataset/split, normalization, profile, metrics, and source-array assets, but
they do not emit standalone repo-git, run-log, or exit-code artifacts. Treat
the locked manifest as bounded `capped_decision_support` authority only, not as
proof that the CNS pillar has reached `paper_grade` provenance completeness.

Required CNS table rows:

- best Hybrid/Hybrid-spectral row under the selected CNS contract
- local FNO row
- U-Net or CNN-style local baseline
- authored FFNO as a cutoff-gated extension row:
  - if authored FFNO is available by the CNS contract-decision cutoff and can
    obey the same local CNS contract, include it in the locked table
  - if it is not available or cannot obey the contract, record an explicit
    `blocked` or `not_protocol_compatible` row and state that CNS claims compare
    against local FNO/U-Net baselines but not authored FFNO
- optional GNOT row if its environment/protocol is credible enough to compare
  without hiding caveats

Required CNS metrics:

- `err_nRMSE`
- `err_RMSE`
- `relative_l2`
- `fRMSE_low`
- `fRMSE_mid`
- `fRMSE_high`
- per-channel versions when available
- parameter count
- runtime and hardware
- training split/cap/full-training status
- row status: `full_training`, `capped_decision_support`, `blocked`, or
  `not_protocol_compatible`

Required CNS visuals:

- one fixed set of test sample IDs across rows
- prediction and ground truth panels for the four CNS fields:
  `density`, `Vx`, `Vy`, `pressure`
- error panels for the same fields
- shared color scales per field and per error quantity
- source arrays sufficient to regenerate figures

CNS claim gate:

- If rows remain capped, the paper may use CNS as bounded evidence only. Claims
  must say the result is capped decision-support and cannot imply full benchmark
  competitiveness.
- If CNS is used for benchmark-performance claims, the table must contain
  same-contract full-training rows for the selected model family and required
  baselines, or the summary must explicitly justify a narrower claim.
- Current capped studies may select which rows deserve full-training budget, but
  they do not by themselves satisfy the full-training benchmark gate.

## Shared Provenance Contract

Every paper-grade table row must have:

- unique run root
- `invocation.sh` or equivalent exact command
- structured invocation/config JSON
- git SHA plus dirty-state note
- Python, PyTorch, CUDA, GPU, and host provenance
- dataset identity manifest with source URL or dataset ID, size, checksum; if a
  checksum is genuinely unavailable, include a reviewed exception with
  size/mtime/source manifest and rationale
- split manifest
- model profile and parameter count
- metric schema and units
- source prediction/reconstruction arrays
- generated figure paths
- run logs and exit-code proof for long runs

The merged package must have:

- `paper_evidence_manifest.json`
- per-pillar table JSON/CSV/TeX
- per-pillar metric schema JSON
- figure manifests with source-array paths
- durable summaries under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`
- later, paper-facing links under `/home/ollie/Documents/neurips/index.md`
  during the roadmap evidence-bundle phase

## Drafting Gate

The paper can be drafted in phases.

Safe to draft now:

- introduction
- method and architecture description
- related work
- experiment-protocol shells
- table and figure placeholders
- limitations phrased around current evidence gaps

Do not draft as final claims yet:

- CDI superiority claims until the `lines128` table is paper-grade
- FFNO CDI conclusions until the FFNO generator row lands
- CNS benchmark-performance claims while rows are capped
- multi-seed robustness claims without pre-declared seed aggregation
- runtime/efficiency claims without parameter count and hardware-normalized
  runtime fields

Minimum result-claim gate:

- CDI table has at least Hybrid-family, CNN/PINN, and FNO rows under one
  paper-grade contract, while clearly labeling the result as the minimum
  draftable CDI subset rather than the complete `lines128` benchmark
- CNS table has at least Hybrid-family, FNO, and U-Net/CNN-style rows under one
  explicitly labeled contract
- each pillar has at least one visual comparison bundle
- every claimed row has complete provenance
- the evidence inventory states which rows are paper-grade and which are
  decision-support only

## Backlog Decomposition

This design should be converted into small backlog items rather than one broad
"write paper evidence" item.

Recommended backlog items:

1. CDI `lines128` paper-grade anchor
   - regenerate or produce the `hybrid_resnet` row under the locked CDI contract
   - output complete provenance, metrics, and visuals

2. CDI `lines128` baseline table
   - run CNN/PINN and FNO under the same CDI contract
   - emit table-ready metrics and visuals

3. CDI spectral row
   - run `spectral_resnet_bottleneck_net` under the locked `lines128` contract
   - keep it separate from "best Hybrid-family" claim selection until the table
     is locked

4. CDI FFNO generator row
   - align with or refine
     `docs/backlog/active/2026-04-27-cdi-ffno-generator-lines-best-config.md`
   - prove FFNO satisfies the same generator/output/stitching contract

5. CDI classical baseline
   - attempt HIO/ER/PyNX or record protocol incompatibility explicitly

6. CDI table and figure bundle
   - merge all locked CDI rows into paper-ready JSON/CSV/TeX and figures

7. CNS table contract decision
   - decide whether the paper will spend compute on full-training CNS rows or
     publish bounded capped evidence only
   - set the authored-FFNO cutoff and claim impact before row locking

8. CNS required rows
   - run or lock Hybrid-family, FNO, and U-Net/CNN-style rows under the selected
     CNS contract
   - include authored FFNO if available by the cutoff, otherwise record a
     row-level blocker and limit claims accordingly

9. CNS table and figure bundle
   - merge all locked CNS rows into paper-ready JSON/CSV/TeX and figures

10. Paper evidence index
   - create `/home/ollie/Documents/neurips/index.md` during the evidence-bundle
     phase and link all source artifacts, summaries, tables, figures, and
     claim boundaries

## Invariants And Failure Modes

Invariants:

- no table may mix rows with incompatible dataset/split/metric contracts unless
  that incompatibility is a visible table column and claim boundary
- no row may be promoted from decision-support to paper-grade by prose alone
- FFNO must adapt to the selected CDI/CNS contracts, not force silent contract
  drift
- visual figures must be regenerable from saved arrays
- every claimed metric must have units and a producing script or manifest

Expected failure modes:

- FFNO cannot satisfy the CDI generator contract
- CNS full-training compute is too expensive before deadline
- a baseline row fails under the fixed contract
- historical CDI configuration cannot be reconstructed confidently enough
- merged figure scripts accidentally compare different sample IDs or scales
- runtime/parameter fields are missing for some rows

Responses:

- block row-specific claims rather than weakening the contract
- keep partial rows as decision-support if provenance is incomplete
- use capped CNS results only for bounded claims if full-training rows do not
  land
- create explicit protocol-incompatibility notes for classical baselines rather
  than forcing misleading comparisons

## Verification Strategy

Design-level checks:

- this design is linked from `docs/index.md`
- the existing `lines128_paper_benchmark_design.md` remains the detailed CDI
  authority
- future backlog items reference this package design or the detailed pillar
  design

Execution-level checks for each pillar:

- parse/check every invocation and manifest JSON
- validate table schemas
- verify all declared source arrays and figure paths exist
- verify rows share the declared contract
- run targeted tests for changed runners, wrappers, and collators
- run final artifact validation before marking a table `paper_complete`

Acceptance criteria for paper-result drafting:

- CDI and CNS durable summaries exist
- table JSON/CSV/TeX exists for both pillars, even if CNS is explicitly bounded
- visual comparison bundles exist for both pillars
- provenance manifests exist for all claimed rows
- `/home/ollie/Documents/neurips/index.md` links final paper-facing artifacts
  when the roadmap reaches evidence-bundle assembly

## Handoff

This design is the package-level source for paper-evidence backlog drafting.
The next step is to create or reconcile backlog items against the decomposition
above, reusing the existing CDI FFNO and `lines128` paper benchmark designs
where possible instead of creating duplicate scopes.

Open decisions:

- whether CDI uses the reconstructed legacy-best `40`-epoch contract or the
  current anchor-regeneration contract as the final paper contract
- whether the FNO CDI row should be `fno` or `fno_vanilla`
- whether CNS receives full-training budget or remains a bounded capped table
- whether classical CDI can be made protocol-compatible in time
