# Recovered Root Audit

- Item: `2026-04-29-cdi-lines128-minimum-paper-table`
- Audit date: `2026-04-29`
- Launch authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md`
- Readiness-only prerequisite note:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`

## Row Mapping

- `baseline` -> `CDI CNN + supervised`
- `pinn` -> `CDI CNN + PINN`
- `pinn_hybrid_resnet` -> `Hybrid ResNet + PINN`
- `pinn_fno_vanilla` -> `FNO Vanilla + PINN`

## Root Classification

- `runs/minimum_subset_20260429T204000Z`
  - classification: `stale_do_not_reuse`
  - basis: invocation artifacts and canonical GT recon exist, but the root
    never produced the required four-row same-root bundle surfaces and is
    superseded by later audited attempts.
- `runs/minimum_subset_20260429T204642Z`
  - classification: `failed_recoverable`
  - basis: TensorFlow `baseline` and `pinn` completed, but row collation failed
    before the Torch rows launched.
- `runs/minimum_subset_20260429T213028Z`
  - classification: `decision_support_only_not_paper_grade`
  - basis: the implementation review was correct that this root lacked the
    required TensorFlow row-local provenance and the required visual bundle
    (`amp_phase_error_*`, `frc_curves.png`), so same-root recovery here is not
    honest paper-grade evidence.
- `runs/minimum_subset_20260429T224103Z`
  - classification: `failed_recoverable`
  - basis: an earlier fresh-root rerun failed during dataset NPZ writing with
    `OSError: [Errno 28] No space left on device`.
- `runs/minimum_subset_20260429T235811Z`
  - classification: `paper_complete_fresh_rerun`
  - basis: all four required rows completed under one fresh root, the tracked
    tmux shell PID exited `0`, the merged bundle reports `paper_complete`, all
    rows report `paper_grade`, and the root now contains the required row-local
    provenance plus the required error-panel and FRC visual surfaces.

## Authority Boundary

- The harness preflight note remains readiness-only authority for the prerequisite item.
- The minimum-subset execution-authority note plus
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`
  remain the launch-controlling surfaces for this item.

## Chosen Execution Path

- chosen path: `fresh_rerun_required`
- chosen root:
  `runs/minimum_subset_20260429T235811Z`
- completion basis:
  - same-root recovery in `minimum_subset_20260429T213028Z` was explicitly
    rejected after the review-validated provenance/visual gaps were rechecked
  - the fresh rerun launched under tmux into a brand-new root with tracked
    shell PID `1211720`
  - the tracked PID exited `0`
  - the fresh root now reports:
    - `benchmark_status=paper_complete`
    - `claim_boundary=minimum_draftable_cdi_subset`
    - empty `missing_fields_by_row` for all four required rows
    - `row_status=paper_grade` for every required row

## Current Root Status

- No active writer remains.
- The authoritative minimum-table bundle root is now
  `runs/minimum_subset_20260429T235811Z`.
- The former missing-resource blocker is resolved for this item; the successful
  fresh rerun demonstrates that enough free space was available for the full
  write-side path during this pass.
