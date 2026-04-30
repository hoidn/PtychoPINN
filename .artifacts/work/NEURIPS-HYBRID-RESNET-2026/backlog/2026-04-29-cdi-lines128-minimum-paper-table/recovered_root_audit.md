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
  - classification: `paper_complete_recovered_in_place`
  - basis: the root still originates from the successful four-row fresh rerun,
    and the latest review-fix pass completed the approved same-root recovery
    path under the stricter existence-based provenance gate: the merged bundle
    was regenerated from the existing row-local artifacts, emitted validation
    loss still propagates correctly, wrapper/root artifact validation now checks
    the required files rather than trusting strings, Torch row-local
    `config.json` artifacts remain recovered honestly from invocation metadata,
    and the merged root again reports `paper_complete` with `paper_grade` rows,
    empty `missing_fields_by_row`, and empty `missing_bundle_artifacts`.

## Authority Boundary

- The harness preflight note remains readiness-only authority for the prerequisite item.
- The minimum-subset execution-authority note plus
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`
  remain the launch-controlling surfaces for this item.

## Chosen Execution Path

- chosen path: `same_root_recovery`
- chosen root: `runs/minimum_subset_20260429T235811Z`
- completion basis: the earlier `minimum_subset_20260429T213028Z` same-root
  candidate remains rejected because its provenance and visual gaps were real.
  The current authoritative root already contained all four required completed
  rows, so this pass reused that root without retraining and reran only the
  bundle/collation path with `--reuse-existing-recons` after tightening
  provenance/path validation. The regenerated bundle now reports
  `benchmark_status=paper_complete`,
  `claim_boundary=minimum_draftable_cdi_subset`,
  empty `missing_fields_by_row`, empty `missing_bundle_artifacts`, and
  `row_status=paper_grade` for every required row.

## Current Root Status

- No active writer remains.
- The authoritative minimum-table bundle root is now
  `runs/minimum_subset_20260429T235811Z`.
- The former missing-resource blocker remains resolved for this item; this pass
  performed in-place recovery only and did not need another expensive row rerun.
