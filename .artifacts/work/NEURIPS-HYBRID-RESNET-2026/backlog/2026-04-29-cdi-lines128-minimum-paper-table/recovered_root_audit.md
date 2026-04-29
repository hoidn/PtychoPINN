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
  - basis: invocation artifacts and canonical GT recon exist, but the root never produced the required four-row same-root bundle surfaces and is superseded by later audited recovery attempts.
- `runs/minimum_subset_20260429T204642Z`
  - classification: `failed_recoverable`
  - basis: TensorFlow `baseline` and `pinn` training/inference plus stitched metrics completed, but row collation failed before Torch rows launched.
- `runs/minimum_subset_20260429T213028Z`
  - classification: `failed_recoverable`
  - basis: all four required rows completed and row-local artifacts were written, but final bundle collation crashed in `scripts/studies/metrics_tables.py` on nested `numpy.float32` JSON serialization.
- `runs/minimum_subset_20260429T224103Z`
  - classification: `failed_recoverable`
  - basis: post-fix fresh-root rerun started under tmux with a tracked PID, then failed during dataset NPZ writing with `OSError: [Errno 28] No space left on device`.

## Authority Boundary

- The harness preflight note remains readiness-only authority for the prerequisite item.
- The minimum-subset execution-authority note plus
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`
  remain the launch-controlling surfaces for this item.

## Next Action Root

- No active writer remains.
- A compliant fresh rerun now requires additional free space on `/` before a new `minimum_subset_<timestamp>` root can be launched.
