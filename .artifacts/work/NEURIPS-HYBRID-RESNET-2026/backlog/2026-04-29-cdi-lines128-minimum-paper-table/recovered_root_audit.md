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
    never produced the required four-row bundle surfaces and is superseded by
    later audited attempts.
- `runs/minimum_subset_20260429T204642Z`
  - classification: `failed_recoverable`
  - basis: TensorFlow `baseline` and `pinn` completed, but row collation failed
    before the Torch rows launched.
- `runs/minimum_subset_20260429T213028Z`
  - classification: `decision_support_only_not_paper_grade`
  - basis: the root lacked the required TensorFlow row-local provenance and the
    required visual bundle (`amp_phase_error_*`, `frc_curves.png`), so same-root
    recovery here is not honest paper-grade evidence.
- `runs/minimum_subset_20260429T224103Z`
  - classification: `failed_recoverable`
  - basis: an earlier fresh-root rerun failed during dataset NPZ writing with
    `OSError: [Errno 28] No space left on device`.
- `runs/minimum_subset_20260429T235811Z`
  - classification: `rejected_synthetic_proof_root`
  - basis: the implementation review was correct that the
    `--reuse-existing-recons` path fabricated row-local `stdout.log`,
    `stderr.log`, and `exit_code_proof.json` artifacts. After hardening the
    provenance contract, this root no longer qualifies as honest paper-grade
    evidence and is retained only as a rejected historical recovery attempt.
- `runs/minimum_subset_20260430T031228Z`
  - classification: `failed_fresh_rerun`
  - basis: the first honest fresh rerun failed during TensorFlow row
    provenance finalization with `NameError: name 'datetime' is not defined` in
    `ptycho/workflows/grid_lines_workflow.py`.
- `runs/minimum_subset_20260430T035104Z`
  - classification: `decision_support_only_shared_tf_logs`
  - basis: under the stricter validator added in this pass, the TensorFlow
    `baseline` and `pinn` rows reuse duplicated shared-workflow `stdout.log`
    payloads, so this root no longer satisfies the paper-grade TF provenance
    contract even though the historical bundle manifest recorded
    `paper_complete`.
- `runs/minimum_subset_20260430T051928Z`
  - classification: `stopped_followup_rerun`
  - basis: a fresh rerun was launched after the TF row-log fix, but it was
    stopped during training because the full benchmark execution exceeds the
    scope of this review-fix pass. Treat the root and its log as incomplete
    follow-up artifacts only.

## Authority Boundary

- The harness preflight note remains readiness-only authority for the
  prerequisite item.
- The minimum-subset execution-authority note plus
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`
  remain the launch-controlling surfaces for this item.

## Chosen Execution Path

- chosen path: `fresh_rerun_required_after_tf_row_log_fix`
- chosen root: none in this pass
- completion basis: the earlier
  `runs/minimum_subset_20260429T235811Z` same-root recovery claim was rejected
  because its provenance artifacts were synthetic, and the later historical
  `runs/minimum_subset_20260430T035104Z` rerun is now downgraded because its
  TensorFlow required rows reuse duplicated shared-workflow logs. A fresh rerun
  remains required before this item can be restored to `paper_complete`.

## Current Root Status

- No authoritative paper-grade minimum-table bundle root is available in this
  pass.
- The interrupted follow-up rerun root
  `runs/minimum_subset_20260430T051928Z` and the earlier historical root
  `runs/minimum_subset_20260430T035104Z` must not be cited as current
  paper-grade evidence.
