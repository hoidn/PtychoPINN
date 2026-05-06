# Execution Report

## Completed In This Pass

This pass addresses the prior implementation review's three HIGH findings by
making the recollate path honest about provenance and bundle authority.

- **HIGH-2 (invocation envelope rewrite):** removed
  `_promote_recovered_row_invocation` from
  `scripts/studies/cdi_natural_patch_benchmark.py`. The recollate path no
  longer rewrites per-row `invocation.json` envelopes from `failed/1` to
  `completed/0`. A new read-only helper
  `_read_row_invocation_metadata` returns the per-row
  `(status, exit_code, git_commit)` exactly as the original launch recorded
  them. The four torch rows of the existing run root were reverted from the
  previous promotion (`status="failed"`, `exit_code=1` restored, all
  `recovered_*_from_recollate_promotion` keys removed).
- **HIGH-3 (corrupted git provenance):** the recollate path now stamps the
  per-row provenance with the original execution commit retrieved from the
  row's `invocation.json` `extra.git_commit`. `_attach_natural_patch_row_provenance`
  takes a new optional `git_commit_override` argument; when provided it
  takes precedence over any pre-existing payload `git.commit`. The old
  recollate corrupted the metrics.json `git.commit` to the recollate-time
  HEAD (`1f6d506e`); the rebuilt bundle restores the original execution
  commit `5c4deddfd9b81431c063276720e7e4d3bf911ff7` for every row.
- **HIGH-1 (overstated bundle authority):** the recollate path now publishes
  every required row as `recovered_non_authoritative` and the bundle resolves
  to `benchmark_status="benchmark_incomplete"`. The recollate path also sets
  the row payload's `row_status="recovered_non_authoritative"` so the bundle
  writer's paper-grade gate downgrades the bundle on its own. The bundle's
  `row_statuses[*]` carry `row_invocation_status` /
  `row_invocation_exit_code` so the four failed torch invocations are
  visible in the published metrics; the recollate launcher exits `0` only
  to record honest republication, not to claim Task 3 success.
- Re-published the existing
  `runs/natural-patch-benchmark-20260505T213458Z` run root via
  `--mode recollate`. The recollate launcher's tracked PID exited `0` and
  the resulting bundle reports `benchmark_status="benchmark_incomplete"`
  with every required row in `recovered_non_authoritative` state and the
  original execution commit preserved per row. Verification logs are
  archived under `verification/recollate-honest-<UTC>.{log,exit_code,pid}`.
- Replaced the prior promotion-path tests with honest equivalents in
  `tests/studies/test_cdi_natural_patch_benchmark.py`:
  - `test_read_row_invocation_metadata_returns_original_failed_envelope`
    asserts the read helper returns the failed envelope's status/exit-code
    and `extra.git_commit` without rewriting the file.
  - `test_read_row_invocation_metadata_returns_none_when_missing` asserts
    the helper safely returns `None` for missing invocations.
  - `test_recollate_mode_publishes_recovered_non_authoritative_bundle`
    asserts the recollate path: (a) publishes
    `benchmark_status="benchmark_incomplete"` with every row in
    `recovered_non_authoritative` state; (b) preserves the original
    execution commit per row instead of stamping the recollation commit;
    (c) leaves a failed/1 invocation envelope intact, with no
    `recovered_exit_code_from_recollate_promotion` audit key written; and
    (d) downgrades the bundle's `missing_fields_by_row` to record
    `row_status` as missing for every row.
- Synchronized durable surfaces with the corrected status:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_expanded_benchmark_summary.md`
    now reads `benchmark_incomplete_recovered_non_authoritative`, surfaces
    each torch row's original `failed/1` invocation, preserves the original
    execution commit, and explicitly states that the bundle is **not**
    paper-grade until a clean tmux launcher exits `0` end-to-end.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` row
    is downgraded to `benchmark_incomplete (recovered, non-authoritative)`.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` headline,
    natural-patch table, and completed-output map all carry the
    non-authoritative state.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
    natural-patch six rows now carry
    `evidence_status="benchmark_incomplete_recovered_non_authoritative"`.

## Completed Current-Scope Work

- Task 1: prerequisite presence gate, dataset-contract preflight, dry-run
  inspection, and prepared-input contract handling are unchanged and still
  green.
- Task 2: the natural-patch harness now refuses to upgrade a recovered
  bundle past `benchmark_incomplete`. Live-run code paths and provenance
  scaffolding are unchanged; only the recollate path was tightened. The
  `_PROVENANCE_FIELDS_TO_DROP` list keeps `git` so the original commit
  from the existing metrics flows through, and `git_commit_override`
  provides the explicit per-row authoritative source from the
  invocation.json the row recorded.
- Task 3: the existing run root is now re-published as a
  `benchmark_incomplete` recovered/non-authoritative bundle with honest
  per-row state. The approved Task 3 completion gate (clean tmux
  `--mode benchmark` PID exit `0`) is **not** satisfied; this is recorded
  honestly rather than masked.
- Task 4: the durable summary, this execution report, evidence index,
  evidence matrix, and model variant index all describe the same
  recovered/non-authoritative state. The discoverability surfaces no
  longer claim `paper_grade` for this benchmark.

## Follow-Up Work

- Authoritative path: relaunch the full six-row natural-patch benchmark on
  the locked dataset under one contiguous tmux launcher
  (`--mode benchmark`). The tracked PID must exit `0` end-to-end before
  this benchmark can be promoted to `paper_complete` and the
  discoverability surfaces re-promoted to `paper_grade`. The provenance
  scaffolding emitted by `_attach_natural_patch_row_provenance` already
  satisfies `require_row_provenance=True` for live runs, so no further
  harness work is required to support a clean rerun.
- Recollation remains available as a republication tool for
  bundle-collation-only failures, but the path now refuses to upgrade a
  recovered bundle past `benchmark_incomplete` so a future bundle-collation
  crash cannot be silently rewritten into paper-grade evidence.

## Verification

- Required input presence gate (this pass log):
  `verification/required_input_gate_revise_20260506T005045Z.log`.
- Required pytest gate (this pass log):
  `verification/pytest_selected_revise_20260506T005045Z.log`
  (`30 passed`, including the two read-helper tests and the rewritten
  recollate end-to-end test).
- Compile gate (this pass log):
  `verification/compileall_revise_20260506T005045Z.log`.
- Repo integration marker (this pass log):
  `verification/pytest_integration_revise_20260506T005045Z.log`
  (`5 passed, 4 skipped`).
- Honest recollate launcher proof:
  `verification/recollate-honest-20260506T004658Z.exit_code` reports `0`,
  `verification/recollate-honest-20260506T004658Z.log` records the harness
  JSON result, and the run root's `metrics.json` /
  `model_manifest.json` report
  `benchmark_status="benchmark_incomplete"` with every row in
  `recovered_non_authoritative` state and the original execution commit
  `5c4deddfd9b81431c063276720e7e4d3bf911ff7` preserved per row.

## Residual Risks

- The approved Task 3 completion gate (clean tmux `--mode benchmark` PID
  exit `0` end-to-end) is **not** satisfied. The bundle is honestly
  published as `benchmark_incomplete` and the discoverability surfaces no
  longer claim paper-grade authority. A clean from-scratch rerun is the
  only path to a paper-citable natural-patch authority.
- The four torch rows surface `row_invocation_status="failed"` /
  `row_invocation_exit_code=1` in the bundle. The on-disk training
  artifacts (`config.json`, `metrics.json`, `history.json`, model
  checkpoints, recons) survive from the original launch, but the row
  processes themselves reported failure. The recovered metrics are
  diagnostic context only and must not be cited as authoritative
  expanded-object CDI evidence.
- TF-row `train_wall_time_sec` (~0.0003) and `inference_time_sec=null`
  remain advisory-only telemetry inherited from the original recovery
  path. They do not block paper completion under the current validator
  but should be regenerated by the clean rerun follow-up.
- This remains single-seed expanded-object CDI evidence on
  `natural_patches128_fixedprobe_v1` and does not replace the `lines128`
  paper-table authority under any reading.
