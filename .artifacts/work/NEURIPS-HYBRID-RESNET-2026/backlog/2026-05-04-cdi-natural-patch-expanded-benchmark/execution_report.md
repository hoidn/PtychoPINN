# Execution Report

## Completed In This Pass

- Patched `run_natural_patch_benchmark` in
  `scripts/studies/cdi_natural_patch_benchmark.py` to call
  `write_paper_benchmark_bundle` with `require_row_provenance=True` and to
  forward executor `row_statuses`. The harness can no longer silently promote
  recovered or incomplete bundles to `paper_complete` (HIGH-2 from the
  implementation review).
- Added two regression tests in
  `tests/studies/test_cdi_natural_patch_benchmark.py` covering benchmark-mode
  completeness:
  - `test_run_natural_patch_benchmark_downgrades_when_visuals_or_provenance_missing`
    asserts `benchmark_incomplete`, non-empty `missing_fields_by_row` covering
    every required provenance field, and `paper_complete != benchmark_status`
    for incomplete row payloads.
  - `test_run_natural_patch_benchmark_downgrades_blocked_or_failed_rows`
    asserts that blocked / failed-launcher row outcomes downgrade the bundle
    and record missing fields for every required row (Medium-2 follow-up
    coverage).
- Re-collated the existing
  `runs/natural-patch-benchmark-20260505T213458Z` bundle through
  `write_paper_benchmark_bundle` with the new flags. The bundle now classifies
  honestly: `metrics.json` and `model_manifest.json` carry
  `benchmark_status="benchmark_incomplete"` with provenance gaps recorded per
  row (every row missing `randomness`, `dataset/manifest_json`,
  `splits/manifest_json`, `outputs.exit_code_proof_json`, full `environment`
  fields, `git.dirty_state_note`; torch rows additionally missing `visuals`
  and recovered-`invocation`).
- Synchronized durable surfaces with the corrected status:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_expanded_benchmark_summary.md`,
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`,
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`,
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`,
  `docs/studies/index.md`, and `docs/index.md` all describe the bundle as
  `benchmark_incomplete` with the launcher-proof and provenance-scaffolding
  gaps named explicitly.

## Completed Current-Scope Work

- Task 1: prerequisite presence gate, dataset-contract preflight, dry-run
  inspection, and prepared-input contract handling are unchanged and still
  green.
- Task 2: narrow harness fixes from the prior pass remain in place, and the
  HIGH-2 contract bug in `run_natural_patch_benchmark` is now patched with
  benchmark-mode regression coverage.
- Task 3: every row trained, ran inference, and emitted row-local metrics /
  configs / histories / recons under
  `runs/natural-patch-benchmark-20260505T213458Z`. The recovered bundle is
  republished with honest `benchmark_incomplete` classification rather than
  the prior unsupported `paper_complete` claim, since the tracked launcher
  exited `1`, torch-row fixed-sample PNGs were never backfilled, and the
  natural-patch harness does not yet emit the locked provenance scaffolding
  expected by `require_row_provenance=True`.
- Task 4: the durable summary and discoverability surfaces are re-synced to
  the corrected status and explicitly route readers to the residual risks and
  follow-up work below.

## Follow-Up Work

- Relaunch the natural-patch benchmark end-to-end on the locked dataset until
  the tracked PID exits `0`, with the harness emitting torch-row fixed-sample
  visuals directly under `visuals/` instead of relying on post-hoc recovery.
- Extend `cdi_natural_patch_benchmark.py` to construct the full provenance
  payloads required by `metrics_tables.py` under
  `require_row_provenance=True`:
  `environment.host`, `environment.torch_version`, `environment.cuda_version`,
  `environment.gpu`, `git.dirty_state_note`, dataset and splits
  `manifest_json` records with size and sha256 entries, `randomness`, and
  `outputs.exit_code_proof_json`. Once these are in place a clean rerun can
  legitimately reach `paper_complete`.
- Investigate the `train_wall_time_sec` ≈ `0.000277` / `0.000257` and
  `inference_time_sec=null` artifacts on the TF rows of the recovered bundle
  (Medium-1). This corruption originated in the recovery path; a clean rerun
  should restore credible runtime telemetry but the harness should also
  defensively round-trip these values from the row-local `history.json` /
  `runs/<row>/metrics.json` rather than the recovery scratch payload.

## Verification

- Required input gate (review-pass log):
  `verification/required_input_gate_review_20260505T233250Z.log`
- Required pytest gate (review-pass log):
  `verification/pytest_selected_review_20260505T233250Z.log`
  (`24 passed`).
- Compile gate (review-pass log):
  `verification/compileall_selected_review_20260505T233250Z.log`.
- Repo integration gate (review-pass log):
  `verification/pytest_integration_review_20260505T233250Z.log`
  (`5 passed, 4 skipped, 2200 deselected`).
- Re-collation evidence: rerunning `write_paper_benchmark_bundle` on the
  existing run root with `require_row_provenance=True` and the recovered
  `row_statuses` produced `benchmark_status="benchmark_incomplete"` with the
  per-row missing fields listed above; `metrics.json` / `model_manifest.json`
  in the run root reflect that classification.

## Residual Risks

- The tracked tmux launcher for
  `natural-patch-benchmark-20260505T213458Z` exited `1`; the bundle is now
  honestly classified `benchmark_incomplete`, but a clean launcher-proof
  rerun is still required before this lane can reach `paper_complete`.
- The recovered bundle does not backfill torch-row fixed-sample PNGs and the
  natural-patch harness does not yet emit the full provenance scaffolding
  required for `paper_complete`. These two gaps will continue to fail
  provenance validation until the follow-up work above is delivered.
- This remains single-seed evidence on `natural_patches128_fixedprobe_v1`. It
  widens expanded-object CDI evidence beyond `lines128` only as advisory
  context until a clean re-publication exists; it does not replace or
  supersede the `lines128` authority under any reading.
