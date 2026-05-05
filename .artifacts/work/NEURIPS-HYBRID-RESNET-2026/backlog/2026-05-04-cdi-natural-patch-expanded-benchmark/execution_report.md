# Execution Report

## Completed In This Pass

- Fixed the row-status plumbing bug flagged as HIGH-3 in the implementation
  review. `run_natural_patch_benchmark` in
  `scripts/studies/cdi_natural_patch_benchmark.py` now translates execution
  statuses into the bundle harness-status form via
  `_bundle_row_statuses_from_execution`: `completed` rows map to
  `supported_for_harness` (and the original execution status is preserved
  under `execution_status`), while `blocked` / `not_protocol_compatible`
  outcomes are passed through verbatim so they continue to downgrade the
  bundle. The execution-time status remains intact in the result envelope and
  in `paper_benchmark_manifest.json` for traceability. Without this fix the
  bundle writer would permanently downgrade fully provenanced runs to
  `benchmark_incomplete`, blocking any path to `paper_complete` even after
  HIGH-1 / HIGH-2 follow-ups.
- Updated
  `tests/studies/test_cdi_natural_patch_benchmark.py::test_run_natural_patch_benchmark_downgrades_when_visuals_or_provenance_missing`
  to codify the new status semantics: bundle row_statuses for completed
  executions now report `supported_for_harness` (with `execution_status:
  "completed"`), the manifest preserves the raw `completed` execution status,
  and the bundle still downgrades when provenance is incomplete.
- Added a new positive-path regression test
  `test_run_natural_patch_benchmark_reaches_paper_complete_when_row_provenance_satisfied`
  that wires a fully provenanced row payload (with on-disk dataset / split
  manifests, exit-code proof, invocation/config/history/metrics/recon side
  files, and fixed-sample visuals) through `run_natural_patch_benchmark` and
  asserts `benchmark_status == "paper_complete"` with no missing fields. This
  covers the reviewer's "After the blocking fixes above land, add one
  positive-path test" follow-up and proves the HIGH-3 fix unblocks the
  paper_complete path end-to-end.
- Patched `run_natural_patch_benchmark` in
  `scripts/studies/cdi_natural_patch_benchmark.py` to call
  `write_paper_benchmark_bundle` with `require_row_provenance=True` and to
  forward executor `row_statuses`. The harness can no longer silently promote
  recovered or incomplete bundles to `paper_complete` (prior pass HIGH-2 from
  the original implementation review).
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
- Task 2: prior-pass harness fixes remain in place; this pass adds the
  HIGH-3 row-status translation in `run_natural_patch_benchmark` plus a
  positive-path bundle regression test, removing the permanent downgrade
  bug that previously prevented the bundle writer from ever promoting a
  fully provenanced run to `paper_complete`.
- Task 3: every row trained, ran inference, and emitted row-local metrics /
  configs / histories / recons under
  `runs/natural-patch-benchmark-20260505T213458Z`. The recovered bundle is
  republished with honest `benchmark_incomplete` classification, since the
  tracked launcher exited `1`, torch-row fixed-sample PNGs were never
  backfilled, and the natural-patch harness does not yet emit the locked
  provenance scaffolding expected by `require_row_provenance=True`. A clean
  launcher-proof rerun (HIGH-1) and live emission of the side-file
  scaffolding (HIGH-2) remain follow-up work.
- Task 4: the durable summary and discoverability surfaces are re-synced to
  the corrected status and explicitly route readers to the residual risks and
  follow-up work below.

## Follow-Up Work

- HIGH-1 (clean live launcher rerun): relaunch the natural-patch benchmark
  end-to-end on the locked dataset until the tracked PID exits `0`, with the
  harness emitting torch-row fixed-sample visuals directly under `visuals/`
  instead of relying on post-hoc recovery. This requires a multi-hour GPU
  training run and is deferred to follow-up work; the existing recovered
  bundle remains classified `benchmark_incomplete` until that rerun lands.
- HIGH-2 (full provenance scaffolding in the harness): extend
  `cdi_natural_patch_benchmark.py` (and where unavoidable, the underlying
  `_write_tf_row_provenance` /
  `grid_lines_torch_runner._write_torch_row_provenance` pathways) to emit
  the locked side artifacts that `metrics_tables.py` validates under
  `require_row_provenance=True`:
  - `dataset.manifest_json` pointing to a JSON record with `train_npz`,
    `test_npz`, `size_bytes`, `sha256`, and `source` entries.
  - `splits.manifest_json` matching `nimgs_train`, `nimgs_test`, `seed`,
    `gridsize`, and `set_phi` against the row payload.
  - `randomness.requested_seed` matching the splits seed and the
    invocation/config seeds (currently emitted only as a partial
    `seed_policy` payload by `_write_tf_row_provenance`).
  - `outputs.exit_code_proof_json` with `model_id`, `exit_code=0`,
    `proof_source`, matching `invocation_json` / `stdout_log` / `stderr_log`
    references, and `invocation_status="completed"`.
  The new positive-path regression test
  (`...reaches_paper_complete_when_row_provenance_satisfied`) demonstrates
  the bundle writer reaches `paper_complete` once these artifacts exist; the
  harness itself still needs to produce them in a live run.
- Investigate the `train_wall_time_sec` ≈ `0.000277` / `0.000257` and
  `inference_time_sec=null` artifacts on the TF rows of the recovered bundle
  (Medium-1). This corruption originated in the recovery path; a clean rerun
  should restore credible runtime telemetry but the harness should also
  defensively round-trip these values from the row-local `history.json` /
  `runs/<row>/metrics.json` rather than the recovery scratch payload.

## Verification

- Required input gate (paper-ready-fix log):
  `verification/required_input_gate_paperready_20260505T235659Z.log`
- Required pytest gate (paper-ready-fix log):
  `verification/pytest_selected_paperready_20260505T235659Z.log`
  (`26 passed`, including the new positive-path test).
- Compile gate (paper-ready-fix log):
  `verification/compileall_selected_paperready_20260505T235659Z.log`.
- Repo integration gate (paper-ready-fix log):
  `verification/pytest_integration_paperready_20260505T235659Z.log`
  (`5 passed, 4 skipped, 2202 deselected`).
- Prior pass logs (review pass) retained alongside:
  `verification/required_input_gate_review_20260505T233250Z.log`,
  `verification/pytest_selected_review_20260505T233250Z.log` (`24 passed`),
  `verification/compileall_selected_review_20260505T233250Z.log`,
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
