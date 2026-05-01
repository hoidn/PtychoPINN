## Completed In This Pass

- added review-fix regression coverage for the remaining validator gaps:
  fresh-row contract drift must fail closed against the frozen study matrix,
  malformed `metrics_table.tex` must be rejected, stale failed/incomplete
  fresh-row leftovers must still be scrubbed and relaunched, reused-anchor
  drift must abort before fresh launches, and failed validation must raise
  instead of returning a false-but-nonfatal report
- repaired `scripts/studies/runbooks/run_cdi_hybrid_spectral_ffno_parameter_space.py`
  so fresh rows carry an explicit frozen contract projection in the preflight
  matrix, closeout validates both `invocation.parsed_args` and
  `config.torch_runner_config` against that projection, copied reused anchors
  still revalidate fail-closed before fresh launches, and `metrics_table.tex`
  must contain the expected frozen roster rather than just any non-empty text
- refreshed the archived machine-readable closeout artifacts so the checked-in
  study root now carries the stricter `preflight/study_matrix.json` contract
  projection and the updated `analysis/bundle_validation.json` fields that the
  durable summary points at

## Completed Current-Scope Work

- review finding 1 resolved: the deterministic closeout validator now fails
  closed unless each fresh row has metrics/history/recon artifacts, completed
  zero-exit proof, and invocation/config values that match the frozen matrix
  contract for that row
- review finding 2 resolved: merged-output validation now rejects malformed
  `metrics_table.tex` and the table roster is checked against the frozen row
  set in the same way as the JSON/CSV outputs
- review finding 3 resolved: the archived machine-readable validation artifact
  named by the durable summary has been refreshed and now exposes
  `fresh_row_completion_failures`, `missing_merged_outputs`, and
  `merged_output_failures` directly in `analysis/bundle_validation.json`

## Follow-Up Work

- expose the stricter bundle validator as a standalone CLI if later backlog
  work needs to audit archived CDI bridge-study roots without importing the
  runbook module

## Residual Risks

- this item remains CDI-only decision-support evidence; no fresh bridge row is
  promoted into paper-facing claim territory
- the pre-existing warning set in the closeout pytest runs remains unchanged:
  `tight_layout`, degenerate-case SSIM/MS-SSIM, FRC divide warnings, and the
  known TensorFlow Addons compatibility warnings in `pytest -m integration`
- the current repair validates the existing archived study root with the
  stricter checker, but it does not rerun the full long CDI bridge study; any
  future relaunch will pick up the repaired fail-closed behavior

## Verification

- focused harness selector:
  `pytest -q tests/studies/test_cdi_hybrid_spectral_ffno_parameter_space.py`
  -> `15 passed in 3.44s`
- required backlog gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `191 passed, 49 warnings in 304.19s (0:05:04)`
- compile check:
  `python -m compileall -q ptycho_torch scripts/studies` -> exit `0`
- integration marker:
  `pytest -v -m integration`
  -> `5 passed, 4 skipped, 1811 deselected, 2 warnings in 302.88s (0:05:02)`
- repaired bundle validation on the archived study root:
  `analysis/bundle_validation.json` and
  `verification/artifact_validation_review_fix4.log`
  -> `"ok": true`, `"reused_root_drift": {}`, `"fresh_row_completion_failures": {}`, `"merged_output_failures": {}`
- archived logs:
  - `verification/pytest_study_harness_review_fix4.log`
  - `verification/pytest_backlog_checks_review_fix4.log`
  - `verification/compileall_review_fix4.log`
  - `verification/pytest_integration_review_fix4.log`
  - `verification/artifact_validation_review_fix4.log`
