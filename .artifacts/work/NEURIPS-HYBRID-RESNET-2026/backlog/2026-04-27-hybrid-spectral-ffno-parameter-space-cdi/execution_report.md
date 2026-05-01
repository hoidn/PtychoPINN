## Completed In This Pass

- added review-fix regression coverage for the CDI bridge-study harness:
  failed fresh-row leftovers must be scrubbed and relaunched, reused-anchor
  drift must abort before fresh launches, and final bundle validation must
  raise instead of returning a false-but-nonfatal report
- repaired `scripts/studies/runbooks/run_cdi_hybrid_spectral_ffno_parameter_space.py`
  so fresh rows only reuse outputs when they have full completion proof,
  stale failed/incomplete row outputs are deleted before relaunch, copied
  reused anchors are revalidated fail-closed before any fresh launch, and final
  bundle validation aborts on drift or missing required artifacts
- updated the durable CDI study summary to complete the Task 5 reporting
  contract for the reused `pinn_ffno` endpoint and to describe the repaired
  fail-closed/relaunch behavior without overclaiming

## Completed Current-Scope Work

- review finding 1 resolved: a leftover fresh-row `recon.npz` no longer causes
  silent reuse after a failed or incomplete run; the harness now scrubs the row
  root and relaunches unless the row has full completion proof
- review finding 2 resolved: reused-anchor drift now aborts before fresh
  launches and after final collation, rather than being written to JSON as a
  non-fatal warning
- review finding 3 resolved: the durable summary now reports each fresh row
  against both its nearest reused anchor and the reused `pinn_ffno` endpoint

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
  -> `11 passed in 3.16s`
- required backlog gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `191 passed, 49 warnings in 304.13s (0:05:04)`
- compile check:
  `python -m compileall -q ptycho_torch scripts/studies` -> exit `0`
- integration marker:
  `pytest -v -m integration`
  -> `5 passed, 4 skipped, 1807 deselected, 2 warnings in 302.21s (0:05:02)`
- repaired bundle validation on the archived study root:
  `verification/artifact_validation_review_fix2.log`
  -> `"ok": true`, `"reused_root_drift": {}`
- archived logs:
  - `verification/pytest_study_harness_review_fix2.log`
  - `verification/pytest_backlog_checks_review_fix2.log`
  - `verification/compileall_review_fix2.log`
  - `verification/pytest_integration_review_fix2.log`
  - `verification/artifact_validation_review_fix2.log`
