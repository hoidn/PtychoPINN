## Completed In This Pass

- hardened the CDI bridge-study preflight so it now freezes the authoritative
  `lines128` bundle contract from the complete-table manifests, validates
  reused-row runner arguments and provenance fail-closed, and emits a richer
  checked-in preflight note plus v2 `study_matrix.json` and
  `reference_runs.json`
- changed reused-anchor materialization in the runbook from live symlinks to
  copy-on-write copies and extended bundle validation to reject reused-row
  symlinks or digest drift against the frozen authoritative bundle
- repaired the existing study artifact root in place by replacing the reused
  `runs/*`, `recons/*`, and `recons/gt` symlinks with copied directories, then
  regenerated `analysis/bundle_validation.json` so the current durable output
  validates under the new contract
- updated durable discoverability and summary docs so the preflight authority
  and copy-on-write repair are reflected in repo docs

## Completed Current-Scope Work

- review finding 1 resolved: the study root no longer exposes writable
  symlinks into the authoritative complete-table bundle, and the harness now
  refuses that materialization mode
- review finding 2 resolved: reused-root contract and provenance validation now
  happens before launch through the authoritative bundle manifests plus
  row-local invocation checks, and final bundle validation now checks copied
  reused rows against frozen source digests
- review finding 3 resolved: the checked-in preflight note now repeats the
  frozen row roster, nearest anchors, expression paths, display labels,
  output-root layout, and reuse-acceptability rationale required by Task 1

## Follow-Up Work

- none for this backlog item

## Residual Risks

- this item remains CDI-only decision-support evidence; no fresh bridge row is
  promoted into paper-facing claim territory
- the pre-existing warning set in the closeout pytest runs remains unchanged:
  `tight_layout`, degenerate-case SSIM/MS-SSIM, FRC divide warnings, and the
  known TensorFlow Addons compatibility warnings in `pytest -m integration`

## Verification

- focused harness selector:
  `pytest -q tests/studies/test_cdi_hybrid_spectral_ffno_parameter_space.py`
  -> `8 passed in 3.20s`
- required backlog gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `191 passed, 49 warnings in 303.94s (0:05:03)`
- compile check:
  `python -m compileall -q ptycho_torch scripts/studies` -> exit `0`
- integration marker:
  `pytest -v -m integration`
  -> `5 passed, 4 skipped, 1804 deselected, 2 warnings in 301.58s (0:05:01)`
- repaired bundle validation:
  `analysis/bundle_validation.json` -> `"ok": true`, `"reused_root_drift": {}`
- archived logs:
  - `verification/pytest_study_harness_review_fix.log`
  - `verification/pytest_backlog_checks_review_fix.log`
  - `verification/compileall_review_fix.log`
  - `verification/pytest_integration_review_fix.log`
