## Completed In This Pass

- resolved the implementation-review blocker by making the compare-wrapper
  collation path persist a top-level `model_manifest.json` from the already
  assembled per-row payloads and by making the CDI bundle validator require and
  parse that manifest alongside the merged metrics/table artifacts
- repaired the delivered archived study root in place without relaunching the
  long CDI rows: reran only the cheap reuse-existing-recons collation path,
  wrote the missing `model_manifest.json`, and refreshed
  `analysis/bundle_validation.json`
- reran the focused selectors plus the required backlog, integration, and
  compile gates and archived fresh `review_fix7` logs under `verification/`

## Completed Current-Scope Work

- review finding resolved: the delivered study bundle now includes the required
  collated model-manifest artifact at `model_manifest.json`
- validator contract closed: the repaired archived bundle now fails closed if
  `model_manifest.json` is missing or if its `rows[*].model_id` set drifts from
  the frozen six-row study matrix
- archived bundle repair verified:
  `analysis/bundle_validation.json` and
  `verification/artifact_validation_review_fix7.log` now report `"ok": true`
  with empty `missing_merged_outputs`, `merged_output_failures`,
  `fresh_row_completion_failures`, `reused_root_drift`, and
  `shared_contract_failures`
- required closeout verification refreshed on the repaired code path:
  `verification/pytest_backlog_checks_review_fix7.log`,
  `verification/pytest_integration_review_fix7.log`, and
  `verification/compileall_review_fix7.log`

## Follow-Up Work

- expose the stricter bundle validator as a standalone CLI if later backlog
  work needs to audit archived CDI bridge-study roots without importing the
  runbook module
- align any remaining documentation that still describes the merged output set
  without `model_manifest.json`

## Residual Risks

- this item remains CDI-only decision-support evidence; no fresh bridge row is
  promoted into paper-facing claim territory
- the repair re-collates and revalidates the archived root but does not rerun
  the full long CDI bridge study, so a future launcher change would still need
  fresh end-to-end execution proof
- the pre-existing warning set in the closeout pytest runs remains unchanged:
  `tight_layout`, degenerate-case SSIM/MS-SSIM, FRC divide warnings from the
  approved backlog selector, plus the long-standing TensorFlow Addons
  compatibility warnings emitted by the integration marker
- tmux login shells in this environment still emit the pre-existing
  `/home/linuxbrew/.linuxbrew/bin/brew: No such file or directory` line before
  command execution; it did not affect the tracked test outcomes

## Verification

- focused study-harness selector:
  `pytest -q tests/studies/test_cdi_hybrid_spectral_ffno_parameter_space.py`
  -> `20 passed in 3.70s`
- focused compare-wrapper selector:
  `pytest -q tests/test_grid_lines_compare_wrapper.py`
  -> `62 passed, 23 warnings in 17.97s`
- required backlog gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `191 passed, 49 warnings in 306.15s (0:05:06)`
- required production-workflow integration gate:
  `pytest -v -m integration`
  -> `5 passed, 4 skipped, 1816 deselected, 2 warnings in 301.99s (0:05:01)`
- compile check:
  `python -m compileall -q ptycho_torch scripts/studies` -> exit `0`
- archived bundle repair and deterministic validation:
  `verification/artifact_validation_review_fix7.log`
  -> `model_manifest_exists: true`, `missing_merged_outputs: []`,
     `merged_output_failures: {}`, `ok: true`
- archived logs:
  - `verification/pytest_study_harness_review_fix7.log`
  - `verification/pytest_grid_compare_wrapper_review_fix7.log`
  - `verification/pytest_backlog_checks_review_fix7.log`
  - `verification/pytest_integration_review_fix7.log`
  - `verification/compileall_review_fix7.log`
  - `verification/artifact_validation_review_fix7.log`
