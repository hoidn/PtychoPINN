## Completed In This Pass

- reran the final required production-workflow integration gate from the
  approved execution plan in `ptycho311`; the tracked pytest PID exited `0`
  and the archived log now lives at
  `verification/pytest_integration_review_fix6.log`
- synchronized this execution report and the durable checked-in study summary
  so the closeout evidence now reflects the refreshed integration proof in
  addition to the earlier `review_fix6` validator and harness repairs

## Completed Current-Scope Work

- review finding resolved: the final required
  `pytest -v -m integration` rerun is now archived at
  `verification/pytest_integration_review_fix6.log` with
  `5 passed, 4 skipped, 1816 deselected, 2 warnings in 301.63s (0:05:01)`,
  so the repaired state now satisfies the execution plan's full closeout
  verification contract
- the earlier `review_fix6` repair remains in force:
  `analysis/bundle_validation.json` still reports `"ok": true` with empty
  `shared_contract_failures`, `fresh_row_completion_failures`,
  `merged_output_failures`, and `reused_root_drift`, and the durable summary
  stays aligned with the final archived validation log

## Follow-Up Work

- expose the stricter bundle validator as a standalone CLI if later backlog
  work needs to audit archived CDI bridge-study roots without importing the
  runbook module

## Residual Risks

- this item remains CDI-only decision-support evidence; no fresh bridge row is
  promoted into paper-facing claim territory
- the pre-existing warning set in the closeout pytest runs remains unchanged:
  `tight_layout`, degenerate-case SSIM/MS-SSIM, FRC divide warnings from the
  approved backlog selector, plus the long-standing TensorFlow Addons
  compatibility warnings emitted by the integration marker
- this repair validates and metadata-repairs the archived study root; it does
  not rerun the full long CDI bridge study, so any future relaunch is still the
  point where launcher-native contract threading is exercised end to end
- tmux login shells in this environment still emit the pre-existing
  `/home/linuxbrew/.linuxbrew/bin/brew: No such file or directory` line before
  command execution; it did not affect the tracked PID exits or test outcomes

## Verification

- focused harness selector:
  `pytest -q tests/studies/test_cdi_hybrid_spectral_ffno_parameter_space.py`
  -> `20 passed in 3.42s`
- required backlog gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `191 passed, 49 warnings in 304.61s (0:05:04)`
- required production-workflow integration gate:
  `pytest -v -m integration`
  -> `5 passed, 4 skipped, 1816 deselected, 2 warnings in 301.63s (0:05:01)`
- compile check:
  `python -m compileall -q ptycho_torch scripts/studies` -> exit `0`
- repaired bundle validation on the archived study root after enforcing the
  canonical `probe_npz` digest contract:
  `analysis/bundle_validation.json` and
  `verification/artifact_validation_review_fix6.log`
  -> `"ok": true`, `"shared_contract_failures": []`,
     `"reused_root_drift": {}`, `"fresh_row_completion_failures": {}`,
     `"merged_output_failures": {}`
- archived logs:
  - `verification/pytest_study_harness_review_fix6.log`
  - `verification/pytest_backlog_checks_review_fix6.log`
  - `verification/pytest_integration_review_fix6.log`
  - `verification/compileall_review_fix6.log`
  - `verification/artifact_validation_review_fix6.log`
