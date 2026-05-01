## Completed In This Pass

- added the missing review regression coverage for exact `probe_npz` identity:
  the harness now fails when the manifest is repointed to a different
  same-basename probe, and it separately proves that declared repo-relative
  probe contracts still validate when they resolve to the canonical file
- repaired shared-contract validation so `probe_npz` no longer trusts a
  self-consistent manifest entry; it now checks the actual/recorded digest
  against the canonical frozen-contract probe while preserving legitimate
  byte-identical copies used by the study harness
- refreshed the archived machine-readable validation artifact and synchronized
  the durable checked-in summary so the documented focused-test count and
  artifact-validation log now match the final repaired evidence

## Completed Current-Scope Work

- review finding 1 resolved: the bundle validator now rejects probe drift even
  when the substituted file keeps the same basename and a recomputed manifest
  SHA, because the recorded probe must match the canonical frozen-contract
  digest
- review finding 2 resolved: the durable checked-in summary now cites the
  final focused harness count (`20 passed`) and the refreshed archived
  validation log `verification/artifact_validation_review_fix6.log`
- the refreshed archived validation artifact at
  `analysis/bundle_validation.json` now reports `"ok": true` with empty
  `shared_contract_failures`, `fresh_row_completion_failures`,
  `merged_output_failures`, and `reused_root_drift`

## Follow-Up Work

- expose the stricter bundle validator as a standalone CLI if later backlog
  work needs to audit archived CDI bridge-study roots without importing the
  runbook module

## Residual Risks

- this item remains CDI-only decision-support evidence; no fresh bridge row is
  promoted into paper-facing claim territory
- the pre-existing warning set in the closeout pytest runs remains unchanged:
  `tight_layout`, degenerate-case SSIM/MS-SSIM, and FRC divide warnings from
  the approved backlog selector
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
  - `verification/compileall_review_fix6.log`
  - `verification/artifact_validation_review_fix6.log`
