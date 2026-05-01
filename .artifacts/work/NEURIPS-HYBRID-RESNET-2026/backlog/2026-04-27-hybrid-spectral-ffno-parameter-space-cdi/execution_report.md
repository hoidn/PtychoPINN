## Completed In This Pass

- added regression coverage for the remaining review gaps:
  fresh-row dataset-path and `probe_source` drift, shared-contract manifest
  drift, and collated-table display-label drift now fail closed in the study
  harness tests
- repaired the CDI parameter-space study plumbing so the frozen matrix carries
  explicit fresh-row `train_npz`/`test_npz`/`probe_source` expectations, the
  compare wrapper now threads `probe_source` into Torch launches, and the
  closeout validator checks the shared study manifests in addition to row-local
  invocation/config projections
- narrowed metrics-table label overrides to explicit row-spec labels only, so
  the CDI study bundle uses the approved frozen labels without regressing the
  wrapper’s generic architecture-mode tables
- refreshed the consumed archived study root:
  `preflight/study_matrix.json` and `preflight/reference_runs.json` were
  regenerated, fresh-row invocation/config metadata was backfilled with the
  fixed `probe_source: custom` contract, the collated CSV/LaTeX tables were
  regenerated with frozen labels, and `analysis/bundle_validation.json` now
  reports a clean fail-closed validation pass

## Completed Current-Scope Work

- review finding 1 resolved: the closeout validator now fails closed on the
  full current-scope row contract for fresh launches, including row-local
  dataset paths and `probe_source`, and it separately rejects shared-contract
  drift in `dataset_identity_manifest.json`, `split_manifest.json`, and
  `preflight/preflight_validation.json`
- review finding 2 resolved: the collated `metrics_table.csv` and
  `metrics_table.tex` now use the frozen bridge-study display labels and the
  validator rejects raw-ID label regressions in both outputs
- the archived machine-readable validation artifact named by the durable
  summary has been refreshed and now reports:
  `"ok": true`, empty `shared_contract_failures`, empty
  `fresh_row_completion_failures`, empty `merged_output_failures`, and empty
  `reused_root_drift`

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
- this repair validates and metadata-repairs the archived study root; it does
  not rerun the full long CDI bridge study, so any future relaunch is still the
  point where the fixed `probe_source` threading becomes launcher-native
- tmux login shells in this environment still emit the pre-existing
  `/home/linuxbrew/.linuxbrew/bin/brew: No such file or directory` line before
  command execution; it did not affect the tracked PID exits or test outcomes

## Verification

- focused harness selector:
  `pytest -q tests/studies/test_cdi_hybrid_spectral_ffno_parameter_space.py`
  -> `18 passed in 3.23s`
- required backlog gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `191 passed, 49 warnings in 304.72s (0:05:04)`
- compile check:
  `python -m compileall -q ptycho_torch scripts/studies` -> exit `0`
- integration marker:
  `pytest -v -m integration`
  -> `5 passed, 4 skipped, 1814 deselected, 2 warnings in 302.71s (0:05:02)`
- repaired bundle validation on the archived study root after regenerating the
  frozen study matrix, relabeling the collated tables, and backfilling the
  missing fresh-row `probe_source` metadata:
  `analysis/bundle_validation.json` and
  `verification/artifact_validation_review_fix5.log`
  -> `"ok": true`, `"shared_contract_failures": []`,
     `"reused_root_drift": {}`, `"fresh_row_completion_failures": {}`,
     `"merged_output_failures": {}`
- archived logs:
  - `verification/pytest_study_harness_review_fix5.log`
  - `verification/pytest_backlog_checks_review_fix5.log`
  - `verification/compileall_review_fix5.log`
  - `verification/pytest_integration_review_fix5.log`
  - `verification/artifact_validation_review_fix5.log`
