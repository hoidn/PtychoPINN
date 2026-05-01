## Completed In This Pass

- refreshed the durable checked-in CDI bridge-study summary so the indexed
  authority now points at the final `review_fix8` evidence instead of the stale
  `review_fix7` verification state
- resolved the remaining implementation-review blocker by preserving the study
  claim boundary in the collated `model_manifest.json` and by preventing the
  three fresh bridge rows from auto-upgrading to `paper_grade` on artifact
  completeness alone
- repaired the delivered archived study root in place without relaunching the
  long CDI rows: reran only the cheap reuse-existing-recons collation path,
  rewrote `model_manifest.json` with the corrected claim semantics, and
  refreshed
  `analysis/bundle_validation.json`
- made the bundle validator fail closed on future claim-boundary or locked
  fresh-row-status drift and archived fresh `review_fix8` verification logs
  under `verification/`

## Completed Current-Scope Work

- checked-in summary authority repaired:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_hybrid_spectral_ffno_parameter_space_summary.md`
  now reports the final `review_fix8` verification counts and archived log
  paths (`21` harness tests, `63` compare-wrapper tests, `192` backlog-gate
  tests, and `verification/artifact_validation_review_fix8.log`)
- review finding resolved: the delivered study bundle now records
  `claim_boundary: "no_paper_promotion_without_later_authority"` at
  `model_manifest.json`
- fresh-row evidence boundary restored: the delivered manifest now keeps
  `pinn_spectral_resnet_bottleneck_ds1`,
  `pinn_spectral_resnet_bottleneck_linear_decoder`, and
  `pinn_hybrid_resnet_ffno_bottleneck` at `row_status: "decision_support"`
  while preserving the reused paper-grade anchors
- validator contract closed: the repaired archived bundle now fails closed if
  `model_manifest.json` is missing, if its `rows[*].model_id` set drifts from
  the frozen six-row study matrix, if its top-level `claim_boundary` drifts
  from the study matrix, or if any locked fresh-row status deviates from the
  approved decision-support contract
- archived bundle repair verified:
  `analysis/bundle_validation.json` and
  `verification/artifact_validation_review_fix8.log` now report `"ok": true`
  with the expected no-promotion claim boundary, decision-support fresh-row
  statuses, and empty `missing_merged_outputs`, `merged_output_failures`,
  `fresh_row_completion_failures`, `reused_root_drift`, and
  `shared_contract_failures`
- required closeout verification refreshed on the repaired code path:
  `verification/pytest_study_harness_review_fix8.log`,
  `verification/pytest_grid_compare_wrapper_review_fix8.log`,
  `verification/pytest_backlog_checks_review_fix8.log`, and
  `verification/compileall_review_fix8.log`

## Follow-Up Work

- consider either populating
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/verification/preflight.log`
  or removing that empty file in favor of the existing
  `preflight/preflight_validation.json`, so the archived preflight evidence
  surface is unambiguous
- expose the stricter bundle validator as a standalone CLI if later backlog
  work needs to audit archived CDI bridge-study roots without importing the
  runbook module
- align any remaining documentation that still describes this bridge-study
  bundle without the no-promotion manifest boundary or locked fresh-row status
  semantics

## Residual Risks

- this item remains CDI-only decision-support evidence; no fresh bridge row is
  promoted into paper-facing claim territory
- the repair re-collates and revalidates the archived root but does not rerun
  the full long CDI bridge study, so a future launcher change would still need
  fresh end-to-end execution proof
- the pre-existing warning set in the closeout pytest runs remains unchanged:
  `tight_layout`, degenerate-case SSIM/MS-SSIM, FRC divide warnings, and the
  existing `params.cfg`/test-data-file warnings from the approved backlog
  selector
- tmux login shells in this environment still emit the pre-existing
  `/home/linuxbrew/.linuxbrew/bin/brew: No such file or directory` line before
  command execution; it did not affect the tracked test outcomes

## Verification

- focused study-harness selector:
  `pytest -q tests/studies/test_cdi_hybrid_spectral_ffno_parameter_space.py`
  -> `21 passed in 3.64s`
- focused compare-wrapper selector:
  `pytest -q tests/test_grid_lines_compare_wrapper.py`
  -> `63 passed, 23 warnings in 17.75s`
- required backlog gate:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `192 passed, 49 warnings in 303.88s (0:05:03)`
- compile check:
  `python -m compileall -q ptycho_torch scripts/studies` -> exit `0`
- archived bundle repair and deterministic validation:
  `verification/artifact_validation_review_fix8.log`
  -> `claim_boundary: "no_paper_promotion_without_later_authority"`,
     fresh bridge rows remain `decision_support`, `ok: true`
- archived logs:
  - `verification/pytest_study_harness_review_fix8.log`
  - `verification/pytest_grid_compare_wrapper_review_fix8.log`
  - `verification/pytest_backlog_checks_review_fix8.log`
  - `verification/compileall_review_fix8.log`
  - `verification/artifact_validation_review_fix8.log`
