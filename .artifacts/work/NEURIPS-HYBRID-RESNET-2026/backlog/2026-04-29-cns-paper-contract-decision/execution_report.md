# Execution Report

## Completed In This Pass

- Completed the missing row-level provenance contract in the CNS same-contract
  audit Markdown and JSON artifacts.
- Made the `history_len=3` authored-FFNO gap explicit as a blocked
  same-contract row instead of leaving it implicit at the lane level.
- Reran the required deterministic CNS checks from `ptycho311` and archived
  fresh logs after the audit fix.
- Added a focused audit-consistency validation log so Task 1 completion is
  backed by machine-checked row-level fields, not just narrative text.

## Completed Current-Scope Work

- Task 1: now complete to plan. The audit separates Lane A, Lane B, and
  contract-divergent context, and each candidate row now records row ID,
  run root, environment, dataset path, split counts, `history_len`,
  `max_windows_per_trajectory`, normalization contract, training loss, model
  profile ID, optimizer/scheduler fields, epoch budget, metrics, parameter
  count, runtime, and evidence status.
- Task 2: no additional work was required in this pass; the existing
  full-training compute/deadline estimate remains current.
- Task 3: no additional decision-doc edits were required once the audit
  contract was brought up to the approved row-level detail.
- Task 4: no additional downstream sync changes were required in this pass.
- Task 5: refreshed deterministic-check evidence and added an audit-contract
  consistency validation log for the corrected audit artifacts.

## Follow-Up Work

- Execute `2026-04-29-cns-paper-benchmark-rows` under the selected bounded
  capped contract to freeze the row manifests and write the row-lock summary.
- Execute `2026-04-29-cns-paper-table-figure-bundle` to publish the CNS table,
  fixed-sample visuals, and source-array manifests with the capped claim
  boundary.
- Execute `2026-04-29-paper-evidence-package-audit` after the CNS and CDI
  bundles exist so the paper-facing claim audit can consume both pillars.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  - result: `47 passed in 53.42s`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/verification/pytest_cns_contract_20260429_rerun.log`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  - result: exit code `0`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/verification/compileall_cns_contract_20260429_rerun.log`
- Audit contract consistency check
  - result: `audit contract consistency looks complete`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/verification/audit_contract_consistency.log`
- Parsed successfully:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_full_training_cost_estimate.json`

## Residual Risks

- The selected CNS paper lane is explicitly capped decision-support evidence,
  not a same-protocol full-training benchmark claim.
- The stronger local `history_len=3` `40`-epoch lane remains incomplete for the
  paper table because it lacks a same-contract authored FFNO row.
- A future full-training promotion would still require roughly `93.72` to
  `102.85` sequential GPU-hours on one RTX 3090 before recovery risk.
