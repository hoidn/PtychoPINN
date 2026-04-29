# Execution Report

## Completed In This Pass

- Ran the required deterministic CNS checks and archived the logs.
- Audited the existing CNS evidence into explicit same-contract lanes.
- Chose and recorded the authoritative CNS paper contract:
  `bounded_capped_decision_support` on the `history_len=2`, `40`-epoch,
  `512 / 64 / 64`, `max_windows_per_trajectory=8`, `mse` lane.
- Wrote durable audit and full-training cost-estimate artifacts.
- Updated the downstream backlog/docs/state surfaces so later items consume the
  same contract.

## Completed Plan Tasks

- Task 1: completed the lane-by-lane same-contract audit and separated
  headline-eligible rows from contract-divergent context.
- Task 2: completed the full-training compute/deadline estimate using the
  existing capped `40`-epoch roots.
- Task 3: completed the durable CNS contract decision, including the authored
  FFNO cutoff/status and the locked row roster.
- Task 4: completed the downstream sync for the CNS row-lock item, the paper
  evidence package design/audit, discoverability indexes, and the NeurIPS
  progress ledger.

## Remaining Required Plan Tasks

- Execute `2026-04-29-cns-paper-benchmark-rows` under the selected bounded
  capped contract to freeze the row manifests and write the row-lock summary.
- Execute `2026-04-29-cns-paper-table-figure-bundle` to publish the CNS table,
  fixed-sample visuals, and source-array manifests with the capped claim
  boundary.
- Execute `2026-04-29-paper-evidence-package-audit` after the CNS and CDI
  bundles exist so the paper-facing claim audit can consume both pillars.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  - result: `47 passed in 53.21s`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/verification/pytest_cns_contract_20260429.log`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  - result: exit code `0`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/verification/compileall_cns_contract_20260429.log`
- Parsed successfully:
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_full_training_cost_estimate.json`
- Comparison standards carried into the decision:
  - cross-history sidecars:
    `Only history_len and its derived sample/input-channel contract may differ.`
  - cross-run gallery alignment where applicable:
    `np.allclose(..., atol=1e-6, rtol=1e-6)`

## Residual Risks

- The selected CNS paper lane is explicitly capped decision-support evidence,
  not a same-protocol full-training benchmark claim.
- The stronger local `history_len=3` `40`-epoch lane remains incomplete for the
  paper table because it lacks a same-contract authored FFNO row.
- A future full-training promotion would still require roughly `93.72` to
  `102.85` sequential GPU-hours on one RTX 3090 before recovery risk.
