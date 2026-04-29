# Execution Report

## Completed In This Pass

- Ran the required deterministic CNS selectors and archived the logs under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/verification/`.
- Audited the accepted `history_len=2` capped CNS run roots against the locked
  paper contract and confirmed contract parity for the four headline rows plus
  the `hybrid_resnet_cns` continuity row.
- Emitted the row-lock artifacts:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_row_lock_audit.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_row_lock_audit.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`
- Wrote the durable summary
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
  and updated the discoverability/package surfaces that now point to it.

## Completed Plan Tasks

- Task 1 complete: required deterministic checks ran green and the accepted CNS
  rows were audited for same-contract parity, provenance pointers, metric
  values, and downstream sample-asset availability.
- Task 2 complete with no code patch required: the audit found no missing field
  that forced a runner/reporting change or rerun for this bounded capped lock.
- Task 3 complete: the locked-row manifest, durable summary, docs-index updates,
  and progress-ledger update were all written.

## Remaining Required Plan Tasks

- None for this execution plan.
- The downstream CNS table/figure assembly remains a separate backlog item:
  `2026-04-29-cns-paper-table-figure-bundle`.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/verification/pytest_20260429T000000Z.log`
  - result: `47 passed in 53.59s`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/verification/compileall_20260429T000000Z.log`
  - result: exit `0`
- Row-lock artifact and docs-sync validation
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/verification/row_lock_consistency_20260429T000000Z.log`
  - result: locked rows, audit, and discoverability references are consistent

## Residual Risks

- The reused accepted run roots still do not emit standalone repo git SHA,
  dirty-state, run-log, or exit-code artifacts, so the locked bundle remains
  `capped_decision_support` only and cannot be relabeled as `paper_grade` or
  `full_training` evidence from this pass alone.
- `hybrid_resnet_cns` remains continuity/support only; downstream consumers
  must preserve the locked headline roster exactly as recorded in the contract
  decision and row-lock manifest.
