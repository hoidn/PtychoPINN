# Execution Report

## Completed In This Pass

- Resumed the backlog item in the current checkout and confirmed the tracked
  `40`-epoch `history_len=3` CNS pilot finished cleanly with exit code `0` at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`
- Generated the missing reporting payloads for the finished run:
  - `compare_40ep_history3_against_history2.json/csv`
  - `history4_gate_decision.json`
- Wrote the durable longer-context summary, synced the CNS summary plus docs
  indexes, added the completion record to
  `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`, and wrote the
  implementation-state bundle.

## Completed Plan Tasks

- Task 1: audit-only reuse remained valid; no production code patching was
  required for the existing history-delta reporting helpers or selectors.
- Task 2: the frozen `history_len=2` manifest and the `history_len=3` inspect
  proof were already present and stayed valid for the approved contract.
- Task 3: both fresh `history_len=3` pilot roots are complete with tracked exit
  code `0`, and the required `10`/`40`-epoch cross-history compare sidecars now
  exist.
- Task 4: the mandatory `history4` gate record was written and stayed closed
  because the spectral `10`-epoch and `40`-epoch signals disagree under the
  plan’s decision rule.
- Task 5: the durable summary, CNS summary, docs entry points, progress ledger,
  execution report, and implementation-state output are complete.

## Remaining Required Plan Tasks

- None for the approved current scope.
- A gated `history_len=4` pilot branch was explicitly not authorized because
  the recorded gate stayed closed. Any future `history_len=4` work needs a new
  approved scope or a revised gate rationale.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/final_pytest.log`
  - result: `45 passed in 51.67s`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/final_compileall.log`
  - result: exit `0`
- Compare/gate payload validation:
  - compare generation log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/history3_compare_generation.log`
  - gate decision log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/history4_gate_decision.log`
  - payload validation log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/final_output_validation.log`
  - result: `final output validation passed`
- Run completion proof:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/launch-history3-pilot-10ep-20260429T071905Z/exit_code = 0`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/launch-history3-pilot-40ep-20260429T073705Z/exit_code = 0`
  - fresh run roots contain the required `comparison_summary.json`,
    `metrics_*.json`, `model_profile_*.json`, `dataset_manifest.json`,
    `split_manifest.json`, and `invocation.json` artifacts

## Residual Risks

- The result is still capped decision-support evidence only and does not satisfy
  the full-training benchmark gate for CNS.
- The optional `history_len=4` branch did not open because the spectral signal
  was not stable across the approved `10`- and `40`-epoch budgets, so this item
  does not answer whether even longer context would help under a stronger
  budget.
- A pre-gate `history4` inspect proof remains under the artifact root as an
  exploratory contract check, but it is not an authorized follow-up run and
  could confuse later readers if they ignore the closed gate record.
