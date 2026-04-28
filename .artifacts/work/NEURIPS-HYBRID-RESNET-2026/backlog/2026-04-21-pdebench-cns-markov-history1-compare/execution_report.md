# Execution Report

## Completed In This Pass

- Wrote the missing `40ep` cross-history compare sidecars:
  `compare_40ep_against_history2.json` and `.csv`.
- Added the durable Markov compare summary and updated the CNS summary,
  discoverability docs, and progress ledger.
- Archived the required pytest evidence log and completed the final verification
  gate for the approved scope.

## Completed Plan Tasks

- Task 1: Confirmed the history-only compare contract is already enforced in the
  current checkout; the required runner/data/metrics selector passed.
- Task 2: Validated the frozen history-2 reference manifest at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2_reference_runs.json`.
- Task 3: Confirmed the missing `40`-epoch `hybrid_resnet_cns` history-2 anchor
  was backfilled at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`.
- Task 4: Validated the fresh `history_len=1` four-row pilot runs at
  `history1-pilot-10ep-20260423T224907Z` and
  `history1-pilot-40ep-20260423T230352Z`.
- Task 5: Confirmed `compare_10ep_against_history2.json/csv` already existed
  and generated the missing `compare_40ep_against_history2.json/csv`.
- Task 6: Published the durable summary, updated the CNS summary, updated
  `docs/index.md` and `docs/studies/index.md`, and recorded the result in
  `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`.

## Remaining Required Plan Tasks

- None for the approved current scope.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  - `31 passed in 35.64s`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/pytest_markov_history1_compare_20260428T003543Z.log`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  - exit `0`
- Cross-history compare standard:
  `Only history_len and its derived sample/input-channel contract may differ.`
- Gallery alignment check:
  `np.allclose(..., atol=1e-6, rtol=1e-6)`

## Residual Risks

- All evidence remains capped decision-support only; this backlog item does not
  satisfy the benchmark-complete CNS gate.
- Both cross-history compare payloads blocked merged prediction/error galleries
  with `target_mismatch` because the saved sample-0 targets did not align
  exactly across the separate run roots.
- The conclusion remains summary-local and was not promoted into
  `docs/findings.md`.
