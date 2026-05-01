# Execution Report

## Completed In This Pass

- Added the multi-reference same-profile history compare reporting path for the
  CNS spectral row, including contract validation, window-count capture, raw
  eligible-window capture, delta payloads, and first-regressed-metric
  reporting.
- Added regression tests covering the new dual-anchor history compare payload
  and the required fail-closed profile-id mismatch behavior.
- Recorded the finished spectral `history_len=4` and `history_len=5`
  follow-up in the durable docs/index surfaces and evidence indexes without
  reopening the locked `history_len=2` paper lane.

## Completed Plan Tasks

- Plan Task 1: extended reporting/test support for the spectral-only
  multi-reference compare sidecars and verified the new path with targeted
  tests plus the required CNS study suite selectors.
- Plan Task 2: completed the fresh inspect proofs, pilots, gate record, and
  compare sidecars under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/`.
- Plan Task 3: updated the discoverability and evidence surfaces:
  `pdebench_2d_cfd_cns_summary.md`, `docs/studies/index.md`, `docs/index.md`,
  `evidence_matrix.md`, `ablation_index.json`, and `model_variant_index.json`.

## Remaining Required Plan Tasks

- None under the approved scope for this backlog item.

## Verification

- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  passed: `57 passed in 61.49s`.
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  passed with exit code `0`.
- Repo-local consistency check passed for:
  summary existence, `history5_gate_decision.json`, all four compare JSON/CSV
  sidecar pairs, `evidence_matrix.md`, `ablation_index.json`,
  `model_variant_index.json`, `docs/index.md`, and `docs/studies/index.md`.
- Comparison standard recorded for this evidence pass:
  fixed capped CNS contract with identical emitted windows
  `4096 / 512 / 512`; only `history_len` and derived input/sample contracts
  were allowed to differ. No `atol`/`rtol` threshold applied because this pass
  was a bounded decision-support compare rather than a numeric parity test.

## Residual Risks

- The longer-context signal is still not stable at `10` epochs, so these rows
  remain `adjacent_capped_context_only`.
- `history_len=5` improved the spectral `40`-epoch row slightly again on the
  headline metrics but regressed `fRMSE_mid` versus `history_len=4`.
- The locked CNS paper lane remains `history_len=2`; downstream paper-facing
  consumers must keep using the contract decision and row-lock authorities.
