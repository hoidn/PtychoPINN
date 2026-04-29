# CNS Spectral History Length 4+ Compare Execution Plan

## Goal

Determine whether the PDEBench CNS `spectral_resnet_bottleneck_base` row,
reported as `SRU-Net*` in manuscript-facing materials, continues improving
beyond `history_len=3` under the same capped decision-support contract family.

## Scope

- Run `history_len=4` for `spectral_resnet_bottleneck_base` first.
- Compare against the frozen `history_len=2` and `history_len=3` spectral
  anchors from the completed CNS history-length studies.
- Run `history_len=5` only if `history_len=4` improves aggregate error without
  an unacceptable high-band regression, or if the implementation plan records a
  pre-run reason to test the second longer-history point.
- Keep the work row-local. Do not rerun FNO, U-Net, authored FFNO, or
  `hybrid_resnet_cns` unless a later checked-in roadmap decision expands the
  contract.

## Contract

- Task: PDEBench `2d_cfd_cns`
- Evidence scope: capped decision-support context only
- Model row: `spectral_resnet_bottleneck_base`
- Manuscript label mapping: `SRU-Net*`
- Metrics: `err_nRMSE`, `err_RMSE`, `relative_l2`, `fRMSE_low`,
  `fRMSE_mid`, `fRMSE_high`
- Required comparison: absolute metrics and deltas versus both `history_len=2`
  and `history_len=3`

## Guardrails

- Do not mix history lengths in the CNS headline model-ranking table.
- Record valid windows per trajectory because longer histories reduce available
  windows.
- Preserve the repo-row to manuscript-label mapping in all machine-readable
  outputs and summaries.
- If the equal-footing reduced setting cannot be preserved, write an explicit
  blocker or context-only limitation instead of widening the claim.

## Verification

Run the backlog item check commands before any result is accepted:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

