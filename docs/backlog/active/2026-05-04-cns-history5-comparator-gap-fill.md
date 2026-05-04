---
priority: 30
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-history5-comparator-gap-fill/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-05-01-cns-author-ffno-history-length-study
  - 2026-04-29-cns-spectral-history-len4plus-compare
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - The manuscript owner wants CNS result tables to use history_len=5 where possible.
  - history_len=5 rows already exist for authored FFNO and spectral SRU-Net*, but not for FNO or U-Net comparators.
  - This item fills only the missing comparator gaps; it must not rerun completed h5 rows.
---

# Backlog Item: Fill CNS History-Len-5 Comparator Gaps

## Objective

- Run `history_len=5`, `40`-epoch PDEBench CNS comparator rows for `fno_base`
  and `unet_strong`, using the same capped local CNS family as the completed
  authored-FFNO and spectral SRU-Net* h5 studies.

## Scope

- Reuse the official `2d_cfd_cns` dataset and task-local CNS runner.
- Match the completed h5 studies on:
  - `history_len=5`;
  - `epochs=40`;
  - capped split `512 / 64 / 64`;
  - `max_windows_per_trajectory=8`;
  - emitted windows `4096 / 512 / 512`;
  - train-only normalization;
  - MSE loss;
  - metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`,
    `fRMSE_mid`, `fRMSE_high`.
- Produce compare sidecars against each model's locked h2 row and against the
  h5 FFNO/SRU-Net* authorities.
- Update `pdebench_cns_history5_comparator_gap_fill_summary.md`,
  `paper_evidence_index.md`, `evidence_matrix.md`, and
  `model_variant_index.json`.

## Notes for Reviewer

- Do not rerun `author_ffno_cns_base` or
  `spectral_resnet_bottleneck_base`; consume the completed h5 authorities.
- Do not update the manuscript headline CNS table until both missing comparator
  rows are completed or explicitly blocked.
- If a row cannot complete, emit a row-level blocker and keep the current h2
  2048-cap table as the active CNS authority.
