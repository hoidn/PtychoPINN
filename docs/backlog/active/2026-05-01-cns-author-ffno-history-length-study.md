---
priority: 29
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-01-cns-author-ffno-history-length-study/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py -k 'author_ffno'
  - pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-21-pdebench-author-ffno-equal-footing-cns
  - 2026-04-29-cns-spectral-history-len4plus-compare
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - SRU-Net has a completed longer-history CNS ablation through history_len=5, but authored FFNO is only locked at history_len=2.
  - The current CNS table must not imply whether authored FFNO benefits from additional temporal context until a same-model history-length study exists.
  - This item is adjacent capped context only unless a later CNS paper-contract decision reopens the headline history lane.
---

# Backlog Item: Test CNS Authored FFNO History Length

## Objective

- Run a controlled PDEBench CNS history-length ablation for
  `author_ffno_cns_base` so the project can say whether authored FFNO benefits
  from extra input frames under the same local CNS adapter, split family,
  normalization, loss, and metric conventions used by the existing CNS studies.

## Scope

- Start from the accepted `author_ffno_cns_base` `history_len=2`, `40`-epoch
  row as the frozen anchor.
- Reuse the official `2d_cfd_cns` file, capped `512 / 64 / 64` trajectory
  split, `max_windows_per_trajectory=8`, train-only per-field normalization,
  MSE loss, batch size, scheduler, and metric family unless the fresh plan
  proves that all compared FFNO anchors must be rerun under a changed contract.
- Run `history_len=3` first for `author_ffno_cns_base`.
- Compare fresh `history_len=3` metrics against the frozen `history_len=2`
  authored-FFNO anchor at matching budget.
- Run `history_len=4` only if `history_len=3` improves authored FFNO on
  aggregate error without an unacceptable high-band regression, or if the
  execution plan records a pre-run scientific reason to test a second
  longer-history point.
- Run `history_len=5` only behind the same predeclared gate after
  `history_len=4`.
- Report absolute metrics and deltas for at least `err_nRMSE`, `err_RMSE`,
  `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`, valid windows,
  train/validation/test counts, parameter count, runtime, and peak memory.
- Update the evidence matrix, model-variant index, and ablation index if any
  new FFNO history row is evaluated.

## Out of Scope

- Do not rerun SRU-Net, FNO, U-Net, local FFNO-close, or FFNO local-conv rows
  unless a later approved plan asks for a complete same-history CNS table.
- Do not mix FFNO longer-history rows into the locked CNS headline table.
- Do not claim paper-grade or full-training benchmark superiority from this
  capped ablation.
- Do not convert this into an autoregressive rollout study.

## Notes for Reviewer

- Check that the authored FFNO adapter truly accepts the wider history-derived
  input channel count without silently changing the model body, task split,
  normalization, or target horizon.
- Longer histories reduce valid windows per trajectory. The summary must state
  whether the sample-contract delta still supports a direct within-model
  temporal-context claim.
- The scientific question is not whether FFNO is strong at `history_len=2`;
  that is already established locally. The question is whether additional
  frames improve, saturate, or degrade the same authored-FFNO row.
- Any manuscript language should distinguish this item from the locked
  `history_len=2` CNS paper table unless a later roadmap-level decision changes
  the headline contract.
