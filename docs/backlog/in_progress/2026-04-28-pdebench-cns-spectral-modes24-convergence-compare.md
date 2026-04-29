---
priority: 20
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-spectral-modes24-convergence-compare/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-21-pdebench-cns-spectral-modes32-compare
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
---

# Backlog Item: Revisit CNS Spectral Mode Count At Converged Budget

## Objective

- Revisit the completed modes-32 CNS result with a convergence-oriented budget
  and a less aggressive mode increase, comparing the shared spectral `12/12`
  row against a new `24/24` row after both have had enough epochs for a fair
  optimization read.

## Scope

- Keep the local PDEBench `2d_cfd_cns` contract fixed unless the execution plan
  records a concrete blocker:
  - official `128x128` CNS file
  - `1024 / 128 / 128` capped split
  - `history_len=2`
  - `max_windows_per_trajectory=8`
  - `mse`
  - batch size `16`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`,
    `fRMSE_mid`, `fRMSE_high`
- Use batch size `16` for both rows. If `16` fails for a concrete runtime
  reason, fall back to the largest smaller identical batch size that allows
  both rows to run, and record the fallback reason.
- Compare exactly two mode settings under the same convergence budget:
  - baseline: `fno_modes=12`, `spectral_bottleneck_modes=12`
  - candidate: `fno_modes=24`, `spectral_bottleneck_modes=24`
- Rerun both rows fresh under the selected `1024 / 128 / 128`
  long/convergence budget. Do not use the old `40`-epoch `12/12` row as the
  final baseline for this item.
- The plan must define the convergence standard before launching runs, using
  the prior `40`-epoch loss trajectories as motivation. At minimum, report:
  - train loss trajectory
  - late-epoch loss slope or last-window delta
  - eval metrics at the final checkpoint
  - whether either row is still materially improving at stop time

## Notes for Reviewer

- This is not a full mode sweep. Do not add `16`, `32`, decoupled
  encoder-only, or bottleneck-only mode settings unless this item is explicitly
  split first.
- The purpose is to separate mode-count value from under-convergence. If both
  rows are still improving at the chosen stop point, mark the result
  inconclusive rather than promoting either profile.
- Keep the completed modes-32 run as context only. The primary comparison here
  is converged-budget `12/12` versus `24/24`.
- Prior batch-4 rows are context only; do not mix them into the final ranking
  as if the optimization contract were identical.
- Do not widen this into the `2048` training-set-size scaling item, FFNO rows,
  CDI/ptycho rows, history-length changes, or physics regularization.
