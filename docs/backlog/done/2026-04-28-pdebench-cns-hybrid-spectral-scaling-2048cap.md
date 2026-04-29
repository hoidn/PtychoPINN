---
priority: 20
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
---

# Backlog Item: Check Hybrid-Spectral CNS Scaling At 2048-Cap

## Objective

- Extend the completed Hybrid-spectral CNS architecture ablation with a
  `2048 / 256 / 256` capped confirmation run so the repo can check whether the
  `512 -> 1024 -> 2048` training-set-size trend suggests a meaningful scaling
  property difference between the finalist rows.

## Scope

- Keep the same PDEBench `2d_cfd_cns` contract used by the completed
  Hybrid-spectral architecture ablation:
  - official `128x128` CNS file
  - `history_len=2`
  - `max_windows_per_trajectory=8`
  - `mse`
  - batch size `4`
  - `40` epochs
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`,
    `fRMSE_mid`, `fRMSE_high`
- Run only the new `2048 / 256 / 256` cap for the prior larger-cap finalists
  unless the plan records a concrete blocker:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`
- Compare the `2048 / 256 / 256` rows against the already recorded
  `512 / 64 / 64` and `1024 / 128 / 128` rows.
- Do not rerun the `512 / 64 / 64` or `1024 / 128 / 128` rows; treat those
  completed artifacts as frozen references for the scaling trend.
- Report both absolute metrics and scaling deltas:
  - `512 -> 1024`
  - `1024 -> 2048`
  - improvement per added training trajectory
  - runtime change

## Notes for Reviewer

- This is a trend check, not a full-training benchmark claim.
- Do not add new architecture axes, history-length changes, mode changes, FFNO
  rows, CDI/ptycho rows, or physics-regularization changes.
- The central question is whether one finalist keeps improving faster as the
  capped training set grows, not whether either row is paper-ready.
- Preserve the existing conclusion boundary if the trend is ambiguous: capped
  scaling evidence may inform the next backlog item, but it does not justify a
  default-profile promotion by itself.
