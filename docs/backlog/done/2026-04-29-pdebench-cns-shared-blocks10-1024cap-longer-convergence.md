---
priority: 19
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation
  - 2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - The 1024cap shared_blocks10 training curve was still dropping sharply over the final 10 epochs.
  - The 2048cap follow-up made the 1024cap convergence question material for interpreting the finalist comparison.
---

# Backlog Item: Run Shared-Blocks10 1024-Cap Longer Convergence

## Objective

- Rerun `spectral_resnet_bottleneck_shared_blocks10` on PDEBench CNS at the
  `1024 / 128 / 128` cap for a longer epoch budget to see the more fully
  converged eval metrics for that variant.

## Scope

- Use the same `2d_cfd_cns`, `history_len=2`, MSE, normalization, split family,
  batch-size, and shell contract as the completed 1024cap finalist run.
- Run only `spectral_resnet_bottleneck_shared_blocks10`; do not rerun the base
  finalist unless the selected plan explicitly justifies a same-budget control.
- Compare against the completed 40-epoch 1024cap finalist metrics and record
  whether the longer run changes the aggregate-vs-frequency interpretation.
- Preserve capped decision-support scope; this is not full-training benchmark
  evidence.

## Notes for Reviewer

- The motivating signal is that the 1024cap shared-blocks10 train MSE fell from
  `0.0104328` at epoch 31 to `0.0044652` at epoch 40, a `57.2%` last-window
  decrease, so the 40-epoch eval row may be under-converged.
- Report train-loss trajectory and held-out eval metrics together. Do not infer
  validation-loss behavior unless a real validation-loss series is emitted.
- Keep the comparison tied to the existing 1024cap and 2048cap summaries so the
  outcome answers whether shared-blocks10 was unfairly truncated at 1024cap.
