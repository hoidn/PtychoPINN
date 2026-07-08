---
priority: 20
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-21-pdebench-cns-markov-history1-compare
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - The completed history_len=1 ablation showed that reducing temporal context hurt the spectral row; the symmetric question is whether extra context beyond the default history_len=2 helps.
  - Any benefit must be separated from changing sample count, normalization, loss, and optimization budget.
---

# Backlog Item: Test CNS History Length Beyond 2

## Objective

- Run a controlled PDEBench CNS ablation to determine whether increasing
  temporal context beyond the current default `history_len=2` improves the
  capped local CNS comparison.

## Scope

- Freeze the same official `2d_cfd_cns`, capped split family, MSE loss,
  normalization, batch-size policy, epoch budgets, and metric family used by
  the completed `history_len=1` versus `history_len=2` comparison.
- Treat the only intended task-contract delta as temporal context:
  `history_len=2 -> history_len=3`, with the derived input-channel count and
  valid-window count recorded explicitly.
- Run the same four comparison rows used by the history-1 ablation unless the
  fresh execution plan proves a narrower pilot is required first:
  `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, and
  `unet_strong`.
- Compare fresh `history_len=3` rows against frozen audited `history_len=2`
  anchors at matching epoch budgets.
- Add `history_len=4` only behind a predeclared gate: either `history_len=3`
  improves the spectral row on aggregate error without worsening high-frequency
  diagnostics, or the execution plan records a concrete scientific reason to
  test a second longer-context point before seeing metrics.

## Notes for Reviewer

- Do not change split counts, loss, metric family, batch size, or epoch budget
  just to make the longer-history row look better.
- Record the lower valid-window count caused by longer histories; if the sample
  contract changes enough to break equal-footing interpretation, label the
  result as context-ablation evidence rather than a direct model-ranking result.
- Do not turn this into an autoregressive rollout study or a full-training
  benchmark item.
- The completed `history_len=1` result is a prerequisite and context only; this
  item answers the opposite direction, not a rerun of the Markov ablation.
