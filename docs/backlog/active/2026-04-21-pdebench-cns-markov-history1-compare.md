---
priority: 15
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_design.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites: []
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
---

# Backlog Item: Add CNS Markov History-1 Equal-Footing Compare

## Objective
- Run a controlled `history_len=1` Markov-style CNS compare so the repo can
  tell whether predicting `u[t]` from only `u[t-1]` helps
  `spectral_resnet_bottleneck_base` and changes the ranking against the current
  capped `history_len=2` rows.

## Scope
- Keep the current capped CNS slice fixed.
- Change only the history contract from `history_len=2` to `history_len=1`.
- Rerun `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, and
  `unet_strong` at `10` and `40` epochs.
- Document the result against the existing `history_len=2` anchor.

## Notes for Reviewer
- Do not change loss, split sizes, batch size, or metric family.
- This is a one-step Markov-style ablation, not a rollout study.
- If someone tries to rerun only the spectral row, that is incomplete for the
  equal-footing question; the full four-row set is required.
