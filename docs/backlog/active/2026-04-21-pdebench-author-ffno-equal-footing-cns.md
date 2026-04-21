---
priority: 14
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128
---

# Backlog Item: Add Author FFNO Equal-Footing CNS Compare

## Objective
- Run the authors' actual FFNO model on the same capped PDEBench `2d_cfd_cns`
  slice and epoch budgets already used for `spectral_resnet_bottleneck_base`,
  `fno_base`, and `unet_strong`.

## Scope
- Identify and document the authoritative author FFNO implementation to use.
- Adapt or wrap it enough to run the local one-step CNS contract without
  changing the slice, epoch budgets, or reported metrics.
- Produce `10`-epoch and `40`-epoch comparison artifacts on equal footing with
  the current local rows.

## Notes for Reviewer
- This is for the actual author FFNO model, not the repo's FFNO-close
  bottleneck proxy.
- Keep the equal-footing contract fixed: same dataset slice, same `history_len`,
  same `mse` training loss, same epoch counts, same metric family.
- If the imported author code cannot satisfy that fairness contract, require an
  explicit incompatibility note rather than a silent protocol drift.
