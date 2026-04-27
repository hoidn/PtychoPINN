---
priority: 16
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
---

# Backlog Item: Add CNS Spectral Modes-32 Compare

## Objective
- Increase both encoder and bottleneck spectral modes from `12` to `32` for the
  spectral CNS variant and see whether that improves capped CNS metrics.

## Scope
- Add a manual spectral profile with `fno_modes=32` and
  `spectral_bottleneck_modes=32`.
- Keep the current capped CNS slice and training contract fixed.
- Run `10`-epoch and `40`-epoch comparisons for the higher-mode spectral row.
- Compare the result against the current `12/12` spectral row and the existing
  `fno_base` / `unet_strong` anchors.

## Notes for Reviewer
- Change both mode knobs together; do not test only one of them in this item.
- Do not widen the scope into a full mode sweep.
- Keep all non-mode spectral config unchanged so the result is attributable.
