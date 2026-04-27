---
priority: 19
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
  - python -m compileall -q scripts/studies/pdebench_image128
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
---

# Backlog Item: Add Convolutional Features To FFNO On CNS

## Objective

- Add one or more FFNO variants with explicit local convolutional features and
  test whether they improve PDEBench `2d_cfd_cns` performance.

## Scope

- Keep the local CNS contract fixed.
- Compare against authored FFNO, local FFNO-close, and Hybrid-spectral anchors.
- Consider bounded variants such as a convolutional stem, local residual branch,
  or decoder-side refinement.

## Notes for Reviewer

- This is an FFNO-family extension, not a Hybrid-spectral ablation.
- Do not change split, loss, normalization, epoch budget, or metrics to make the
  new row look better.
- Report both aggregate `relative_l2` and high-frequency behavior.
