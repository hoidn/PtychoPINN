---
priority: 21
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_spectral_ffno_parameter_space_cns_cdi_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/torch/test_grid_lines_hybrid_resnet_integration.py
  - python -m compileall -q scripts/studies ptycho_torch
prerequisites:
  - 2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation
  - 2026-04-27-pdebench-ffno-convolutional-features-cns
  - 2026-04-27-cdi-ffno-generator-lines-best-config
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
  - phase-3-cdi-anchor-regeneration
---

# Backlog Item: Explore Hybrid-Spectral To FFNO Architecture Space

## Objective

- Run a staged architecture study of intermediate points between
  Hybrid-spectral and FFNO, with comparisons on both CNS and CDI/ptycho.

## Scope

- Include encoder/downsampling ablations.
- Include decoder ablations.
- Include bottleneck ablations.
- Compare each selected row on PDEBench `2d_cfd_cns` and on the best
  study-indexed CDI/ptycho lines configuration.

## Notes for Reviewer

- This is deliberately blocked on narrower CNS and CDI FFNO follow-ups so the
  study does not start as an unbounded Cartesian sweep.
- Report CNS and CDI/ptycho separately. Do not collapse them into one scalar
  ranking.
- Keep each row attributable by changing one architecture-family axis at a time.
