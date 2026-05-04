---
priority: 21
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/execution_plan.md
check_commands:
  - pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-27-cdi-ffno-generator-lines-best-config
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
---

# Backlog Item: Explore CDI Hybrid-Spectral To FFNO Parameter Space

## Objective

- Run the CDI/ptycho half of the Hybrid-spectral to FFNO architecture-space
  study on the best study-indexed lines configuration after the CDI FFNO
  generator baseline is available.

## Scope

- Compare selected intermediate architecture points between the current
  Hybrid-spectral family and FFNO-like variants on CDI/ptycho only.
- Include one-axis-at-a-time encoder/downsampling, decoder, and bottleneck
  variants where each row has an attributable architecture-family change.
- Use study-indexed CDI evidence; do not substitute PDEBench CNS metrics for
  CDI metrics.

## Notes for Reviewer

- This is the Phase 3 portion of the former mixed CNS/CDI parameter-space item.
- The current deterministic Phase 2 roadmap gate should exclude this item until
  the roadmap explicitly opens Phase 3 CDI work.
- Report CDI/ptycho outcomes separately from the Phase 2 CNS split item rather
  than collapsing both domains into one scalar ranking.
