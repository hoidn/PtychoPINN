---
priority: 20
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_plan.md
check_commands:
  - pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
  - python -m compileall -q ptycho_torch scripts/studies
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
---

# Backlog Item: Use FFNO As CDI Generator On Best Lines Config

## Objective

- Add an FFNO generator option to the CDI/ptycho Torch reconstruction path and
  compare it against Hybrid ResNet on the best documented lines data/training
  configuration.

## Scope

- Use `docs/studies/index.md` to identify the relevant best lines run contract.
- Keep dataset, split, probe-scaling, loss, training budget, stitching, and
  metrics aligned with the Hybrid comparison row.
- Produce quantitative metrics plus standard amplitude/phase comparison figures.

## Notes for Reviewer

- Do not infer CDI generator quality from PDEBench CNS FFNO results.
- If FFNO cannot satisfy the generator output contract, record a blocker rather
  than changing the CDI workflow silently.
- Keep this separate from CNS-only FFNO and Hybrid-spectral ablations.
