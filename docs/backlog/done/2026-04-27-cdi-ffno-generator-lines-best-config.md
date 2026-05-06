---
priority: 20
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/execution_plan.md
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
  configuration needed by the Lines128 paper-quality CDI benchmark design.

## Scope

- Use `docs/studies/index.md` to identify the relevant best lines run contract.
- Treat
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
  as the downstream benchmark authority: the paper-benchmark target is the
  study-indexed `N=128`, `gridsize=1`, fixed-seed `seed=3` grid-lines contract,
  unless a later checked-in pre-run decision artifact changes that contract for
  every row.
- Keep dataset, split, probe-scaling, loss, training budget, stitching, and
  metrics aligned with the Hybrid comparison row.
- Produce quantitative metrics plus standard amplitude/phase comparison figures.

## Notes for Reviewer

- Do not infer CDI generator quality from PDEBench CNS FFNO results.
- If FFNO cannot satisfy the generator output contract, record a blocker rather
  than changing the CDI workflow silently.
- Keep this separate from CNS-only FFNO and Hybrid-spectral ablations.
- This item unlocks the Lines128 paper benchmark harness; it does not by itself
  produce the full four-row paper table.
- Post-hoc architecture caveat from 2026-05-06: this completed row used
  `fno_cnn_blocks=2`, which adds local CNN residual refinement after the FFNO
  stack. Treat it as historical FFNO-local-refiner proxy evidence. Pure CDI
  FFNO claims require the active no-refiner rerun
  `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`.
