---
priority: 20
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md
check_commands:
  - pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-27-cdi-ffno-generator-lines-best-config
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The Lines128 paper benchmark design requires a shared wrapper/harness before the full benchmark can be run without dataset, metric, or provenance drift.
  - This item creates the pre-run contract and decision artifacts that prevent post-hoc comparator or metric selection.
---

# Backlog Item: Build Lines128 Paper Benchmark Harness

## Objective

- Extend the CDI/grid-lines benchmark path so Hybrid ResNet, Hybrid-spectral,
  FNO/FNO-vanilla, and FFNO can be run through one paper-quality Lines128
  wrapper or thin harness with shared dataset, provenance, metrics, and figures.

## Scope

- Prefer extending `scripts/studies/grid_lines_compare_wrapper.py`; add a thin
  dedicated Lines128 paper harness only if the existing wrapper cannot cleanly
  own the paper-benchmark collation.
- Keep `scripts/studies/grid_lines_torch_runner.py` as the per-model authority
  for model construction, training, inference, stitching, per-model metrics,
  and reconstruction arrays.
- Add wrapper/harness routing for `spectral_resnet_bottleneck_net` and the
  newly available FFNO CDI/grid-lines generator profile.
- Produce a durable pre-run contract-reconstruction validation artifact that
  names the historical sources, confidence for each reconstructed contract
  field, launch flags, go/no-go status, selected FNO comparator, and seed
  policy.
- Produce or emit a durable benchmark decision manifest covering the selected
  FNO comparator, fixed `seed=3` policy or approved multi-seed extension, and
  any approved deviations from the design.
- Add paper-metric schema enforcement so missing required metrics downgrade the
  merged result to `benchmark_incomplete` instead of silently passing.
- Add or update targeted tests for wrapper/harness routing, metric-schema
  validation, fixed-sample visual collation, and preflight failure modes.

## Notes for Reviewer

- Do not launch the full paper benchmark from this item. This is the harness,
  preflight, and contract-validation tranche.
- Do not choose `fno` versus `fno_vanilla` inside implementation code without
  writing the durable decision artifact required by the design.
- Do not change the CDI task contract to accommodate FFNO. If FFNO cannot emit
  the standard complex object/reconstruction outputs, block and explain the
  generator-contract mismatch.
- Treat historical `N=128` roots as contract-reconstruction inputs only, not as
  paper-grade results.
