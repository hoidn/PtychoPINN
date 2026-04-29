---
priority: 20
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md
check_commands:
  - pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-27-cdi-ffno-generator-lines-best-config
  - 2026-04-29-cdi-lines128-paper-benchmark-harness
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The benchmark should run only after the FFNO generator profile and paper harness/preflight contract are complete.
  - The output is the paper-facing CDI table and reconstruction comparison package for the fixed Lines128 contract.
---

# Backlog Item: Execute Lines128 Paper-Quality CDI Benchmark

## Objective

- Run the full Lines128 paper-quality CDI benchmark for Hybrid ResNet,
  Hybrid-spectral, the selected FNO comparator, and FFNO, then publish the
  validated metrics tables, visual reconstruction comparisons, manifests, and
  durable summary.

## Scope

- Use the checked-in contract-reconstruction validation artifact and durable
  FNO/seed decision note produced by the harness item as launch inputs.
- Run all required rows under one fresh output root, one shared dataset/split,
  one fixed probe preprocessing contract, one loss/scheduler/epoch contract,
  and one metric schema.
- Launch long-running benchmark execution in tmux from the repo root with the
  required `ptycho311` environment, a unique output root, and exact PID
  tracking.
- Accept completion only when the tracked PID exits `0` and all required child
  row artifacts, merged metrics tables, metric schema, provenance manifests,
  reconstruction arrays, and visual comparison figures are freshly written.
- Write the durable result summary under
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/` and add the study result to
  `docs/studies/index.md`.
- Label the result `paper_complete` only if every required metric is present for
  every row or explicitly not applicable under the metric schema; otherwise
  label it `benchmark_incomplete` with missing-field reasons.

## Notes for Reviewer

- Do not reuse historical incomplete roots as paper evidence.
- Do not publish merged paper tables as final if any required row fails or any
  required metric is missing.
- Do not change the selected FNO comparator or seed policy after seeing
  metrics. Any change must be a pre-run design/decision amendment and rerun all
  rows under the revised contract.
- Keep multi-seed claims out of scope unless a separate checked-in extension
  pins seed list, aggregation rules, and runtime budget before launch.
