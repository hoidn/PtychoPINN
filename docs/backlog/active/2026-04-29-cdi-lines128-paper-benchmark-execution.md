---
priority: 20
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md
check_commands:
  - pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-27-cdi-ffno-generator-lines-best-config
  - 2026-04-29-cdi-lines128-paper-benchmark-harness
  - 2026-04-29-cdi-lines128-minimum-paper-table
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The complete benchmark should run only after the FFNO generator profile, paper harness/preflight contract, and minimum Hybrid/CNN-PINN/FNO subset are complete.
  - The output is the complete paper-facing CDI table and reconstruction comparison package for the fixed Lines128 contract, including spectral_resnet_bottleneck_net and FFNO.
---

# Backlog Item: Execute Lines128 Paper-Quality CDI Benchmark

## Objective

- Extend the minimum Lines128 paper-quality CDI subset with the required
  `spectral_resnet_bottleneck_net` and FFNO rows, then publish the complete
  validated metrics tables, visual reconstruction comparisons, manifests, and
  durable summary.

## Scope

- Use the checked-in contract-reconstruction validation artifact, durable
  FNO/seed decision note, and minimum CDI table summary as launch inputs.
- Produce the `spectral_resnet_bottleneck_net` and FFNO rows under the same
  dataset/split, probe preprocessing contract, loss/scheduler/epoch contract,
  sample IDs, and metric schema used by the minimum subset. First audit existing
  row roots such as the completed FFNO generator run; promote them if the
  current contract is satisfied directly or after deterministic provenance
  recovery. Rerun only rows with an actual contract mismatch or unrecoverable
  metadata, metric, visual, or provenance gap.
- Launch long-running benchmark execution in tmux from the repo root with the
  required `ptycho311` environment, a unique output root, and exact PID
  tracking.
- Accept newly launched runs only when the tracked PID exits `0` and all
  required child row artifacts, merged metrics tables, metric schema,
  provenance manifests, reconstruction arrays, and visual comparison figures
  are freshly written. Accept promoted existing roots only when the audit
  manifest records the source root, contract match, recovered fields, and any
  rejected gaps.
- Write the durable result summary under
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/` and add the study result to
  `docs/studies/index.md`.
- Label the result `paper_complete` only if every required metric is present for
  every row or explicitly not applicable under the metric schema; otherwise
  label it `benchmark_incomplete` with missing-field reasons.

## Notes for Reviewer

- Do not reuse roots with unrecoverable contract or provenance gaps as paper
  evidence. A root's original exploratory or decision-support label is not by
  itself disqualifying if the current audit proves the row contract is complete.
- Do not publish merged paper tables as final if any required row fails or any
  required metric is missing.
- Do not change the selected FNO comparator or seed policy after seeing
  metrics. Any change must be a pre-run design/decision amendment and rerun all
  rows under the revised contract.
- Keep multi-seed claims out of scope unless a separate checked-in extension
  pins seed list, aggregation rules, and runtime budget before launch.
