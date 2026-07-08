---
priority: 21
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution_plan.md
check_commands:
  - pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-harness
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The paper can start drafting result shells once the minimum CDI subset has paper-grade hybrid_resnet, CDI `cnn` U-Net-class supervised, CDI `cnn` U-Net-class + PINN, and FNO rows under one Lines128 contract.
  - Spectral_resnet_bottleneck_net and FFNO remain required for the complete Lines128 benchmark, but they should not block the first minimum CDI claim subset if the core rows are ready.
---

# Backlog Item: Run Minimum Lines128 Paper CDI Table

## Objective

- Produce the minimum paper-grade Lines128 CDI result subset and visual bundle
  for `hybrid_resnet`, paired local convolutional baselines, and the selected
  FNO comparator under one locked contract.

## Scope

- Use the paper benchmark harness preflight artifact as the source of truth for
  dataset, split, probe preprocessing, seed policy, training budget, metric
  schema, sample IDs, and visual scales.
- Run or regenerate the `hybrid_resnet` row under the locked contract.
- Run both local-baseline rows for the CDI `cnn` U-Net-class architecture under
  the same dataset, split, metrics, training budget, and output schema:
  - CDI `cnn` U-Net-class + supervised
  - CDI `cnn` U-Net-class + PINN
- Align this baseline family with the PDEBench `unet_strong` CNS row for paper
  comparison purposes, but do not imply the implementations are identical.
  Record row labels, architecture IDs, parameter counts, and training procedure
  so readers can see the CDI `cnn` generator and PDEBench `unet_strong` are
  U-Net/CNN-style local baselines with different task-local implementations.
- Do not treat SRU-Net as interchangeable with either CDI `cnn` or PDEBench
  `unet_strong`; if SRU-Net is added later, it must be a separately labeled
  architecture row.
- Run the selected FNO comparator under the same dataset, split, metrics,
  training budget, and output schema.
- Emit table-ready JSON/CSV/TeX, source reconstruction arrays, fixed-sample
  amplitude/phase panels, error panels, FRC curves, and a durable summary under
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.
- Mark the subset `paper_grade` only if every row has complete
  invocation/config/git/environment/dataset/split/metric/visual provenance.

## Notes for Reviewer

- Do not wait for FFNO or `spectral_resnet_bottleneck_net`. This item is the
  minimum CDI claim gate; those rows are still required for the complete
  `lines128` benchmark.
- Do not reuse roots with unrecoverable contract or provenance gaps as
  paper-grade evidence. A root's original exploratory or decision-support label
  is not by itself disqualifying if the current audit proves the row contract is
  complete.
- Do not change the selected FNO comparator or seed policy after seeing
  metrics. Any change requires a checked-in pre-run decision and a rerun of all
  affected rows.
- If the CDI `cnn` U-Net-class architecture cannot be run under both supervised
  and PINN training procedures with the locked contract, block with a precise
  contract mismatch rather than silently dropping the local convolutional
  comparison.
