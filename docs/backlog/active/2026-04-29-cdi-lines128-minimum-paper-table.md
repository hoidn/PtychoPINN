---
priority: 21
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md
check_commands:
  - pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-harness
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The paper can start drafting result shells once the minimum CDI subset has paper-grade hybrid_resnet, U-Net/SRU-Net + PINN local-baseline, and FNO rows under one Lines128 contract.
  - Spectral_resnet_bottleneck_net and FFNO remain required for the complete Lines128 benchmark, but they should not block the first minimum CDI claim subset if the core rows are ready.
---

# Backlog Item: Run Minimum Lines128 Paper CDI Table

## Objective

- Produce the minimum paper-grade Lines128 CDI result subset and visual bundle
  for `hybrid_resnet`, a U-Net/SRU-Net + PINN local convolutional baseline,
  and the selected FNO comparator under one locked contract.

## Scope

- Use the paper benchmark harness preflight artifact as the source of truth for
  dataset, split, probe preprocessing, seed policy, training budget, metric
  schema, sample IDs, and visual scales.
- Run or regenerate the `hybrid_resnet` row under the locked contract.
- Run the U-Net/SRU-Net + PINN local convolutional baseline and selected FNO
  comparator under the same dataset, split, metrics, training budget, and
  output schema.
- Use a simpler CNN/PINN row only as an explicit fallback if U-Net/SRU-Net is
  not protocol-compatible with the locked Lines128 runner contract; record the
  exact compatibility blocker before falling back.
- Emit table-ready JSON/CSV/TeX, source reconstruction arrays, fixed-sample
  amplitude/phase panels, error panels, FRC curves, and a durable summary under
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.
- Mark the subset `paper_grade` only if every row has complete
  invocation/config/git/environment/dataset/split/metric/visual provenance.

## Notes for Reviewer

- Do not wait for FFNO or `spectral_resnet_bottleneck_net`. This item is the
  minimum CDI claim gate; those rows are still required for the complete
  `lines128` benchmark.
- Do not reuse historical incomplete roots as paper-grade evidence.
- Do not change the selected FNO comparator or seed policy after seeing
  metrics. Any change requires a checked-in pre-run decision and a rerun of all
  affected rows.
- If neither U-Net/SRU-Net + PINN nor the fallback CNN/PINN row is
  protocol-compatible, block with a precise contract mismatch rather than
  silently dropping the local convolutional baseline.
