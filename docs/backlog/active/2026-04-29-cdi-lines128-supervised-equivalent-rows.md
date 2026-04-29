---
priority: 23
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md
check_commands:
  - pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The main Lines128 CDI table should first lock the physics-informed rows and shared contract before supervised equivalents are launched.
  - The minimum CDI table already owns the CDI `cnn` U-Net-class supervised versus PINN pair; this extension owns the required supervised FFNO control row.
---

# Backlog Item: Add Required Lines128 Supervised FFNO Control Row

## Objective

- Extend the fixed Lines128 CDI evidence set with a required supervised FFNO
  control row after the CDI `cnn` U-Net-class supervised versus PINN pair and
  the complete PINN-trained Lines128 table are locked.

## Scope

- Reuse the final locked Lines128 dataset, split, probe preprocessing,
  normalization, sample IDs, metric schema, and visual scales from the complete
  paper-quality CDI execution.
- Do not rerun the CDI `cnn` U-Net-class supervised row here; it belongs to the
  minimum CDI table alongside the matching CDI `cnn` U-Net-class + PINN row.
- Run the FFNO + supervised row under the same locked Lines128 contract as the
  complete FFNO + PINN row. This row is required; do not substitute supervised
  FNO for it.
- If the supervised FFNO training path is not protocol-compatible with the
  locked Lines128 contract, record a precise `not_protocol_compatible` outcome
  with the missing interface or artifact contract rather than silently dropping
  the row or replacing it with a different architecture.
- Label rows by both architecture and training procedure, for example
  `FFNO + supervised` versus `FFNO + PINN`, so result tables do not conflate the
  model body with the training objective.
- Emit table-ready JSON/CSV/TeX fragments, provenance manifests, source
  reconstruction arrays, fixed-sample amplitude/phase panels, error panels, and
  a concise summary under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.
- Keep the extension separate from the primary CDI claim unless every included
  supervised row is run under the same locked contract and passes the same
  metric and provenance checks.

## Notes for Reviewer

- Do not change the Lines128 contract after seeing supervised-row metrics. If a
  supervised trainer requires a contract change, record a pre-run decision and
  rerun every affected comparator.
- Do not compare `model + supervised` rows against `model + PINN` rows without
  making the training-procedure distinction explicit in table labels and prose.
- Do not reuse historical supervised outputs as paper-grade evidence unless
  they exactly match the locked Lines128 contract and have complete provenance.
- If FFNO has no supervised training path under the locked contract, record
  `not_protocol_compatible` and do not invent a broader supervised sweep as a
  substitute.
