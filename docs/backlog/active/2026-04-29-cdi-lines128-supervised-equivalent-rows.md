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
  - The supervised extension exists to separate architecture effects from the PINN forward-model training procedure.
---

# Backlog Item: Add Lines128 Supervised CDI Equivalent Rows

## Objective

- Extend the fixed Lines128 CDI evidence set with supervised-training
  equivalents for CNN, FNO, SCR, and FFNO so the paper can distinguish
  architecture effects from the PINN forward-model training procedure.

## Scope

- Reuse the final locked Lines128 dataset, split, probe preprocessing,
  normalization, sample IDs, metric schema, and visual scales from the complete
  paper-quality CDI execution.
- Add supervised rows for the same architecture families represented by the
  physics-informed CDI table: CNN, FNO, SCR, and FFNO.
- Label rows by both architecture and training procedure, for example
  `SCR + supervised` versus `SCR + PINN`, so result tables do not conflate the
  model body with the training objective.
- Emit table-ready JSON/CSV/TeX fragments, provenance manifests, source
  reconstruction arrays, fixed-sample amplitude/phase panels, error panels, and
  a concise summary under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.
- Keep the extension separate from the primary CDI claim unless every
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
- If one architecture cannot be trained supervised under the locked contract,
  report that row as `not_protocol_compatible` with the exact interface or data
  contract mismatch.
