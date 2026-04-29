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
  - The supervised extension exists to separate architecture effects from the PINN forward-model training procedure without requiring supervised counterparts for every architecture family.
---

# Backlog Item: Add Targeted Lines128 Supervised CDI Control Rows

## Objective

- Extend the fixed Lines128 CDI evidence set with targeted supervised-training
  control rows, anchored by the local U-Net/SRU-Net architecture, so the paper
  can distinguish model-body effects from the PINN forward-model training
  procedure without running supervised counterparts for every architecture.

## Scope

- Reuse the final locked Lines128 dataset, split, probe preprocessing,
  normalization, sample IDs, metric schema, and visual scales from the complete
  paper-quality CDI execution.
- Add a supervised row for the same local convolutional architecture used by
  the minimum CDI table, preferably `U-Net/SRU-Net + supervised` paired against
  `U-Net/SRU-Net + PINN`.
- Add at most one additional supervised comparator, chosen before launch, only
  if it answers a concrete paper question:
  - selected FNO + supervised, if the paper needs a spectral-operator
    training-procedure control
  - FFNO + supervised, if the FFNO generator path is mature and the paper needs
    a Fourier-factorized training-procedure control
- Label rows by both architecture and training procedure, for example
  `SRU-Net + supervised` versus `SRU-Net + PINN`, so result tables do not conflate the
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
- If the local U-Net/SRU-Net supervised row cannot be trained under the locked
  contract, report it as `not_protocol_compatible` with the exact interface or
  data-contract mismatch and do not expand to unrelated supervised rows as a
  substitute.
