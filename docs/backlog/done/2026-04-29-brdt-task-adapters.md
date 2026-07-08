---
priority: 122
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_born_rytov_dt_adapters.py
  - python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
prerequisites:
  - 2026-04-29-brdt-dataset-preflight
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - BRDT should reuse model bodies through task-specific adapters, not the CDI generator registry.
  - The adapter contract must distinguish model architecture from training procedure.
  - This item creates the runnable surface needed before the four-row preflight.
---

# Backlog Item: BRDT Task Adapters

## Objective

- Add the task-local loading, adapter, and training surfaces needed to run BRDT
  rows without changing the CDI/PtychoPINN generator contract.

## Scope

- Consume the BRDT dataset preflight output.
- Add task-local modules under `scripts/studies/born_rytov_dt/` or the closest
  established study surface.
- Implement or stub the dataset loader, collator, model adapter, loss wrapper,
  training entry point, and evaluation entry point needed for the first four
  rows.
- Support at minimum:
  - classical Born backpropagation as a non-neural reference path;
  - U-Net model body;
  - FNO vanilla model body;
  - SRU-Net or Hybrid-family model body.
- Keep row metadata split into `model`, `training`, `input_mode`, `dataset_id`,
  `operator_version`, and `row_status`.

## Notes for Reviewer

- Do not register BRDT as a normal CDI generator.
- Do not compare a direct-sinogram row against a Born-initialization-image row
  in the same table.
- Do not label supervised plus Born consistency as `PINN-only`.
