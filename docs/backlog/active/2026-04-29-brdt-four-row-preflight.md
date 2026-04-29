---
priority: 123
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_born_rytov_dt_preflight.py
  - python -m compileall -q scripts/studies/born_rytov_dt
prerequisites:
  - 2026-04-29-brdt-task-adapters
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - The first BRDT run should be a bounded four-row decision-support preflight, not a broad benchmark suite.
  - Classical, U-Net, FNO vanilla, and SRU/Hybrid rows are enough to test whether the lane is worth promotion.
  - This item should emit metrics, visuals, manifests, and explicit claim boundaries.
---

# Backlog Item: BRDT Four-Row Preflight

## Objective

- Run the bounded BRDT decision-support preflight under one dataset, operator,
  input, split, metric, and training contract.

## Scope

- Consume the validated BRDT operator, dataset, and task adapters.
- Run exactly the first preflight roster unless the plan records a narrow
  blocker:
  - classical Born backpropagation;
  - U-Net with supervised plus Born consistency;
  - FNO vanilla with supervised plus Born consistency;
  - SRU-Net or Hybrid-family model with supervised plus Born consistency.
- Report image-space metrics on physical `q`, measurement-space residuals,
  parameter counts for neural rows, runtime/hardware, and row statuses.
- Emit fixed-sample visuals, source arrays, table JSON/CSV, metric schema, and
  a manifest sufficient to regenerate the bundle.

## Notes for Reviewer

- Do not add Rytov, limited-angle, FFNO, physics-only, external FDTD mismatch,
  or multi-seed rows in this item.
- Do not call this paper-grade evidence.
- Do not mix input representations, splits, or normalization policies across
  rows.
