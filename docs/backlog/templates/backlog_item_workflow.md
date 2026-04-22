# Workflow Backlog Item Template

Use this template for items queued in `docs/backlog/active/` and executed by:
`workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`
or `workflows/examples/neurips_steered_backlog_drain.yaml`

```md
---
priority: 10
plan_path: docs/plans/<YYYY-MM-DD-initiative-plan>.md
check_commands:
  - <targeted check command 1>
  - <targeted check command 2>
prerequisites:
  - <optional prerequisite id>
related_roadmap_phases:
  - <optional roadmap phase id>
---

# Backlog Item: <Short Title>

## Objective
- <one sentence on the work outcome>

## Scope
- <what is in scope>
- <what is out of scope>

## Notes for Reviewer
- <key risks, assumptions, or constraints>
```

## Required Frontmatter Fields

- `priority`:
  - integer
  - lower value runs earlier
- `plan_path`:
  - repository-relative path to an existing plan file under `docs/plans/`
- `check_commands`:
  - non-empty YAML list of targeted commands
  - keep fast; avoid full test-suite commands unless necessary
- `prerequisites`:
  - optional YAML list of prerequisite ids
  - used by steering-aware selection to keep blocked items out of the runnable queue
- `related_roadmap_phases`:
  - optional YAML list of roadmap phase or tranche ids
  - helps the selector and reviewers keep backlog work aligned with roadmap order

## Queue Placement

- `docs/backlog/active/`: eligible for workflow selection
- `docs/backlog/in_progress/`: selected by the NeurIPS drain and owned by the active run until approval, resume, or explicit reset
- `docs/backlog/paused/`: excluded from selection
- `docs/backlog/done/`: completed items

The workflows move approved items from `active` to `done` automatically.
For the NeurIPS drain, `plan_path` may be rewritten to the fresh approved plan before implementation begins, so backlog items should treat the frontmatter `plan_path` as workflow-managed state after selection.
