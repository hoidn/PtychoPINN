# Workflow Backlog Item Template

Use this template for items queued in `docs/backlog/active/` and executed by:
`workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`

```md
---
priority: 10
plan_path: docs/plans/<YYYY-MM-DD-initiative-plan>.md
check_commands:
  - <targeted check command 1>
  - <targeted check command 2>
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

## Queue Placement

- `docs/backlog/active/`: eligible for workflow selection
- `docs/backlog/paused/`: excluded from selection
- `docs/backlog/done/`: completed items

The workflow moves approved items from `active` to `done` automatically.
