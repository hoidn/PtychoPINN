---
priority: 20
plan_path: docs/plans/legacy-blocked-plan.md
check_commands:
  - python -c "print('blocked-check')"
prerequisites:
  - phase-99-not-done
related_roadmap_phases:
  - phase-2-pdebench-blocked-work
---

# Backlog Item: Blocked Item

## Objective
- Represent a backlog item whose prerequisite is not yet satisfied.

## Scope
- Keep the item blocked until its prerequisite is complete.

## Notes for Reviewer
- The selector should not choose this item while the prerequisite is missing.
