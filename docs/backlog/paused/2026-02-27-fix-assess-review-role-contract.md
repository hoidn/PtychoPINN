# Backlog: Declarative Fix/Assess/Review Role Contract Validation

**Created:** 2026-02-27
**Status:** Paused (Later)
**Priority:** Medium
**Related:** `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`, `/home/ollie/Documents/agent-orchestration/specs/dsl.md`

## Summary

Add optional DSL semantics for workflow step roles so loader-time graph validation can reject workflows where a successful fix step can terminate before reassessment and review.

## Problem

Workflow correctness currently depends on transition wiring conventions. If a workflow routes `FixIssues` to a terminal path before reassess/review, the latest fixes can be unreviewed.

## Proposed Direction

1. Add optional step role metadata (example: `fix`, `assess`, `review`, `terminal`).
2. Add optional workflow-level invariants (example: `after_fix_requires: [assess, review]`).
3. In loader validation, analyze success paths and reject graphs where any `fix` step can reach terminal without required roles in order.
4. Keep this opt-in for backward compatibility.

## Acceptance Criteria

1. A workflow declaring `after_fix_requires` fails validation when a violating path exists.
2. A compliant workflow passes and can execute unchanged.
3. Unit tests cover positive/negative graph-shape cases.
4. Specs document semantics and migration guidance.
