Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `steering`, `design`, `roadmap`, `selected_item_context`, `progress_ledger`, `plan`, and `open_findings` artifacts before acting.
Treat `selected_item_context` as the authoritative backlog-item content and queue location; use any `selection_source_path` there only as provenance.

First, review the current plan from scratch.

Check that the plan faithfully carries the steering constraints, design requirements, roadmap order and gates, selected backlog scope, prerequisite status, and deterministic `check_commands` into executable work.

Reject plans that:
- expand beyond the selected backlog item without a roadmap or dependency reason
- assume later implementation must reread the raw backlog item, steering document, or roadmap to discover scope, claim boundaries, prerequisites, or required checks
- weaken or drop the backlog item's required `check_commands` without an explicit rationale and replacement
- collapse multiple meaningful responsibilities or durable boundaries into vague shared work
- blur maintained source files and generated outputs or omit concrete validation for generated artifacts
- use a non-terminal state as the normal way to stop after launching training, benchmarks, or data generation
- convert ordinary failing tests, import errors, path issues, environment
  propagation issues, or test-harness failures directly into item-level
  `BLOCKED`. A failed check may gate an expensive later step, but the plan must
  require diagnose/fix/rerun before allowing `BLOCKED`, unless the blocker is a
  missing resource, unavailable hardware, roadmap conflict, external dependency
  outside current authority, user decision required, or unrecoverable after a
  documented narrow fix attempt.

Then reconcile your fresh review against the carried-forward `open_findings` ledger.

For each prior finding, classify it as one of:
- `RESOLVED`
- `STILL_OPEN`
- `SUPERSEDED`
- `SPLIT`

You may add `NEW` findings only if they are materially distinct.

For the output contract's `plan_review_report_path`, read the path recorded in that file and write JSON there using this shape:

```json
{
  "decision": "APPROVE",
  "summary": "short summary",
  "unresolved_high_count": 0,
  "unresolved_medium_count": 0,
  "findings": [
    {
      "id": "PLAN-H1",
      "status": "RESOLVED",
      "severity": "high",
      "title": "short title",
      "description": "short explanation",
      "evidence": ["path#L1"]
    }
  ]
}
```

Also write:
- `APPROVE` or `REVISE` to the `plan_review_decision` path
- the unresolved high count integer to the `unresolved_high_count` path
- the unresolved medium count integer to the `unresolved_medium_count` path

Approve only if there are no unresolved high findings and the plan is ready to execute without inventing missing context during implementation.
