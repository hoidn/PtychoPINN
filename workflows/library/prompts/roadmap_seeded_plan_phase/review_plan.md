Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design`, `roadmap`, `tranche_context`, `plan`, and `open_findings` artifacts before acting.

First, review the current plan from scratch.
Check that the plan faithfully carries the consumed design, consumed roadmap, and selected tranche context into executable work: material requirements should appear as concrete tasks with proportionate verification, or be explicitly identified as outside this plan's scope with the reason.
Reject plans that expand beyond the selected tranche without a roadmap gate or dependency reason.
Reject plans that collapse the design's component boundaries, interfaces, invariants, roadmap phase order, gates, fallback decisions, or durable artifact contracts into undifferentiated implementation work instead of assigning tasks and tests along those boundaries.
Reject plans for work with multiple meaningful responsibilities, future dependents, cross-boundary behavior, or meaningful review or verification risk if they do not define implementable units, owned boundaries, dependency direction, and focused tests, unless the design explicitly justifies a small single-unit implementation.
Reject plans that blur hand-edited source files and generated outputs, omit concrete generation or validation checks for generated artifacts, or introduce reusable or large helper scripts without tests or a maintainability rationale.
Then reconcile your fresh review against the carried-forward `open_findings` ledger.

For each prior finding, classify it as one of:
- `RESOLVED`
- `STILL_OPEN`
- `SUPERSEDED`
- `SPLIT`

You may add `NEW` findings only if they are materially distinct.
Do not preserve a finding only because it existed before.

For the output contract's `plan_review_report_path`, read the path recorded in that file and write JSON to that current-checkout-relative path using this shape. Leave the `plan_review_report_path` file containing only the path.

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
- `APPROVE` or `REVISE` to the `plan_review_decision` path specified in the Output Contract
- the unresolved high count integer to the `unresolved_high_count` path specified in the Output Contract
- the unresolved medium count integer to the `unresolved_medium_count` path specified in the Output Contract

Approve only if there are no unresolved high findings and the plan is ready to execute without inventing architecture, reordering the roadmap unsafely, or silently dropping material design or roadmap requirements.
