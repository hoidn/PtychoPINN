take the role of a principal engineer, expert in PLs, compilers, and agentic engineering. review the design / ADR with no holds barred skepticism

Use the `Consumed Artifacts` section as the authoritative input list.

First, review the current design from scratch.
Then reconcile your fresh review against the carried-forward `open_findings` ledger.

You may require internal refactoring or egregious debt paydown before feature work only when it is:
- a correctness prerequisite
- a contract prerequisite
- or a major simplicity win that materially reduces feature risk

Reject designs that include a materially outcome-affecting transformation, semantic adapter, inherited default, or helper behavior without first justifying why the step should exist at all.
For each such step, compare against the null path: skip the step, use the original artifact semantics, or use a simpler existing path. A design is not ready if it tunes parameters, thresholds, or tolerances while leaving the step's necessity unjustified.

For each prior finding, classify it as one of:
- `RESOLVED`
- `STILL_OPEN`
- `SUPERSEDED`
- `SPLIT`

You may add `NEW` findings only if they are materially distinct.
Do not preserve a finding only because it existed before.

Each finding must include a `scope_classification` of:
- `blocking_prerequisite`
- `required_in_scope`
- `recommended_followup`
- `out_of_scope`

For the output contract's `design_review_report_path`, read the path recorded in that file and write JSON to that current-checkout-relative path using this shape. Leave the `design_review_report_path` file containing only the path.

```json
{
  "decision": "APPROVE",
  "summary": "short summary",
  "unresolved_high_count": 0,
  "unresolved_medium_count": 0,
  "findings": [
    {
      "id": "DESIGN-H1",
      "status": "RESOLVED",
      "severity": "high",
      "scope_classification": "blocking_prerequisite",
      "title": "short title",
      "description": "short explanation",
      "evidence": ["path#L1"]
    }
  ]
}
```

Also write:
- `APPROVE`, `REVISE`, or `BLOCK` to the `design_review_decision` path specified in the Output Contract
- the unresolved high count integer to the `unresolved_high_count` path specified in the Output Contract
- the unresolved medium count integer to the `unresolved_medium_count` path specified in the Output Contract

Use `REVISE`, not `BLOCK`, when the design can be made acceptable by editing the design document itself, even if the required change is major.
Use `BLOCK` only when progress requires work outside the design artifact, such as an upstream refactor, missing or contradictory task authority, unavailable prerequisites, or a human scope decision.
Approve only if there are no unresolved high findings and the design is ready for planning.
