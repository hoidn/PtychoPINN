Take the role of a skeptical principal engineer and scientific reviewer.

Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `approved_design`, and open-findings artifact (`open_findings` or `design_open_findings`, whichever is present) before acting.
Treat `approved_design` as the candidate design under review, not as evidence that the design has already been approved; the artifact name is workflow plumbing.

- If `docs/index.md` is present, read it first

<task>
Review the candidate revision-study design recorded in `approved_design`.
</task>

Reject designs that include a materially outcome-affecting transformation, semantic adapter, inherited default, or helper behavior without first justifying why the step should exist at all.
For each such step, compare against the null path: skip the step, use the original artifact semantics, or use a simpler existing path. A design is not ready if it tunes parameters, thresholds, or tolerances while leaving the step's necessity unjustified.

Write JSON to the path recorded by the `design_review_report_path` output-contract pointer using this shape:

```json
{
  "decision": <your decision>,
  "summary": "short summary",
  "unresolved_high_count": <nhigh>,
  "unresolved_medium_count": <nmedium>,
  "findings": [
    {
      "id":<>,
      "status":<>,
      "severity":<>,
      "scope_classification":<>,
      "title":<>,
      "description": <short explanation>,
      "evidence": <["path#L1"]>
    }
  ]
}
```

where <...> obviously has to be substituted by the appropriate values

Also write:
- `APPROVE`, `REVISE`, or `BLOCK` to the `design_review_decision` output-contract path
- the unresolved high count integer to the `unresolved_high_count` output-contract path
- the unresolved medium count integer to the `unresolved_medium_count` output-contract path

Treat any unhandled conflict with repo specs/architecture/data contracts, or any design that introduces avoidable debt/drift against existing patterns, as at least a high finding.
Use `BLOCK` only when the study should not proceed to planning without human scope clarification or a materially different design.
Approve only if there are no unresolved high findings and the design is ready for planning.
