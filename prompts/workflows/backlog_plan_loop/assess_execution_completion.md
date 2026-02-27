Read the `Consumed Artifacts` section and treat it as the authoritative input list.
Read those files before writing your assessment.

Also read:
- `state/backlog_item_path.txt` and the referenced backlog item for scope context.

Evaluate completion from the execution session log and plan.

<completion criteria>
`COMPLETE` only if every in-scope plan step is either:
- `COMPLETED`, or
- `BLOCKED` with concrete blocker details and evidence of attempted execution.
Otherwise mark `INCOMPLETE`.
</completion criteria>

<output format>
After execution:
1. Write `artifacts/work/latest-execution-log.md` containing:
   - overall status: `COMPLETE | INCOMPLETE | BLOCKED`
   - plan path used
   - step-by-step results (`COMPLETED | BLOCKED | NOT_RUN`) with evidence
   - files changed
   - verification commands and outcomes
   - incomplete items
   - blockers
2. Write exactly this path to `state/execution_log_path.txt`:
   artifacts/work/latest-execution-log.md
</output format>
