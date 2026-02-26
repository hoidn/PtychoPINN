<inputs>
Use the `Consumed Artifacts` section as the authoritative input list, and read those files before acting.
Also read `state/backlog_item_path.txt` and the referenced backlog item for scope context.
</inputs>

<task>
Implement the selected backlog item's full plan using the executing-plans superpower / skill
</task>

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


