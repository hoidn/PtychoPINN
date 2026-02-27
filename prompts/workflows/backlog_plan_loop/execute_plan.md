<inputs>
Use the `Consumed Artifacts` section as the authoritative input list, and read those files before acting.
Also read `state/backlog_item_path.txt` and the referenced backlog item for scope context.
</inputs>

<task>
Implement the selected backlog item's full plan using the executing-plans superpower / skill
</task>

<output format>
After execution:
1. Write `artifacts/work/latest-execution-session-log.md` containing:
   - plan path used
   - commands executed and outcomes
   - files changed
   - what was completed
   - blockers or follow-ups
2. Write exactly this path to `state/execution_session_log_path.txt`:
   artifacts/work/latest-execution-session-log.md
</output format>

<constraints>
- Keep changes focused on plan/backlog scope.
- Do not fabricate or backfill run results.
- No unrelated refactors.
</constraints>
