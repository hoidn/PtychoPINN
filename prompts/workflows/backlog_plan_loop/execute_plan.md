<inputs>
Use the `Consumed Artifacts` section as the authoritative input list, and read those files before acting.
Also read `state/backlog_item_path.txt` and the referenced backlog item for scope context.
</inputs>

<task>
Implement the selected backlog item's full plan using the executing-plans superpower / skill
</task>

<output format>
After execution:
1. Read destination path from `state/execution_session_log_path.txt`, then write the execution session log to that path.
2. The session log must contain:
   - plan path used
   - commands executed and outcomes
   - files changed
   - what was completed
   - blockers or follow-ups
</output format>

<constraints>
- Keep changes focused on plan/backlog scope.
- Do not fabricate or backfill run results.
- No unrelated refactors.
- Do not modify `state/execution_session_log_path.txt` in this step.
- Do not create or use git worktrees; execute in the current checked-out workspace and write artifacts in that same workspace.
</constraints>
