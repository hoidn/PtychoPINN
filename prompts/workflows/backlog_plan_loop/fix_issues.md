Re-run the selected backlog item's full plan after a `REVISE` decision.

Use the `Consumed Artifacts` section as the authoritative input list, and read those files before acting.

Do:
1. Execute the full plan again, incorporating required fixes from the code review report.
2. Use the consumed check log to prioritize concrete failing checks first.
3. Keep changes focused on plan completion.
4. Do not fabricate or backfill run results.
5. Read destination path from `state/execution_session_log_path.txt`, then write the execution session log to that path, with:
   - plan path used
   - fixes implemented
   - commands executed and outcomes
   - files changed
   - blockers or follow-ups

Constraints:
- No unrelated refactors.
- No queue movement.
- No absolute paths.
- Do not modify `state/execution_session_log_path.txt` in this step.
- Do not create or use git worktrees; execute in the current checked-out workspace and write artifacts in that same workspace.
