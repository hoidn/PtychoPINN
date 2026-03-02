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

<long-running-work-policy>
  This policy applies to any required long-running process in the selected plan.

  1. Treat required long-running work as blocking.
  2. If required long-running process is not running, launch it and wait for completion.
  3. If it is already running, poll it every 30s until completion or timeout. For very long-running processes, use a backoff schedule.
  4. Record PID, start time, current status, and final exit code 
  5. Do not stop polling while any required long-running process is still running.
  6. After process exit, verify required completion evidence from the plan:
     - required END/RC markers (if used),
     - required output artifacts exist and are fresh for this run,
     - required verification checks pass.

  Default timeout: 12 hours unless the plan defines a different limit.
</long-running-work-policy>
