Re-run the selected backlog item's full plan after a `REVISE` decision.

Use the `Consumed Artifacts` section as the authoritative input list, and read those files before acting.

Do:
1. Execute the full plan again, incorporating required fixes from the code review report.
2. Keep changes focused on plan completion.
3. Do not fabricate or backfill run results.
4. Write `artifacts/work/latest-execution-session-log.md` with:
   - plan path used
   - fixes implemented
   - commands executed and outcomes
   - files changed
   - blockers or follow-ups
5. Write exactly this path to `state/execution_session_log_path.txt`:
   artifacts/work/latest-execution-session-log.md

Constraints:
- No unrelated refactors.
- No queue movement.
- No absolute paths.
