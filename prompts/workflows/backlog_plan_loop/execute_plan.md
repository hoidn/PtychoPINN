Implement the selected backlog item's full plan.

Use the `Consumed Artifacts` section as the authoritative input list, and read those files before acting.
Also read `state/backlog_item_path.txt` and the referenced backlog item for scope context.

Do:
1. Execute the plan end-to-end.
2. Keep changes focused on the plan/backlog scope.
3. Do not fabricate or backfill run results.
4. Write `artifacts/work/latest-execution-log.md` with:
   - files changed
   - what is complete
   - what is incomplete
   - blockers
5. Write exactly this path to `state/execution_log_path.txt`:
   artifacts/work/latest-execution-log.md

Constraints:
- No unrelated refactors.
- No absolute paths.
