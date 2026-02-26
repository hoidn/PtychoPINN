You are executing the selected backlog item's full implementation plan end-to-end.

Read these inputs first:
- state/backlog_item_path.txt and referenced backlog item
- state/plan_path.txt and referenced plan

Required actions:
1. Implement the full plan (not a single slice).
2. Keep changes focused on the plan scope and backlog objective.
3. Do not mark the backlog item done; workflow handles queue transitions.
4. Write an execution log to `artifacts/work/latest-execution-log.md` including:
   - major files changed
   - what was fully implemented
   - what remains incomplete, if anything
   - notable risks/assumptions
5. Write exactly this relative path to `state/execution_log_path.txt`:
   artifacts/work/latest-execution-log.md

Constraints:
- Do not write absolute paths.
- Do not perform unrelated refactors.
- Keep implementation and docs aligned with the plan contract.
