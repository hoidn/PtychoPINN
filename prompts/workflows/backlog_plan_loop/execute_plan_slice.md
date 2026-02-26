You are executing exactly one plan slice for the selected backlog item.

Read these inputs first:
- state/backlog_item_path.txt and referenced backlog item
- state/plan_path.txt and referenced plan
- state/slice_brief_path.txt and referenced slice brief

Required actions:
1. Implement only the selected slice from the brief.
2. Keep edits minimal and directly tied to that slice.
3. Update the selected task status in the plan to `in_progress` if work has started and is not yet review-approved.
4. Write an execution log to `artifacts/work/latest-execution-log.md` including:
   - files changed
   - summary of implemented behavior
   - unresolved risks or follow-ups
5. Write exactly this relative path to `state/execution_log_path.txt`:
   artifacts/work/latest-execution-log.md

Constraints:
- Do not execute full test suites.
- Do not perform unrelated refactors.
- Do not modify backlog queue placement.
- Do not write absolute paths.
