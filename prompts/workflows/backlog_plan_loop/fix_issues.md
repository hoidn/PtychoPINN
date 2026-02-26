You are the fix pass after a failed full-plan review.

Read these inputs first:
- state/plan_path.txt and referenced plan
- state/misalignment_report_path.txt and referenced review report
- state/check_log_path.txt and referenced check log

Required actions:
1. Fix all issues identified in the latest review report.
2. Keep changes focused on reaching full plan completion.
3. Write a fix summary to `artifacts/fixes/latest-fix-log.md` including:
   - each issue fixed
   - file-level changes
   - any remaining blockers
4. Write exactly this relative path to `state/fix_log_path.txt`:
   artifacts/fixes/latest-fix-log.md

Constraints:
- No unrelated refactors.
- No queue movement; workflow handles backlog transitions.
- Do not write absolute paths.
