You are the fix pass after a failed implementation-vs-plan review.

Read these inputs first:
- state/plan_path.txt and referenced plan
- state/slice_brief_path.txt and referenced brief
- state/misalignment_report_path.txt and referenced review report
- state/check_log_path.txt and referenced check log

Required actions:
1. Fix only the issues identified in the latest review.
2. Keep changes limited to bringing the selected slice into alignment.
3. Keep the selected task status in the plan as `in_progress` unless all issues are fully addressed and ready for review.
4. Write a fix summary to `artifacts/fixes/latest-fix-log.md` including:
   - each issue fixed
   - file-level changes
   - remaining known issues (if any)
5. Write exactly this relative path to `state/fix_log_path.txt`:
   artifacts/fixes/latest-fix-log.md

Constraints:
- No unrelated refactors.
- No queue movement; workflow handles backlog transitions.
- Do not write absolute paths.
