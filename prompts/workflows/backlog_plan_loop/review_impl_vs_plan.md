Read the implmentation plan of the currently in-progress backlog item and verify the correctness and completion state of the implementation with respect to the plan.

After finishing your review of the implementation, write:
1. `artifacts/review/latest-review.md` with:
   - alignment verdict
   - findings
   - required fixes
   - final decision
2. `state/review_decision.txt`:
   - `APPROVE` only if plan is implemented, no blocking issues remain, and failed_count is 0
   - otherwise `REVISE`
3. `state/issue_count.txt` as the number of blocking issues.
4. `state/misalignment_report_path.txt` with:
   artifacts/review/latest-review.md

Constraints:
- Focus on correctness and plan alignment.
- Do not block on style-only issues.
- If required checks failed, decision must be `REVISE`.
