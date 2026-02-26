Read the implementation plan of the currently in-progress backlog item and verify the correctness and completion state of the implementation with respect to the plan. Use the superpowers:code-reviewer skill. 

Use the `Consumed Artifacts` section as the authoritative input list, and read those files before writing your decision.
Also read `state/failed_count.txt`, and if `state/check_log_path.txt` exists, read the referenced check log.

After finishing your review of the implementation, write:
1. `artifacts/review/latest-review.md` with:
   - alignment verdict
   - findings
   - required fixes
   - final decision
2. `state/review_decision.txt`:
   - `APPROVE` only if plan is implemented, no blocking issues remain, and `state/failed_count.txt` reports `0`
   - otherwise `REVISE`
3. `state/issue_count.txt` as the number of blocking issues.
4. `state/code_review_path.txt` with:
   artifacts/review/latest-review.md

Constraints:
- Focus on correctness and plan alignment.
- Do not block on style-only issues.
- If required checks failed, decision must be `REVISE`.
