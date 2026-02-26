Use the `Consumed Artifacts` section as the authoritative input list, and read those files before writing your decision.

Read the implementation plan of the currently in-progress backlog item and verify the correctness and completion state of the implementation with respect to the plan. Use the superpowers:code-reviewer skill. 

After finishing the above code-focued review, do a second review pass from the perspective of which data, study runs, artifacts, and / or non-code edits should have been created by the plan's complete execution. Check exhaustively: the existence of *some* expected outputs does not guarantee that the plan was successfully and fully executed. 


After finishing your review of the implementation, write:
1. `artifacts/review/latest-review.md` with:
   - alignment verdict
   - findings
   - required fixes
   - final decision
2. `state/review_decision.txt`:
   - `APPROVE` only if plan is implemented and no blocking issues remain
   - otherwise `REVISE`
3. `state/issue_count.txt` as the number of blocking issues.
4. `state/code_review_path.txt` with:
   artifacts/review/latest-review.md

Constraints:
- Focus on correctness and plan alignment.
- Do not block on style-only issues.
- If required checks failed, decision must be `REVISE`.
