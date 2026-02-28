Use the `Consumed Artifacts` section as the authoritative input list, and read those files before writing your decision.

Read the implementation plan of the currently in-progress backlog item and verify the correctness and completion state of the implementation with respect to the plan. Use the superpowers:code-reviewer skill.

After finishing the above code-focused review, do a second review pass from the perspective of which data, study runs, artifacts, and/or non-code edits should have been created by the plan's complete execution. Check exhaustively: the existence of *some* expected outputs does not guarantee that the plan was successfully and fully executed.

Interpretation rule for execution evidence:
- Treat concrete evidence from canonical artifacts (files changed, generated reports, run outputs, and check-log outcomes) as primary.
- Missing narrative prose in the execution session log is non-blocking if the concrete artifacts and check results prove completion.
- If required checks failed, required artifacts are missing, or the outputs materially diverge from the plan, decision must be `REVISE`.

If either review exposes bugs (beyond straightforward plan misalignment), clarify root cause(s). Use the systematic debugging superpower if appropriate.

After finishing your review of the implementation, write:
1. Read destination path from `state/code_review_path.txt`, then write the review report to that path, with:
   - alignment verdict
   - findings
   - root cause(s), if applicable
   - required fixes of the plan misalignments and root causes, if applicable
   - final decision
2. `state/review_decision.txt`:
   - `APPROVE` only if plan is implemented and no blocking issues remain
   - otherwise `REVISE`

Constraints:
- Focus on correctness and plan alignment.
- Do not block on style-only issues.
- If required checks failed, decision must be `REVISE`.
- Do not modify `state/code_review_path.txt` in this step.
