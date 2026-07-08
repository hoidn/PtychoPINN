Use the injected redesign brief, design note, implementation plan, execution session log, check log, and candidate metadata as the authoritative review inputs.

Task:
- Review whether this redesign candidate package is internally coherent and ready for the outer controller to smoke/score.

Required outputs:
- Write the review report to the `review_log_path` named in the redesign brief.
- Write one line to the `review_decision_path` named in the redesign brief:
  - `APPROVE`
  - `REVISE`
  - `BLOCK`

Decision rubric:
- `APPROVE` only if the candidate package is complete, coherent, and ready for controller-owned smoke/scored evaluation.
- `REVISE` if the candidate looks promising but the implementation package is incomplete, inconsistent, or under-verified.
- `BLOCK` if the redesign is unsound or the candidate package cannot be made valid without rethinking the idea itself.

The review report must include:
- candidate-package verdict
- concrete findings
- required fixes if decision is `REVISE`
- blocker rationale if decision is `BLOCK`

Constraints:
- Focus on candidate-package correctness and readiness, not style-only issues.
- Do not move queue items or edit controller-owned state.
