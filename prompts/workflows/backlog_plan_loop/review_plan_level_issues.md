Use the `Consumed Artifacts` section as the authoritative input list and read those files first.
Also read `state/backlog_item_path.txt` and the referenced backlog item.

Do a second-pass review of the plan itself (not code quality).
Focus only on blocking plan issues:
- scope mismatch between backlog item, selected plan, and executed work
- unclear/unsafe canonical promotion source or lane mixing (canonical vs control)
- missing baseline coverage for active evaluation/promotion contexts
- tie-break logic that conflicts with the declared primary objective
- missing guardrail for degenerate optimization evidence (for example uniformly poor quality such as `ssim < 0.5` across all tested conditions)
- missing unblock path after invalid evidence (revise plan, regenerate artifacts, rerun dependent scopes)
- missing escalation path when runtime evidence shows plan assumptions are wrong

Decision:
- `APPROVE` if no blocking plan issue is found
- `REVISE` if any blocking plan issue is found

Outputs:
1. Read destination path from `state/plan_review_path.txt`, then write a concise report with:
   - verdict
   - blocking issues with evidence paths
   - required plan edits
   - required plan-revision commit subject(s) using: `plan-revision(<backlog-item-stem>): <short reason>`
   - reruns required
2. Write `APPROVE` or `REVISE` to `state/plan_review_decision.txt`.

Constraints:
- Keep it concise and evidence-based.
- Do not modify `state/plan_review_path.txt`.
- Do not move backlog items.
