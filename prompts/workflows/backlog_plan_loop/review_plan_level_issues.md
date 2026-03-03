Use the `Consumed Artifacts` section as the authoritative input list and read those files first.
Also read `state/backlog_item_path.txt` and the referenced backlog item.

Do a review of the plan validity as evidenced by the session log / transcript, which records
a recent attempt to implement / execute the plan.
Focus only on blocking plan bugs that caused executio issues:
- unclear/unsafe canonical promotion source or lane mixing (canonical vs control)
- missing baseline coverage for optimization runs
- confusing lack of dataset provenance documentation
- degenerate optimization evidence (for example uniformly poor quality such as `ssim < 0.5` across all tested conditions) indicating that a plan-defined metric or optimization target is invalid
- invalid evidence without a recovery path (regenerate artifacts, rerun dependent scopes)
- runtime evidence exposing incorrect plan assumptions 

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
- Because plan revisions are highly disruptive, any changes require high confidence and solid supporting evidence and / or causal attribution
- Do not modify `state/plan_review_path.txt`.
- Do not move backlog items.
