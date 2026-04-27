use receiving-code-review to address the feedback

Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design`, `plan`, and `plan_review_report` artifacts before acting.

Revise the plan in place to address every unresolved or new in-scope finding.
Keep scope, order, and ownership coherent.

When revising a plan, preserve or add an Implementation Architecture section if correctness or maintainability depends on a boundary decision: component or file ownership, API or command surface, data or artifact contract, authored-vs-derived split, dependency direction, compatibility or migration boundary, or future consumer contract. If no such boundary decision is needed, state why the plan remains a single implementation unit.

Do not resolve findings by blindly broadening or narrowing scope. Preserve the intended deliverable from the consumed design and any consumed brief, roadmap, or selection context. If the plan is over-broad, sequence or slice the work with an explicit rationale. If the plan is under-scoped, bring material design requirements back into current work unless the plan records clear authority, rationale, and handoff criteria for deferring them. Follow-up work should name the deferred requirement, not hide it behind a generic bucket.

For the output contract's `plan_path`, read the path recorded in that file and write the updated plan document to that current-checkout-relative path. Leave the `plan_path` file containing only the path.
