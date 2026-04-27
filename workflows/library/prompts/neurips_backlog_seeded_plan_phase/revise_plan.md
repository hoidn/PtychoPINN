Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `steering`, `design`, `roadmap`, `selected_item_context`, `progress_ledger`, `plan`, and `plan_review_report` artifacts before acting.
Treat `selected_item_context` as the authoritative backlog-item content and queue location; use any `selection_source_path` there only as provenance.

Revise the plan to resolve the review findings while preserving the steering constraints, approved design, roadmap order and gates, and selected backlog scope.

Do not silently change the selected backlog scope or roadmap order. If a finding cannot be resolved without changing them, make that blocker explicit in the plan instead of quietly widening or redirecting the work.

Preserve the plan's self-contained execution role: after revision, implementation should still be able to execute from the approved design and approved plan without rediscovering scope from the raw backlog item.

Keep the backlog item's required `check_commands` unless the finding requires a documented replacement.

For the output contract's `plan_path`, read the path recorded in that file and update the plan document there. Leave the `plan_path` file containing only the path.
