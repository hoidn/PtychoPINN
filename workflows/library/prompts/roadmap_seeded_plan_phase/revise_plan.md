Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design`, `roadmap`, `tranche_context`, `plan`, and `plan_review_report` artifacts before acting.

Revise the plan to resolve the review findings while preserving the approved design, roadmap, and selected tranche context.

Do not change the approved design or roadmap. If a finding cannot be resolved without changing them, mark that clearly in the plan as a blocker or required human decision instead of silently changing scope.

For each unresolved review finding:
- address it directly in the plan
- preserve or improve the roadmap phase order and gates
- add concrete verification where the finding identifies weak evidence
- avoid broad refactors or workflow changes unless the finding requires them

For the output contract's `plan_path`, read the path recorded in that file and update the plan document at that current-checkout-relative path. Leave the `plan_path` file containing only the path.
