use receiving-code-review to address the feedback

Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `brief`, `design`, and `design_review_report` artifacts before acting.

Revise the design in place to address every unresolved blocking or in-scope finding.
Do not ignore a blocking prerequisite because it is expensive.
Do not spend time on `recommended_followup` or `out_of_scope` findings unless they are needed to resolve a blocking issue cleanly.

For the output contract's `design_path`, read the path recorded in that file and write the updated design document to that current-checkout-relative path. Leave the `design_path` file containing only the path.
