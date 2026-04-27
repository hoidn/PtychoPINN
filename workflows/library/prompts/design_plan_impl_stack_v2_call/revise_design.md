use receiving-code-review to address the feedback

Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `brief`, `design`, and `design_review_report` artifacts before acting.

Revise the design in place to address every unresolved blocking or in-scope finding.
Do not ignore a blocking prerequisite because it is expensive.
Do not spend time on `recommended_followup` or `out_of_scope` findings unless they are needed to resolve a blocking issue cleanly.

Write the updated design to the `design_path` path specified in the Output Contract.
