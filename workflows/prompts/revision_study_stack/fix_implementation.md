Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `approved_design`, `revision_context`, `plan`, `execution_report`, and `implementation_review_report` artifacts before acting.

Use executing-plans to address the implementation review while staying aligned with the approved design and full plan.
Do not use `git worktree` or another checkout.
If the repo is dirty, stay in the current checkout and leave unrelated files alone.
Do not modify workflow YAML, prompt files, or runtime state files unless the plan explicitly requires it.
Do not edit the original revision design seed unless the approved plan explicitly requires it.
Do not commit unless the approved plan explicitly requires a commit.

Your task may include either or both of:
- fixing defects or provenance gaps in already-completed work
- implementing the next coherent required tranche that remains unfinished

Determine remaining work by:
1. reading the consumed `plan`
2. reading the consumed `implementation_review_report`
3. inspecting the current checkout and execution report

Do not assume the review report is complete.
If the review misses required unfinished plan tasks, identify and implement the next coherent required tranche yourself.

Prioritize in this order:
1. fix any blocking high-severity correctness, contract, provenance, or claim-boundary issue in already-implemented work
2. identify the earliest required unfinished plan task or coherent tranche
3. implement that tranche before optional later work

Write an updated execution report to the path recorded by the `execution_report_path` output-contract pointer.

The execution report must include:
- `Completed In This Pass`
- `Completed Plan Tasks`
- `Remaining Required Plan Tasks`
- `Generated Or Updated Artifacts`
- `Verification`
- `Pivots Or Stop Conditions`
- `Residual Risks`
