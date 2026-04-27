Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `approved_design`, `revision_context`, and `plan` artifacts before acting.

Use executing-plans to implement the approved revision-study plan in the current checkout.
Do not use `git worktree` or another checkout.
If the repo is dirty, stay in the current checkout and leave unrelated files alone.
Do not modify workflow YAML, prompt files, or runtime state files unless the plan explicitly requires it.
Do not edit the original revision design seed unless the approved plan explicitly requires it.
Do not commit unless the approved plan explicitly requires a commit.

Follow the plan's provenance and pivot gates. If evidence shows that a planned metric, figure, experiment, or manuscript claim would be scientifically unsafe, record the stop or pivot in the execution report instead of papering over it.

Write a concise execution report to the path recorded by the `execution_report_path` output-contract pointer.

The execution report must include:
- `Completed In This Pass`
- `Completed Plan Tasks`
- `Remaining Required Plan Tasks`
- `Generated Or Updated Artifacts`
- `Verification`
- `Pivots Or Stop Conditions`
- `Residual Risks`
