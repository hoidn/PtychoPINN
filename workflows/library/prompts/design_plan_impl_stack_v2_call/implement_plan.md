Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design` and `plan` artifacts before acting.

Use executing-plans to implement the approved plan in the current checkout.
Do not use `git worktree` or another checkout.
If the repo is dirty, stay in the current checkout and leave unrelated files alone.
Do not modify workflow YAML, prompt files, or runtime state files unless the plan explicitly requires it.

Write a concise execution report to the `execution_report_path` path specified in the Output Contract.

The execution report must include:
- `Completed In This Pass`
- `Completed Plan Tasks`
- `Remaining Required Plan Tasks`
- `Verification`
- `Residual Risks`

Finally, stage and commit only the implementation changes with a descriptive commit message.
