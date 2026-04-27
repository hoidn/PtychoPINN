Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design` and `plan` artifacts before acting.

Use executing-plans to implement the approved plan in the current checkout.
Do not use `git worktree` or another checkout.
If the repo is dirty, stay in the current checkout and leave unrelated files alone.
Do not modify YAML, prompt files, or transient state files unless the plan explicitly requires it.
Preserve layout and ownership decisions from the design and plan. If implementation needs to change a location or unit boundary, record the deviation and rationale in the execution report.

For the output contract's `execution_report_path`, read the path recorded in that file and write the concise execution report to that current-checkout-relative path. Leave the `execution_report_path` file containing only the path.

The execution report must include:
- `Completed In This Pass`
- `Completed Plan Tasks`
- `Remaining Required Plan Tasks`
- `Verification`
- `Residual Risks`

For numerical parity or regression checks, report the `atol`/`rtol` or comparison standard used when the plan identifies one.
For parity or benchmark work, expected outputs, oracle data, fixtures, and generated evidence may be used only for tests, diagnostics, or validation. Do not use them as production answers or runtime lookup tables unless the approved design explicitly defines the feature as reference-data lookup.

Finally, stage and commit only changes required for the current task with a descriptive commit message. Include durable design, plan, report, summary, and docs-index updates; exclude unrelated files, `.orchestrate/`, `state/`, and caches unless the plan requires them.
