Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design`, `plan`, `execution_report_target`, and `progress_report_target` artifacts before acting.

Use executing-plans to implement the approved plan in the current checkout.
Do not use `git worktree` or another checkout.
If the repo is dirty, stay in the current checkout and leave unrelated files alone.
Do not modify YAML, prompt files, or transient state files unless the plan explicitly requires it.
Preserve layout and ownership decisions from the design and plan. If implementation needs to change a location or unit boundary, record the deviation and rationale in the execution report.

Choose exactly one implementation state for this pass:
- `COMPLETED` when the approved current-scope work is actually done and ready for checks/review
- `RUNNING` when long-running execution is still in progress and the final report is not ready yet
- `BLOCKED` only for a real semantic blocker that prevents completion in the current scope

Do not use `BLOCKED` just because training, benchmarking, or data generation is still running. Use `RUNNING` in that case.

Read the path recorded in the consumed `execution_report_target` artifact and write the concise final execution report there only when the state is `COMPLETED`.
Read the path recorded in the consumed `progress_report_target` artifact and write the concise progress report there when the state is `RUNNING` or `BLOCKED`.
Do not modify workflow pointer files; the workflow publishes them deterministically.

Write the JSON output bundle exactly as required by the output contract:
- always set `implementation_state`
- set `execution_report_path` only for `COMPLETED`
- set `progress_report_path` only for `RUNNING` or `BLOCKED`

For `COMPLETED`, the execution report must include:
- `Completed In This Pass`
- `Completed Plan Tasks`
- `Remaining Required Plan Tasks`
- `Verification`
- `Residual Risks`

For `RUNNING` or `BLOCKED`, the progress report must include:
- `Active Work`
- `Current Status`
- `Next Resume Condition`
- `Blocker` when the state is `BLOCKED`

For numerical parity or regression checks, report the `atol`/`rtol` or comparison standard used when the plan identifies one.

Finally, stage and commit only changes required for the current task with a descriptive commit message. Include durable design, plan, report, summary, and docs-index updates; exclude unrelated files, `.orchestrate/`, `state/`, and caches unless the plan requires them.
