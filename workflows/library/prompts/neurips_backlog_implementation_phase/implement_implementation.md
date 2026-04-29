Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design`, `plan`, `execution_report_target`, and `progress_report_target` artifacts before acting.

Use executing-plans to implement the approved plan in the current checkout.
Do not use `git worktree` or another checkout.
If the repo is dirty, stay in the current checkout and leave unrelated files alone.
Do not modify YAML, prompt files, or transient state files unless the plan explicitly requires it.
Preserve layout and ownership decisions from the design and plan. If implementation needs to change a location or unit boundary, record the deviation and rationale in the execution report.

Choose exactly one implementation state for this pass:
- `COMPLETED` when current-scope work is done, verified, and ready for review.
- `BLOCKED` only for a real blocker outside implementation authority, or after a documented failed recovery attempt.

If you launch a long-running command, keep ownership until it exits. On success, validate artifacts before `COMPLETED`. On failure, diagnose, fix in scope, and rerun or resume before choosing a final state.
Do not use `BLOCKED` just because a verification check, import, path, environment, or test-harness failure is red. Treat those as presumed recoverable implementation work: diagnose the failure, make a narrow in-scope fix when safe, and rerun the failing check before deciding state.
If the plan says to stop or block when a check fails before an expensive training, benchmark, or scientific run, interpret that as "do not start the expensive step until the check is fixed." Only emit `BLOCKED` when the blocker is a missing resource, unavailable hardware, roadmap conflict, external dependency outside current authority, user decision required, or unrecoverable after a documented narrow fix attempt.
When emitting `BLOCKED`, include a `blocker_class` in the output bundle using one of:
`missing_resource`, `unavailable_hardware`, `roadmap_conflict`, `external_dependency_outside_authority`, `user_decision_required`, or `unrecoverable_after_fix_attempt`.

Read the path recorded in the consumed `execution_report_target` artifact and write the concise final execution report there only when the state is `COMPLETED`.
Read the path recorded in the consumed `progress_report_target` artifact and write the concise progress report there when the state is `BLOCKED`.
Do not modify workflow pointer files; the workflow publishes them deterministically.

Write the JSON output bundle exactly as required by the output contract:
- always set `implementation_state`
- set `execution_report_path` only for `COMPLETED`
- set `progress_report_path` only for `BLOCKED`
- set `blocker_class` only for `BLOCKED`

For `COMPLETED`, the execution report must include:
- `Completed In This Pass`
- `Completed Plan Tasks`
- `Remaining Required Plan Tasks`
- `Verification`
- `Residual Risks`

For `BLOCKED`, the progress report must include:
- `Active Work`
- `Current Status`
- `Next Resume Condition`
- `Blocker` when the state is `BLOCKED`
- `Blocker Class` when the state is `BLOCKED`

For numerical parity or regression checks, report the `atol`/`rtol` or comparison standard used when the plan identifies one.
For parity or benchmark work, expected outputs, oracle data, fixtures, and generated evidence may be used only for tests, diagnostics, or validation. Do not use them as production answers or runtime lookup tables unless the approved design explicitly defines the feature as reference-data lookup.

Finally, stage and commit only changes required for the current task with a descriptive commit message. Include durable design, plan, report, summary, and docs-index updates; exclude unrelated files, `.orchestrate/`, `state/`, and caches unless the plan requires them.
