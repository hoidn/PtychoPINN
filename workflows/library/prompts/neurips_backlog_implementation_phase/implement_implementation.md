Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design`, `plan`, `execution_report_target`, and `progress_report_target` artifacts before acting.

Use executing-plans to implement the approved plan in the current checkout.
Do not use `git worktree` or another checkout.
If the repo is dirty, stay in the current checkout and leave unrelated files alone.
Do not modify YAML, prompt files, or transient state files unless the plan explicitly requires it.
Do not move backlog queue files between `docs/backlog/active/`, `docs/backlog/in_progress/`, `docs/backlog/done/`, or `docs/backlog/paused/`; the workflow owns queue transitions after review.
Preserve layout and ownership decisions from the design and plan. If implementation needs to change a location or unit boundary, record the deviation and rationale in the execution report.

For any reviewable in-scope progress, write the execution report at the path recorded in `execution_report_target`.
This includes partial current-scope progress. If required plan tasks remain, record them under `Remaining Required Plan Tasks`; do not write a blocked progress report merely because work remains.

Write the progress report at the path recorded in `progress_report_target` only for a real blocker outside implementation authority, or after a documented failed recovery attempt.

If you launch a long-running command, keep ownership until it exits. On success, validate artifacts before writing the execution report. On failure, diagnose, fix in scope, and rerun or resume before deciding whether a real external blocker remains.
Do not use `BLOCKED` just because a verification check, import, path, environment, or test-harness failure is red. Treat those as presumed recoverable implementation work: diagnose the failure, make a narrow in-scope fix when safe, and rerun the failing check before deciding state.
If the plan says to stop or block when a check fails before an expensive training, benchmark, or scientific run, interpret that as "do not start the expensive step until the check is fixed." Only write a blocked progress report when the blocker is a missing resource, unavailable hardware, roadmap conflict, external dependency outside current authority, user decision required, or unrecoverable after a documented narrow fix attempt.
When writing a blocked progress report, include a `Blocker Class` using one of:
`missing_resource`, `unavailable_hardware`, `roadmap_conflict`, `external_dependency_outside_authority`, `user_decision_required`, or `unrecoverable_after_fix_attempt`.

Read the path recorded in the consumed `execution_report_target` artifact and write the concise execution report there for reviewable progress.
Read the path recorded in the consumed `progress_report_target` artifact and write the concise progress report there only for a real blocker.
Do not modify workflow pointer files; the workflow publishes them deterministically.
The workflow derives its control state from those report artifacts. You may summarize the chosen state in your final response, but do not rely on prose as the durable record.

The execution report must include:
- `Completed In This Pass`
- `Completed Plan Tasks`
- `Remaining Required Plan Tasks`
- `Verification`
- `Residual Risks`

The blocked progress report must include:
- `Active Work`
- `Current Status`
- `Next Resume Condition`
- `Blocker`
- `Blocker Class`

For numerical parity or regression checks, report the `atol`/`rtol` or comparison standard used when the plan identifies one.
For parity or benchmark work, expected outputs, oracle data, fixtures, and generated evidence may be used only for tests, diagnostics, or validation. Do not use them as production answers or runtime lookup tables unless the approved design explicitly defines the feature as reference-data lookup.

Finally, stage and commit only changes required for the current task with a descriptive commit message. Include durable design, plan, report, summary, and docs-index updates; exclude unrelated files, `.orchestrate/`, `state/`, and caches unless the plan requires them.
