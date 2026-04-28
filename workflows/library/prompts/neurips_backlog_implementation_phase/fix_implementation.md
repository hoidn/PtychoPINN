Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design`, `plan`, `execution_report`, `execution_report_target`, `checks_report`, and `implementation_review_report` artifacts before acting.

Use executing-plans to address the implementation review while staying aligned with the design and plan.
Do not use `git worktree` or another checkout.
If the repo is dirty, stay in the current checkout and leave unrelated files alone.
Do not modify YAML, prompt files, or transient state files unless the plan explicitly requires it.
Preserve layout and ownership decisions from the design and plan. If implementation needs to change a location or unit boundary, record the deviation and rationale in the execution report.

Determine remaining work by:
1. reading the consumed `plan`
2. reading the consumed `checks_report`
3. reading the consumed `implementation_review_report`
4. inspecting the current codebase and execution report

Prioritize in this order:
1. fix blocking correctness, contract, or maintainability issues in already-implemented work
2. make the required backlog checks pass when they still represent the approved verification contract
3. complete any current-scope work still required for approval
4. record genuine follow-up work without implementing it

Treat failing checks, import errors, path issues, environment propagation
issues, and test-harness failures as presumed recoverable implementation work.
Diagnose and fix them narrowly when they are within current authority. A failed
check may gate an expensive later step, but it is not by itself a reason to
abandon the current item.

If a failing check should change rather than the implementation, make that change only when the approved plan or review makes the authoritative verification contract itself part of the current scope. Record the rationale clearly in the execution report.
For parity or benchmark work, expected outputs, oracle data, fixtures, and generated evidence may be used only for tests, diagnostics, or validation. Do not use them as production answers or runtime lookup tables unless the approved design explicitly defines the feature as reference-data lookup.

Read the path recorded in the consumed `execution_report_target` artifact and update the concise execution report at that exact current-checkout-relative target path. Do not modify workflow pointer files; the workflow publishes them deterministically.

The execution report must include:
- `Completed In This Pass`
- `Completed Current-Scope Work`
- `Follow-Up Work`
- `Residual Risks`

Finally, stage and commit only changes required for the current task with a descriptive commit message. Include durable design, plan, report, summary, and docs-index updates; exclude unrelated files, `.orchestrate/`, `state/`, and caches unless the plan requires them.
