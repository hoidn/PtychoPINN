Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `steering`, `design`, `roadmap`, `selected_item_context`, and `progress_ledger` artifacts before acting.
Treat `selected_item_context` as the authoritative backlog-item content and queue location; use any `selection_source_path` there only as provenance.

Draft a fresh execution-ready plan for the selected backlog item.

The steering document, roadmap, and selected backlog item are binding planning context, not background reading. The plan must preserve their scope boundaries, fairness constraints, required evidence, and non-goals unless the consumed design or roadmap explicitly allows a pivot.

The approved plan will later be implemented through a generic implementation phase that sees only the approved design and this approved plan. Make the plan self-contained enough that implementation does not need to reread the raw backlog item, steering document, or roadmap to discover scope, claim boundaries, prerequisites, or required checks.

Use the selected backlog item's previous `plan_path` and any previous plan content only as background context. The fresh plan you write is the new execution authority.
Plans must keep ordinary long-running commands under implementation ownership until terminal success or recoverable failure handling is complete.

Before writing the task checklist, decide whether the work needs an `Implementation Architecture` section.

Include that section when the work has more than one meaningful implementation unit, crosses module or artifact boundaries, changes durable contracts, or would become hard to verify or maintain if responsibilities were left vague. Keep it brief and concrete.

The plan must include:
- the selected backlog objective, scope, and explicit non-goals
- steering and roadmap constraints that bound the work
- prerequisite status from the progress ledger when it matters
- the concrete file and artifact targets likely to change
- proportionate implementation tasks in execution order
- verification steps for each meaningful tranche
- the backlog item's `check_commands` as required deterministic checks unless the plan explicitly justifies a narrower or stronger replacement
- any required documentation or index updates when durable project knowledge changes

When a check must pass before an expensive training, benchmark, or scientific
run, state that the expensive step must wait for a green check. Do not instruct
implementation to mark the item `BLOCKED` merely because a normal verification
check, import, path, environment, or test-harness failure occurs. Instead,
require diagnose/fix/rerun first, and reserve `BLOCKED` for missing resources,
unavailable hardware, roadmap conflict, external dependency outside current
authority, user decision required, or a failure that remains unrecoverable after
a documented narrow fix attempt.

Do not silently expand the work to later roadmap phases or unrelated backlog items.

For the output contract's `plan_path`, read the path recorded in that file and write the plan document there. Leave the `plan_path` file containing only the path.
