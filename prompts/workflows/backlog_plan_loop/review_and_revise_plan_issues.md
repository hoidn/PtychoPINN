Use the `Consumed Artifacts` section injected above as the authoritative input list.
Read all consumed artifacts before acting.

Also read:
- `state/backlog_item_path.txt` if present
- any backlog item file referenced there

Task:
Review and revise plan files in one pass so they are safe for canonical stage-to-stage execution.
This prompt must work for:
- one plan
- multiple related plans
- one or more backlog items that each reference plan files

Scope resolution:
- If consumed artifacts include `plan` or `plans`, include those files.
- If consumed artifacts include `backlog_item` or `backlog_items`, read each item's `plan_path` and include those files.
- Deduplicate the resulting plan list.
- If no plan file can be resolved, write a blocker report and set decision to `REVISE`.

Fix only these issue classes:
1. Inter-stage handoff clarity
- Each downstream stage must name exactly one upstream transition anchor artifact.
- For each stage transition, specify producer and consumer argument/field.
- Keep baseline/default comparison lane separate from transition-anchor lane.

2. Path and anchor integrity
- Remove hardcoded run-specific paths (timestamped roots, machine-local absolute paths, legacy run dirs).
- Replace them with plan-declared artifact references.
- Add fail-closed language: if required anchor source is missing or ambiguous, stop and report the missing source.

Actions:
1. Review all in-scope plan files against the two issue classes above.
2. Edit in-scope plan files in place to fix what can be fixed from available context.
3. If something cannot be resolved from available context, leave that part unchanged and record an explicit blocker.

Outputs:
1. Read destination path from `state/plan_review_path.txt`, then write a concise report containing:
   - in-scope plan files
   - files changed
   - what was fixed
   - remaining blockers
   - reruns required (if any)
   - updated stage handoff chain(s) for affected stages
2. Write decision to `state/plan_review_decision.txt`:
   - `APPROVE` only if no blockers remain
   - `REVISE` if any blocker remains

Constraints:
- Keep edits narrowly scoped to plan correctness and handoff clarity.
- Do not modify workflow code, DSL code, or unrelated docs.
- Do not fabricate artifacts, run results, or evidence.
- Do not modify `state/plan_review_path.txt`.
- Do not move backlog items.
