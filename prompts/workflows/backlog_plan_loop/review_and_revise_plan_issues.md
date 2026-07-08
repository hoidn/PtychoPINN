Use the `Consumed Artifacts` section injected above as the authoritative input list.
Read all consumed artifacts before acting.

Task:
Review and revise plan files in one pass so they are safe for canonical stage-to-stage execution.
We might be looking at:
- one plan
- multiple related plans
- one or more backlog items that each reference plan files

Scope resolution:
- If consumed artifacts include `plan` or `plans`, include those files.
- If consumed artifacts include `backlog_item` or `backlog_items`, read each item's `plan_path` and include those files.
- Deduplicate the resulting plan list.
- If no plan file can be resolved, report that as a blocker and set final decision to `BLOCKED`.

Artifact vocabulary constraint (KISS):
- Do not introduce new artifact file names, artifact classes, or contract nouns.
- When replacing hardcoded paths, use only artifact names already present in the in-scope plans (or explicitly cited consumed design docs).
- If current artifact naming is inconsistent, normalize by selecting from existing names in scope; do not invent a new one.
- If no existing artifact name can satisfy a required handoff, do not invent one: record a blocker and set final decision to `BLOCKED`.

Fix only these issue classes:
1. Inter-stage handoff clarity
- Each downstream stage must name exactly one upstream transition anchor artifact.
- For each stage transition, specify producer and consumer argument/field.
- Keep baseline/default comparison lane separate from transition-anchor lane.

2. Path and anchor integrity
- Remove hardcoded run-specific paths (timestamped roots, machine-local absolute paths, legacy run dirs).
- Replace them with plan-declared artifact references.
- Add fail-closed language: if required anchor source is missing or ambiguous, stop and report the missing source.

3. Lack of inter-plan back-references in sets of plans that are intended to be executed sequentially

Actions:
1. Review all in-scope plan files against the two issue classes above.
2. Edit in-scope plan files in place to fix what can be fixed from available context.
3. If something cannot be resolved from available context, leave that part unchanged and record an explicit blocker.

Revision mode:
- Apply minimal textual edits only (reference replacement, clarification, fail-closed wording).
- Do not redesign artifact schema or naming model in this pass.

Outputs:
1. Write a concise report containing:
   - in-scope plan files
   - files changed
   - what was fixed
   - remaining blockers
   - reruns required (if any)
   - updated stage handoff chain(s) for affected stages
2. Write final decision (post-revision status):
   - `READY` only if no blockers remain after this revision pass
   - `BLOCKED` if any blocker remains after this revision pass
3. Write all outputs exactly as specified by the injected output contract for this invocation.

Constraints:
- Keep edits narrowly scoped to plan correctness and handoff clarity.
- Do not modify workflow code, DSL code, or unrelated docs.
- Do not fabricate artifacts, run results, or evidence.
- Do not move backlog items.
