Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design`, `roadmap`, and `progress_ledger` artifacts before acting.

Select the next plan scope to draft, review, implement, and review.

Use the roadmap phase order and gates as the primary ordering authority, the design as the scope authority, and the progress ledger as the record of work already completed.

Roadmap phases are coarse ordering and gate boundaries; they are not necessarily the exact implementation tranche list. Decide the next plan scope at the granularity that makes implementation and review coherent. A selected scope may cover a whole roadmap phase, a prerequisite slice within a phase, a fallback/pivot decision, or a cleanup/evidence-packaging slice, as long as the scope is justified by the roadmap, design, and progress ledger.

A selected scope should be one coherent unit that can go through plan drafting, plan review iterations, implementation, and implementation review iterations without absorbing unrelated later roadmap phases.

Decision rules:
- Return `DONE` only when the progress ledger shows every material roadmap phase is complete or explicitly out of scope with a recorded reason.
- Return `BLOCKED` only when no next tranche can be selected without a human decision, missing prerequisite, failed gate, unavailable data, or unresolved contradiction between design and roadmap.
- Otherwise return `SELECTED` for exactly one next tranche.
- Prefer the earliest roadmap phase whose gate is not satisfied, then choose the smallest useful plan scope needed to advance that gate.
- Do not skip required work just because easier work is available.
- Keep optional or secondary work behind the roadmap gates that make it safe.

For `SELECTED`, write a concise tranche context Markdown file and put its relative path in `tranche_context_path`.
The tranche context must include:
- tranche id and title
- roadmap phase(s) covered
- design decisions that bind the tranche
- prerequisite evidence from the progress ledger
- exact scope and explicit non-goals
- expected outputs and gate checks
- any carry-forward notes from previous plans, execution reports, pivots, or blocked work

Use a safe slug for the selected tranche: lowercase letters, digits, and hyphens only. Encode the slug as the parent directory of `tranche-context.md`, for example:

```text
state/<project-or-drain-root>/items/<safe-tranche-slug>/tranche-context.md
```

For the output contract's bundle path, write a JSON object using this shape:

```json
{
  "selection_status": "SELECTED",
  "tranche_context_path": "state/example-project/items/phase-1-core-implementation/tranche-context.md",
  "selection_reason": "Phase 1 is the earliest unsatisfied required gate."
}
```

For `DONE`, omit selected-tranche fields and explain why:

```json
{
  "selection_status": "DONE",
  "selection_reason": "All required roadmap tranches are complete in the progress ledger."
}
```

For `BLOCKED`, omit selected-tranche fields and explain the blocker:

```json
{
  "selection_status": "BLOCKED",
  "selection_reason": "The next phase cannot start because its prerequisite gate is not satisfied."
}
```
