# Debug Task Plan Template (Optional)

> Copy to `docs/plans/YYYY-MM-DD-<initiative>-debug.md` for a bounded,
> hypothesis-driven investigation. Use the canonical implementation plan
> template at `plans/templates/implementation_plan.md` for authorized fixes,
> refactors, or feature work. A separate initiative folder is optional only for
> genuinely multi-document work.

## Investigation contract

- **Symptom:** <observable failure, including exact input and environment>
- **Goal:** <root cause or decision this investigation must produce>
- **Authority:** <bug report, affected contract, or parent plan>
- **Scope:** <systems, files, inputs, and time/runs authorized>
- **Out of scope:** <implementation or adjacent investigations not authorized>
- **Baseline / reproduction:** `<exact command>`
  - Expected: <specific failure signature>
- **Controlled variables:** <seed, dataset, checkpoint, version, hardware, or
  config that must remain fixed>

## Hypothesis ledger

Test the cheapest discriminating hypothesis first. Do not run broad experiments
that cannot change the next decision.

| ID | Hypothesis | Discriminating test | Expected if true | Expected if false | Status |
|---|---|---|---|---|---|
| H1 | <possible cause> | `<exact command or probe>` | <signal> | <signal> | open |

Use `confirmed`, `refuted`, or `inconclusive` only after recording the observed
result. When an experiment produces evidence that must survive the working
session, link its durable path in the relevant row or finding. Do not create an
archive/report tree when durable evidence is not needed.

## Investigation tasks

### Task 1: <test hypothesis H1>

**Files**

- Read: `<exact/path>`
- Temporarily instrument or create, if authorized: `<exact/path>`

**Procedure**

1. <single controlled change or observation>
2. Run: `<exact command>`
3. Expected: <result that confirms or refutes H1>
4. Record: <ledger update and durable evidence path only if evidence is
   produced>

### Task 2: <next hypothesis or boundary check>

Repeat only if Task 1 leaves more than one plausible cause.

## Decision and handoff

- **Root cause / current best explanation:** <claim supported by results>
- **Evidence:** <commands and observations; durable paths only where produced>
- **Ruled out:** <hypotheses and their falsifying evidence>
- **Claim limits:** <what this investigation did not establish>
- **Next action:** <link an implementation plan, name another bounded
  investigation, or state the return condition for a blocker>

## Completion and stop conditions

Complete when the root cause is reproducible and isolated enough to plan a fix,
or when a concrete blocker and return condition are documented. Stop earlier if
the next experiment exceeds authorized scope, changes production state, cannot
preserve controlled variables, or requires an unresolved design decision.
