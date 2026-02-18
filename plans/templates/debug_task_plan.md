# Debug Task Plan Template (Optional)

> Use this template only for debugging/evidence-focused investigations.
> For features/refactors/behavior changes, use `docs/plans/templates/implementation_plan.md`.

## Initiative
- ID: <initiative-id>
- Title: <short title>
- Owner: <name>
- Status: pending | in_progress | blocked | done
- Parent Plan (optional): `docs/plans/<initiative-id>/implementation.md`

## Debug Goal
- <state the regression/bug to isolate>

## Scope Guardrails
- Investigation/evidence only unless explicitly authorized.
- Do not broaden into implementation work without updating or linking a canonical implementation plan.
- Preserve deterministic inputs (fixed seeds/checkpoints/datasets) for comparability.

## Hypothesis Ledger
| ID | Hypothesis | Status | Evidence | Next Test |
|---|---|---|---|---|
| H1 | <hypothesis> | Open | <artifact path> | <test to run> |

## Tasks

### Task 1: <name>
**Files:**
- Create/Modify: `<path>`

**Step 1: <action>**
- Run: `<command>`
- Expected: `<pass/fail signal>`

### Task 2: <name>
**Files:**
- Create/Modify: `<path>`

**Step 1: <action>**
- Run: `<command>`
- Expected: `<pass/fail signal>`

## Decision Gate
- Mark each hypothesis `Confirmed` / `Refuted` / `Inconclusive` with artifact paths.
- Choose one root-cause track (or mark blocked with reason).
- If inconclusive after planned tasks, define escalation (for example: bounded `git bisect`).

## Exit Criteria
1. Root cause isolated with reproducible evidence, or blocker documented with concrete return condition.
2. Artifacts captured under `docs/plans/<initiative-id>/reports/<YYYY-MM-DDTHHMMSSZ>/`.
3. Next step is explicit: implementation plan handoff or additional scoped investigation.

## Artifacts Index
- Reports root: `docs/plans/<initiative-id>/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`

