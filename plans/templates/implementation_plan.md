# Implementation Plan Template (Lean)

> Copy this file to `docs/plans/<initiative-id>/implementation.md` and customize.

## Initiative
- ID: <initiative-id>
- Title: <short title>
- Status: pending | in_progress | blocked | done | archived
- Owner: <name> (optional)
- Spec/Source: <primary spec or doc path> (optional)

## Goals
- <goal 1>
- <goal 2>

## Scope
- In scope:
  - <what this plan will do>
- Out of scope:
  - <what this plan will not do>

## Risks / Assumptions
- <risk or assumption 1>
- <risk or assumption 2>

## Phases

### Phase A — <name>
- [ ] A0: Nucleus / test-first gate
- [ ] A1: <task>
- [ ] A2: <task>

### Phase B — <name>
- [ ] B1: <task>
- [ ] B2: <task>

### Phase C — <name> (optional)
- [ ] C1: <task>
- [ ] C2: <task>

## Workflow Compatibility Contract

When this plan is executed by a backlog workflow:

- Backlog item must include:
  - `plan_path` pointing to this plan
  - `priority` integer
  - targeted `check_commands`
- Execution unit is the full plan (not ad-hoc partial slices).
- Completion requires all three:
  - verification commands pass
  - completion criteria are satisfied
  - required evidence is present

## Verification Commands
```bash
# Add the exact commands required to validate completion.
# Keep this targeted; avoid full-suite commands unless required.
```

## Completion Criteria
- [ ] <objective criterion 1>
- [ ] <objective criterion 2>
- [ ] <objective criterion 3>

## Required Evidence
- <artifact/log path 1>
- <artifact/log path 2>

## Artifacts Index
- Reports root: `docs/plans/<initiative-id>/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
