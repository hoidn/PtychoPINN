# Implementation Plan Template (Phased)

> How to use
> - Copy this file to `plans/active/<initiative-id>/implementation.md` and customize.
> - Keep a single evolving plan per focus; reuse a long‑lived Reports Hub until a milestone lands.
> - Record structural plan changes with a small Plan‑Update XML block inside the plan (not in `docs/fix_plan.md`).

## Initiative
- ID: <initiative-id>
- Title: <short title>
- Owner/Date: <name> / <YYYY‑MM‑DD>
- Status: pending | in_progress | blocked | done | archived
- Priority: <High|Medium|Low>
- Working Plan: this file
- Reports Hub (primary): `plans/active/<initiative-id>/reports/<ISO8601Z>/<slug>/`

## Context Priming (read before edits)
> Replace the example entries below with the documentation that is actually relevant to this initiative—do **not** copy these paths verbatim.
- [ ] docs/specs/spec-ptycho-core.md — Core physics/data contracts (example)
- [ ] docs/specs/spec-ptycho-workflow.md — Pipeline semantics (example)
- [ ] docs/architecture_torch.md — Backend-specific architecture notes (example)
- **Spec Alignment:** This initiative enforces/modifies the following active findings or spec shards (cite IDs/paths):
  - e.g., `CONFIG-001`, `docs/specs/spec-ptycho-config-bridge.md`

## Problem Statement
<1–3 sentences framing the problem and constraints>

## Objectives
- <objective 1>
- <objective 2>

## Deliverables
1. <artifact or outcome>
2. <artifact or outcome>

## Phases Overview
- Phase A — <name>: <one‑line objective>
- Phase B — <name>: <one‑line objective>
- Phase C — <name>: <one‑line objective>

## Exit Criteria
1. <criterion 1>
2. <criterion 2>
3. <criterion 3>
4. Test registry synchronized: update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` if tests changed; save `pytest --collect-only` logs under the active Reports Hub. Do not close if any selector marked Active collects 0 tests.

## Phase A — <name>
### Checklist
- [ ] A0: Nucleus — minimal guard/test proving the current failure (selector + expected artifact)
- [ ] A1: <task> (expected artifacts)
- [ ] A2: <task>
- [ ] A3: <task>

### Pending Tasks (Engineering)
- <next actionable steps grouped here; selectors live in input.md>

### Notes & Risks
- <risk or nuance>
- Rollback strategy if Phase A regresses the code/tests: <describe how to restore clean state quickly>

## Phase B — <name>
### Checklist
- [ ] B1: <task>
- [ ] B2: <task>

### Pending Tasks (Engineering)
- <next actionable steps grouped here; selectors live in input.md>

### Notes & Risks
- <risk>
- Rollback strategy if Phase B work regresses functionality: <describe revert path>

## Phase C — <name>
### Checklist
- [ ] C1: <task>
- [ ] C2: <task>

### Pending Tasks (Engineering)
- <next actionable steps grouped here; selectors live in input.md>

### Notes & Risks
- <risk>
- Rollback strategy if Phase C work regresses functionality: <describe revert path>

## Deprecation / Policy Banner (optional)
- <Use this slot to mark deprecations or policy reminders relevant to the initiative>

## Artifacts Index
- Reports root: `plans/active/<initiative-id>/reports/`
- Latest run: `<ISO8601Z>/<slug>/`

## Open Questions & Follow-ups
- <decision pending>
- <follow‑up validation>
