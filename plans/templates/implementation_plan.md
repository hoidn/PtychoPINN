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
- [ ] A1: <task> (expected artifacts)
- [ ] A2: <task>
- [ ] A3: <task>

### Do Now (Engineering)
- <single next actionable step + validating pytest selector>

### Notes & Risks
- <risk or nuance>

## Phase B — <name>
### Checklist
- [ ] B1: <task>
- [ ] B2: <task>

### Do Now (Engineering)
- <single next actionable step + selector>

### Notes & Risks
- <risk>

## Phase C — <name>
### Checklist
- [ ] C1: <task>
- [ ] C2: <task>

### Do Now (Engineering)
- <single next actionable step + selector>

### Notes & Risks
- <risk>

## Deprecation / Policy Banner (optional)
- <Use this slot to mark deprecations or policy reminders relevant to the initiative>

## Artifacts Index
- Reports root: `plans/active/<initiative-id>/reports/`
- Latest run: `<ISO8601Z>/<slug>/`

## Open Questions & Follow-ups
- <decision pending>
- <follow‑up validation>

## Plan‑Update Protocol (reminder)
Paste a small XML block immediately above a major plan edit to preserve audit history (keep XML inside the plan only):

```xml
<plan_update version="1.0">
  <trigger>why the change is needed</trigger>
  <focus_id><initiative-id></focus_id>
  <documents_read>docs/index.md, docs/TESTING_GUIDE.md, docs/fix_plan.md, …</documents_read>
  <current_plan_path>plans/active/<initiative-id>/implementation.md</current_plan_path>
  <proposed_changes>bullet summary of concrete edits</proposed_changes>
  <impacts>risks or required test reruns</impacts>
  <ledger_updates>what to add in docs/fix_plan.md Attempts History</ledger_updates>
  <status>draft|approved|blocked</status>
</plan_update>
```

