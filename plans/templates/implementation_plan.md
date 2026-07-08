# Implementation Plan Template (Lean)

> Copy this file to `docs/plans/<initiative-id>/implementation.md` and customize.

## Initiative
- ID: <initiative-id>
- Title: <short title>
- Status: pending | in_progress | blocked | done | archived
- Owner: <name> (optional)
- Spec/Source: <primary spec or doc path> (optional)

## Compliance Matrix (Mandatory)
> List the specific Spec constraints, Fix-Plan ledger rows, and Findings/Policies this initiative must honor.
- [ ] **Spec Constraint:** <e.g., `spec-db-core.md §5.2 — Variance model definition`>
- [ ] **Fix-Plan Link:** <e.g., `docs/fix_plan.md — Row [PHYSICS-LOSS-001]`>
- [ ] **Finding/Policy ID:** <e.g., `CONFIG-001`, `POLICY-001 (PyTorch Optional)`>

## Spec Alignment
- **Normative Spec:** <path to spec file>
- **Key Clauses:** <specific requirements this plan must satisfy>

## Architecture / Interfaces (optional)
- Components/boundaries touched: <2-4 bullets>
- Primary data flow: <request/job/data path summary>
- External interfaces/contracts impacted: <APIs, files, schemas, CLI flags>

## Context Priming (read before edits)
- Primary docs/specs to re-read: <list files + sections>
- Required findings/case law: <`docs/findings.md` IDs + summary>
- Related telemetry/attempts: <links to relevant artifacts or plan history>
- Data dependencies to verify: <external inputs and manifest references>

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

## Dependency Analysis (Refactors only)
> Include this section only if the work changes module boundaries, import graph, or state ownership.
- Touched modules: <list>
- Circular import risks: <analysis>
- State migration: <how state moves from old to new>

## Workflow Compatibility Contract

When this plan is executed by a backlog workflow:

- Backlog item must include:
  - `plan_path` pointing to this plan
  - `priority` integer
  - targeted `check_commands`
- Execution unit is the full plan (not ad-hoc partial slices).
- Completion requires both:
  - verification commands pass
  - completion criteria are satisfied

## Verification Commands
```bash
# Add the exact commands required to validate completion.
# Keep this targeted; avoid full-suite commands unless required.
```

## Completion Criteria
- [ ] <objective criterion 1>
- [ ] <objective criterion 2>
- [ ] <objective criterion 3>

## Artifacts Index
- Reports root: `docs/plans/<initiative-id>/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
