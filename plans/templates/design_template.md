# Design Template

> Copy this file to the design location for the initiative and customize it.
> Use this before writing an implementation plan when the work needs design,
> scientific/provenance decisions, architecture choices, or ADR-style rationale.
>
> Keep the design as small as the decision requires. For narrow changes, write
> concise answers and mark optional subsections `N/A`; do not pad the document to
> satisfy headings that are irrelevant. For workflow-driven design-plan-implement
> stacks, use this document as the design source of truth before planning.

## Design Metadata

- ID: `<initiative-or-study-id>`
- Title: `<short title>`
- Status: draft | candidate | approved | superseded | abandoned
- Owner: `<name>` (optional)
- Date: `<YYYY-MM-DD>`
- Source brief / issue: `<path or tracker id>`
- Related plan: `<path>` (leave blank until planning starts)
- Related checklist / backlog item: `<path>` (optional)

Use `approved` only after the review loop or human owner accepts the design.
Drafting prompts should create a `draft` or `candidate` design, not an
"approved" design.

## Consumed Inputs and Authority

List the documents and artifacts that constrain this design. Prefer specific
paths and sections over broad references.

- Primary source: `<path>`
- Docs index: `docs/index.md` read first? `<yes | no | absent>`
- Normative specs: `<path> §section>`
- Architecture docs: `<path> §section>`
- Workflow guides / runbooks: `<path> §section>`
- Project docs: `<path> §section>`
- Prior artifacts or reports: `<path>`
- Known findings / policies: `<Finding ID or policy>`

When `docs/index.md` exists, read it first and use it to select the relevant
specs, architecture docs, workflow guides, runbooks, and `docs/findings.md`.
List only the documents that actually constrain the design; do not turn broad
doc globs into a required reading list.

Authority order for this design:

1. `<highest-priority source>`
2. `<next source>`
3. `<lower-priority context>`

## Problem and Scope

State the problem this design solves and the claim or behavior surface it is
allowed to change.

- Problem:
- User / reviewer / system need:
- In scope:
- Out of scope:
- Non-goals:

## Decision Summary

Summarize the intended direction in a few bullets. This section should let a
reviewer understand the design without reading the whole document.

- Decision:
- Rationale:
- Expected implementation shape:
- Claim or behavior limits:
- Pivot or abandonment condition:

## Decision Records

Use ADR entries for consequential choices. Include them when a choice affects
architecture, dependencies, scientific claims, reproducibility, external APIs,
data contracts, baseline policy, or future maintenance.

### ADR-001: `<decision name>`

- Status: proposed | accepted | superseded | rejected
- Decision:
- Context:
- Rationale:
- Alternatives considered:
  - `<alternative>` - `<reason accepted/rejected>`
- Consequences:
- Evidence required before implementation:
- Follow-up required if this decision changes:

### ADR-002: `<decision name>` (optional)

- Status:
- Decision:
- Context:
- Rationale:
- Alternatives considered:
- Consequences:
- Evidence required before implementation:
- Follow-up required if this decision changes:

## Proposed Design

Describe the system, study, process, or workflow at the level needed for a
planner to produce concrete tasks.

### Implementation Shape

- Existing code or workflow to reuse:
- New files or artifacts likely needed:
- Files or APIs likely touched:
- Files or APIs that must not be touched:
- One-off versus reusable boundary:
- Design-plan-implement boundary:
  - Decisions this design must make:
  - Details intentionally deferred to the implementation plan:

### Core Contracts and Invariants

List the properties that must hold even if implementation details change.

- Contract:
- Invariant:
- Failure mode if violated:
- How the implementation should prove it:

### Data Flow / Control Flow

```text
<input/source>
  -> <step/component>
  -> <artifact/output>
  -> <consumer/verification>
```

### Workflow / Artifact Contract (if workflow-affecting)

Use this subsection when the design may author, modify, or be executed by an
orchestration workflow. Keep the workflow-level contract surface small; local
implementation files belong in the implementation plan.

- Workflow or stack affected:
- Workflow-level artifacts:

| Artifact | Producer | Consumer | Type | Validation / gate rule |
|---|---|---|---|---|
| `<artifact>` | `<step>` | `<step>` | `<file/json/report/etc>` | `<rule>` |

- Runtime-only or local artifacts not promoted to workflow contract:
- Prompt-owned judgment standard:
- Workflow-owned deterministic control:
- Resume / re-run implications:
- Non-contract details deferred to implementation plan:

## Data, Dependency, and Provenance Decisions

Use this section even for non-study work if the design depends on external
state, installed packages, generated artifacts, model checkpoints, datasets, or
environment-specific behavior.

### Data and Artifact Identity

- Required inputs:
- Required outputs:
- Checksum / manifest fields:
- Freshness or cache policy:
- Reuse policy for historical artifacts:

### Dependency Discovery (optional)

Use when the implementation might need a new package, solver, tool, service, or
system dependency.

- Discovery scope:
  - Search current repo and environment:
  - Search external package sources:
  - Candidate acceptance criteria:
  - Candidate rejection criteria:
- Required candidate manifest fields:
  - name:
  - URL or source:
  - version / commit:
  - license:
  - install command:
  - import or API entry point:
  - supported algorithm / feature:
  - fit for this design:
  - rejection reason, if rejected:
- Installation policy:
- Fallback if no acceptable dependency is found:

### Provenance and Reproducibility

- Required command capture:
- Required environment capture:
- Required random seeds or determinism policy:
- Required artifact manifest fields:
- Required evidence logs:

## Claim, Behavior, or API Boundaries

State what the design may claim after implementation and what it must not imply.
For non-paper work, interpret "claim" as user-visible behavior or API contract.

- Allowed claim / behavior:
- Disallowed claim / behavior:
- Required caveat or limitation:
- Conditions that narrow the claim:

## Pivot Criteria and Stop Conditions

Define fail-closed behavior before implementation starts.

- Pivot to smaller scope if:
- Stop before user-facing / reviewer-facing claims if:
- Treat as exploratory only if:
- Escalate for human decision if:

## Required Final Assets

List final outputs if the design succeeds.

- Code or scripts:
- Tests:
- Machine-readable outputs:
- Figures / tables / docs:
- Manifests / logs:
- Checklist, backlog, or changelog updates:

If the design is abandoned or pivots:

- Required artifact note:
- Required rejected-candidate or failed-attempt summary:
- Required docs / checklist updates:

## Verification Plan

List checks that prove the design was implemented correctly. Keep this at the
design level; exact command sequencing belongs in the implementation plan.

- Unit or integration tests:
- Artifact inspections:
- Manifest/schema checks:
- Reproducibility checks:
- Manual inspection:
- Paper or docs build checks, if relevant:

## Open Questions

Track unknowns that must be resolved before implementation planning.

| ID | Question | Owner | Resolution needed by | Status |
|---|---|---|---|---|
| Q1 | `<question>` | `<owner>` | before plan | open |

## Planning Handoff Checklist

- [ ] Design status is `draft`, `candidate`, or post-review `approved` with approval source recorded.
- [ ] `docs/index.md` was used when present, and selected specs/docs/findings are listed.
- [ ] Optional sections are concise, omitted, or marked `N/A`; the design is not padded for small changes.
- [ ] ADR entries record consequential choices and rejected alternatives.
- [ ] Workflow/artifact contract is filled in or marked `N/A`; workflow-level artifacts have producer, consumer, and gate rule.
- [ ] Dependency discovery is specified if new packages/tools might be needed.
- [ ] Data/provenance contracts include enough fields to audit outputs.
- [ ] Pivot and stop conditions are concrete enough for an implementation review.
- [ ] Required final assets include docs/checklist/changelog updates where relevant.
- [ ] Open questions are either resolved or carried into the plan as blocking tasks.
