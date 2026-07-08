# Design Template

Use this template as an authoring aid for design documents in design -> plan -> implementation workflows.

Omit sections that do not apply. Do not pad small changes. Keep the design specific enough that the plan can execute without inventing architecture, contracts, or verification gates.

## Context And Authority

- Consumed brief or seed:
- Relevant repo docs, specs, architecture docs, findings, or templates consulted:
- Existing implementation or workflow conventions that govern this work:
- Decisions already made upstream:

## Problem And Scope

- Problem being solved:
- In scope:
- Out of scope:
- Non-goals:
- User-visible, scientific, workflow, or developer-facing outcome expected:

## Decision Summary

- Recommended implementation shape:
- Why this shape is appropriate:
- Alternatives considered:
- Semantically material choices and justification:
- Existing helpers, conventions, defaults, adapters, or transformations reused or rejected:

## Architecture And Modularity

Use this section when the work affects nontrivial tooling, validators, generators, services, workflows, stable APIs, data contracts, durable artifacts, or long-lived project structure.

- Components or modules:
- Responsibility of each component:
- Owned interfaces between components:
- Source-of-truth boundaries:
- Maintained files versus generated outputs:
- Curated data location and provenance, if any:
- Why a smaller single-unit implementation is acceptable, if choosing one:

## Contracts And Data Flow

- Inputs and producers:
- Outputs and consumers:
- Data, artifact, API, CLI, or workflow contracts:
- Type/schema expectations:
- Path and ownership expectations:
- Compatibility and migration constraints:

## Invariants And Failure Modes

- Invariants the implementation must preserve:
- Failure modes the design expects:
- How failures should be detected:
- When to pivot, narrow scope, or block:
- Work that must not be hidden behind vague "follow existing pattern" language:

## Documentation And Discoverability

- Docs/specs/indexes/templates/guides the implementation plan should update:
- Behavioral spec or API documentation impact:
- Internal architecture, development-process, or test-convention documentation impact:
- State explicitly when no durable documentation update is needed.

## Verification Strategy

- Unit, integration, runtime, workflow, or artifact checks:
- Generated artifact validation checks:
- Manual inspection checks:
- Review boundaries:
- Acceptance criteria:

## Rollback And Handoff

- Rollback strategy:
- Files or artifacts downstream phases may rely on:
- Deferred decisions and owners:
- Open risks:
