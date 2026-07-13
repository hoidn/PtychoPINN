# Design Template

> Copy this file to the design location for the initiative and customize it.
> Use a separate design when behavior, architecture, a public or scientific
> contract, migration policy, or material alternatives need a durable decision
> before implementation planning.
>
> Keep the document proportional to the decision. **Omit headings and prompts
> that are irrelevant** instead of filling them with `N/A` or boilerplate. This
> template does not require a separate ADR, dependency search, workflow contract,
> or asset inventory. Add one of those only when the design actually depends on
> it or a governing repository contract requires it.

## Metadata And Status

- ID: `<initiative-or-study-id>`
- Title: `<short title>`
- Status: `draft | candidate | approved | superseded | abandoned`
- Owner: `<name or role>`
- Date: `<YYYY-MM-DD>`
- Approval source: `<reviewer, decision record, or link; required when approved>`
- Supersedes / superseded by: `<repository-relative path, if applicable>`
- Related brief, issue, or plan: `<repository-relative path or tracker ID>`

Use `approved` only when the named authority has accepted the design. Keep live
progress, blockers, task ownership, and scheduling in the implementation plan or
tracker rather than updating this durable decision record with execution state.

## Context And Authority

Describe the context needed to understand the decision and identify the sources
that constrain it. Read `docs/index.md` first when present and list only sources
that materially govern this design.

- Context:
- Primary authority: `<repository-relative path and section>`
- Other governing specs, designs, policies, or findings:
- Existing behavior or implementation being changed:
- Authority or ownership boundary: `<which source owns each overlapping contract>`
- Assumptions that still need proof:

If sources disagree, resolve or explicitly escalate the conflict instead of
silently choosing one.

## Problem

- Problem to solve:
- Who or what is affected:
- Why the current behavior or structure is insufficient:
- Consequence of leaving it unchanged:

## Goals And Non-Goals

### Goals

- `<observable outcome this design should enable>`

### Non-Goals

- `<nearby outcome intentionally excluded>`

State the decision boundary precisely enough that later planning cannot expand
the initiative by implication.

## Decision And Alternatives

### Decision

Summarize the selected direction, why it fits the governing constraints, and
the most important consequence or trade-off.

- Selected direction:
- Rationale:
- Trade-offs accepted:

### Alternatives Considered

Record only credible alternatives that affected the choice.

| Alternative | Advantages | Disadvantages | Why not selected |
|---|---|---|---|
| `<option>` | `<benefit>` | `<cost or risk>` | `<reason>` |

Use a separate ADR only when repository convention or the scope of the decision
warrants one; this section is sufficient for an ordinary design.

## Design Details

Describe the system, study, process, or behavior at the level needed for an
implementation plan to name concrete work without inventing architecture.

### Components And Ownership

| Component or surface | Responsibility | Owner / source of truth | Depends on |
|---|---|---|---|
| `<component>` | `<single responsibility>` | `<module, document, or team>` | `<dependency>` |

Clarify maintained versus generated state and one-off versus reusable behavior
when either distinction matters.

### Data Or Control Flow

```text
<producer/input>
  -> <component or transformation>
  -> <output/state transition>
  -> <consumer>
```

Explain ordering, state transitions, and ownership changes that are not obvious
from the flow.

## Contracts And Interfaces

Document only interfaces affected or introduced by this design.

- Inputs and producers:
- Outputs and consumers:
- API, CLI, schema, path, type, or artifact contract:
- Validation and error contract:
- Ownership and source-of-truth rules:
- External or user-visible behavior:

For workflow-affecting work, define promoted workflow contracts only when a
producer and consumer need a stable boundary. Do not inventory local or runtime
artifacts merely because they may exist during implementation.

## Dependencies And Sequencing

- Existing dependencies or capabilities reused:
- New dependency or capability required:
- Preconditions:
- Ordering constraints and why they exist:
- Work that may proceed independently:

When a new external dependency is material, record its selection criteria,
version/source, license or operational constraints, and fallback. An exhaustive
dependency search is not required when the design does not introduce or select
one.

## Invariants, Failure, And Recovery

| Invariant | Credible failure mode | Detection | Required behavior / recovery |
|---|---|---|---|
| `<property that must remain true>` | `<how it can fail>` | `<signal or check>` | `<fail-closed, retry, rollback, repair, or escalation>` |

Also state, where relevant:

- Partial-write or interrupted-operation behavior:
- Idempotency, retry, and resume semantics:
- Rollback or safe-disable path:
- Behavior when required evidence or inputs are missing:

## Security, Operations, And Performance

Include only the applicable concerns and the constraints the implementation must
preserve.

- Security, privacy, trust-boundary, or secrets implications:
- Deployment, observability, support, or runbook implications:
- Capacity, latency, memory, compute, or cost constraints:
- Performance budget and how regressions will be detected:

## Evidence And Implementation Boundaries

Separate what must be proven before this design is accepted from details the
implementation plan may decide.

### Evidence Required For The Decision

- Existing authority or test that supports the design:
- Feasibility proof needed for an unproven capability:
- Evidence gap that remains an open prerequisite:

Require a small feasibility proof when the decision relies on an untested
generic mechanism, subsystem combination, API-preserving substrate change, or
negative architecture claim. Do not present an unproven capability as accepted.

### Fixed By This Design

- `<architecture, behavior, contract, policy, or evidence gate the plan must preserve>`

### Deferred To Implementation Planning

- `<file-level mechanics, task decomposition, command sequence, or reversible detail>`

## Compatibility And Migration

- Existing users, callers, data, or artifacts affected:
- Backward/forward compatibility contract:
- Migration or rollout path:
- Mixed-version or mixed-state behavior:
- Deprecation and removal conditions:
- Rollback implications:

## Verification Strategy

Name the kinds of evidence that can falsify the design-level claims. Exact
commands and task-by-task RED/GREEN sequencing belong in the implementation
plan.

- Contract and behavioral checks:
- Unit, integration, end-to-end, or workflow checks:
- Schema, artifact, or provenance checks:
- Performance or operational checks:
- Manual or reviewer inspection:
- Why this evidence is sufficient for the stated scope:

## Declarative Acceptance Scenarios

Express important externally observable outcomes without prescribing internal
implementation steps.

### Scenario: `<accepted behavior>`

- **Given** `<initial state and governing preconditions>`
- **When** `<actor or system action>`
- **Then** `<observable result or contract>`
- **And** `<additional invariant or evidence, if needed>`

### Scenario: `<failure or recovery behavior>`

- **Given** `<failure precondition>`
- **When** `<operation encounters it>`
- **Then** `<fail-closed, recovery, or diagnostic behavior>`

## Success Criteria

- `<measurable outcome that establishes the design solved its problem>`
- `<required contract or quality threshold>`
- `<evidence that must exist before implementation can be called complete>`

## Stop, Revise, Or Pivot Criteria

- Stop before implementation if:
- Revise this design if:
- Narrow or pivot the scope if:
- Escalate for a new decision if:

Keep these criteria tied to evidence, violated assumptions, safety, feasibility,
or claim limits—not dates or routine task status.

## Optional Scientific And Evidence Contract

Omit this entire section for work without scientific data, experiment artifacts,
reproducibility obligations, or reviewer-facing claims.

### Data And Artifact Identity

- Dataset/cohort/sample identity and split policy:
- Input and output artifact identities:
- Manifest, checksum, schema, or freshness requirements:
- Historical artifact reuse policy:

### Provenance And Reproducibility

- Code revision and configuration capture:
- Environment, hardware, and dependency capture:
- Randomness and determinism policy:
- Command, log, and lineage capture:
- Independent reproduction or rerun expectation:

### Claim Boundaries

- Claim the evidence may support:
- Claim the evidence must not imply:
- Required caveats, comparators, uncertainty, or generalization limits:
- Conditions that narrow or invalidate the claim:

### Scientific Pivot Criteria

- Continue the selected study direction if:
- Pivot to a smaller, exploratory, or alternate study if:
- Stop before paper- or reviewer-facing use if:

## Documentation Impact

- Normative specs or API docs to update:
- Architecture or developer docs to update:
- User, workflow, operations, or scientific docs to update:
- Index/discoverability updates:
- No durable documentation change is needed because: `<state why, if applicable>`

## Implementation Handoff

The implementation plan should preserve the accepted decisions above while
turning them into exact files, tasks, verification commands, and checkpoints.

- Design decisions the plan must not reopen:
- Required implementation sequence or gates:
- Required compatibility or migration work:
- Required verification and acceptance evidence:
- Deliberately deferred choices the implementer may make:
- Handoff owner or next decision authority:

Handoff readiness:

- [ ] The status and approval source are accurate.
- [ ] Governing authority and ownership are unambiguous.
- [ ] Material decisions, alternatives, contracts, and invariants are explicit.
- [ ] Acceptance, success, and stop/revise criteria are observable.
- [ ] Unproven capability claims are proven or recorded as prerequisites.
- [ ] Open questions that would materially change the design are resolved.

## Open Questions

List only unresolved questions that could materially change the decision,
contract, feasibility, or claim boundary.

| ID | Question | Decision owner | Resolution needed before | Resolution / status |
|---|---|---|---|---|
| Q1 | `<question>` | `<owner>` | `<approval or implementation>` | `<open or durable resolution>` |
