# Workflow-Contract Plan Supplement (Optional)

Use this only when an implementation plan adds or changes data that crosses
orchestrated workflow step or stage boundaries. It supplements—not replaces—the
canonical [implementation plan template](implementation_plan.md), whose tasks,
files, verification, completion criteria, and stop conditions still govern.

Prefer adding the sections below to the single plan at
`docs/plans/YYYY-MM-DD-<initiative>.md`. Create a companion document in an
initiative folder only when a genuinely multi-document contract needs its own
durable review surface.

## Coordination

- **Implementation plan:** <repository-relative path>
- **Workflow definition:** <exact path>
- **Backlog/runtime item, if applicable:** <identifier or path>
- **Owning contract / authority:** <spec, accepted design, or schema>
- **Change summary:** <cross-stage behavior being added or changed>

## Minimal-sufficient artifact set

Promote only artifacts needed for a downstream stage to execute, validate,
resume, or audit an acceptance claim. Keep step-local diagnostics and
intermediates implementation-local. There is no numeric ceiling, but every
promoted artifact must justify why existing artifacts cannot carry its contract;
if the set is difficult to reason about, simplify the boundary or split the
workflow deliberately.

### Artifact: <canonical artifact name>

- **Purpose / why workflow-level:** <downstream need this artifact satisfies>
- **Direction:** input | output | published state
- **Producer:** <exact step/stage and production condition>
- **Consumer(s):** <exact step/stage and how each uses it>
- **Type / schema:** <relpath, enum, integer, string, or schema + version>
- **Authority:** <source that defines meaning, required fields, and precedence>
- **Resolution:** <how the current producer output is located; no implicit
  timestamp-pinned historical fallback>
- **Validation:** <syntax, schema, path containment, completeness, or semantic
  checks performed before consumption>
- **Freshness:** <run/attempt identity, content hash, timestamp relationship, or
  producer-state rule proving it belongs to the current execution>
- **Failure behavior:** <fail-closed, retry, recover, or explicit optional path;
  name the surfaced error/status and forbid silent stale fallback>

Repeat this block for each minimal-sufficient workflow-level artifact.

## Implementation-local I/O

- `<name/path>` — <why it remains local and is not part of cross-stage
  producer/consumer semantics>

## Workflow wiring and evidence

For each workflow-level artifact, the implementation plan must name exact files
and commands that prove:

- the producer publishes the declared type/schema;
- the consumer resolves the current producer output and validates it before use;
- missing, invalid, unauthorized, or stale data follows the declared failure
  behavior;
- resume/retry behavior preserves authority and freshness; and
- workflow DSL declarations match runtime producer/consumer behavior.

Use RED/GREEN evidence for changed behavior, including at least one negative
case for missing or stale required data. State exact commands and expected
results in the implementation plan rather than duplicating its execution
checklist here.

## Contract acceptance

- [ ] The promoted set is minimal but sufficient for every cross-stage need.
- [ ] Every artifact names producer, consumer, type, validation, authority,
  freshness, resolution, and failure behavior.
- [ ] Workflow definition, runtime behavior, and owning contract agree.
- [ ] Current-run lineage is demonstrated without silent historical fallback.
- [ ] Implementation-local I/O has not been unnecessarily promoted.
