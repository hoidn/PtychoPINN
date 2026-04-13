# Probe Mischaracterization Second-Pass Implementation Review

Decision: REVISE

Manual correction after user review: source-provenance enforcement is out of
scope for this study. The review should not require parent/child source
fingerprints, child Git commit matching gates, `expected_source_provenance`
request fields, or paper-export blocks based on child source provenance. If a
diagnostic run overlapped with code changes, treat that as an operational
run-selection issue and rerun from the settled implementation before relying on
reviewer-facing metrics.

## Resolved

### IMPL-001 - Final PDF inspection and checklist closure

Status: RESOLVED
Severity: medium
Scope classification: required_in_scope

The execution report records PDF/manuscript inspection and checklist closure,
and the paper checklist marks the R3 probe-mischaracterization item complete
with the fixed-probe sensitivity resolution note.

### IMPL-003 - Required smoke child dispatch test

Status: RESOLVED
Severity: medium
Scope classification: required_in_scope

The implementation now includes a smoke child-mode dispatch test that calls
`main(["--child-smoke-runner", "--child-request-json", ...])` and asserts
parent-only work is skipped while child output paths come from the request.

## Withdrawn

### IMPL-002 - Full numeric grid has cross-child source provenance drift

Status: WITHDRAWN
Severity: info
Scope classification: non_blocking_observation

This is no longer a blocking finding. The design and plan now explicitly reject
source-provenance enforcement for this one-off reviewer-revision study. The
workflow should not add a source-provenance subsystem and should not block
execution or export solely on source-fingerprint drift.

The remaining practical rule is simpler: once the implementation is settled,
run the reviewer-facing grid from a fresh output root before relying on metrics.

## Remaining Required Plan Tasks

- Run focused verification after the manual implementation change.
- Run a fresh implementation review before treating the workflow as approved.
- If reviewer-facing metrics are still needed, use a new run root and rerun the
  full process-isolated grid from the settled implementation rather than trying
  to salvage or audit the old mixed-development diagnostic run.
