# `lines_256` Candidate Factory Redesign Spec

## Status

Proposed workflow/controller design. This document specifies the next modular
boundary for broad `source` candidates in the `lines_256` v2 study.

## Purpose

The current `lines_256` v2 controller has the right outer shape:

- the thin workflow starts the controller
- the controller owns deterministic study mechanics
- smoke, scoring, keep/discard, queue movement, timeout/crash, and resume all
  remain controller-owned

The weakness is the proposal boundary. Today, proposal generation is
effectively one provider step that emits one candidate package. That works well
for:

- `run_config` candidates
- narrow `source` candidates

It is a poor fit for broader architecture redesigns that benefit from:

- explicit design
- scoped implementation planning
- bounded implementation/review loops
- targeted checks before the controller spends smoke/scored budget

This spec introduces a modular `candidate_factory` layer so candidate
production can vary while candidate evaluation stays unchanged.

## Goals

- keep the outer v2 controller loop unchanged in responsibility
- support two candidate-production modes:
  - `direct`
  - `redesign`
- keep one uniform candidate package contract downstream of proposal
- make broad redesigns queue-driven and explicit in phase 1
- reuse existing workflow patterns already present in `agent-orchestration` and
  `PtychoPINN`

## Non-Goals

- moving keep/discard or accepted-state mutation into YAML
- replacing the controller with a fully YAML-authored loop
- changing `results.tsv` semantics
- changing smoke/scored run contracts
- enabling controller-selected non-queued redesigns in phase 1

## Current State

Current authoritative surfaces:

- controller doc: `docs/studies/lines_256_controller_loop.md`
- controller script: `scripts/studies/lines_256_session_controller.py`
- thin workflow: `workflows/agent_orchestration/lines_256_session_controller.yaml`

Current behavior:

- the controller builds `proposal_context.json`
- the controller chooses prompt mode (`exploit` or `explore`)
- the controller runs one proposal-producing provider path
- the controller validates `candidate_metadata.json`
- the controller owns all downstream deterministic mechanics

The redesign in this spec is limited to the proposal/candidate-production
boundary.

## Proposed Architecture

Introduce a pluggable `candidate_factory` interface with two implementations:

1. `direct`
- the current exploit/explore prompt-driven proposal path
- intended for:
  - `run_config`
  - small `source` candidates

2. `redesign`
- a subordinate workflow that performs:
  - design
  - plan
  - implementation
  - targeted checks
  - optional bounded review/fix loop
  - candidate package finalization
- intended for:
  - broader architecture redesigns
  - queue-authored ideas that are too large for a single-step proposal path

The controller remains the sole owner of:

- iteration accounting
- smoke execution
- scored execution
- keep/discard/timeouts/crashes
- accepted-state mutation
- queue item movement after terminal outcome

## Factory Selection Policy

Phase 1 policy:

- default factory: `direct`
- only queued workflow ideas may request `redesign`
- non-queued free-form proposal generation always uses `direct`

Queue-item opt-in is explicit through frontmatter:

```yaml
candidate_factory: redesign
```

If `candidate_factory` is absent:

- treat the item as `direct`

If `candidate_factory` is invalid:

- controller records proposal failure artifacts
- controller does not move the queue item
- session remains retryable at proposal phase

The controller does not infer redesign mode heuristically in phase 1.

## Candidate Factory Interface

The controller should treat candidate production as one interface:

```text
produce_candidate(iteration_context) -> candidate_package
```

### Factory Input Contract

Every factory receives the same logical inputs:

- `session_id`
- `session_root`
- `iteration_index`
- `accepted_state`
- `recent_history`
- `search_summary`
- `proposal_mode`
- `proposal_mode_reason`
- optional queued workflow idea
- authoritative iteration output paths

These inputs may be materialized as:

- JSON files
- pointer files
- workflow inputs

but the logical contract must stay stable across factories.

### Factory Output Contract

Every factory must produce:

- `candidate_metadata.json`
- `proposal_result.json`

Every factory may additionally produce:

- `candidate_context.json`
- design doc
- plan doc
- execution session log
- check log
- review log

The controller must be able to continue using only:

- `candidate_metadata.json`
- existing deterministic smoke/scored behavior

without factory-specific downstream branches.

## Candidate Package Contract

`candidate_metadata.json` remains the authoritative candidate package file.

Required fields:

- `status`
- `candidate_kind`
- `base_ref`
- `note`
- `hypothesis`
- `smoke_command`
- `smoke_output_root`
- `smoke_log_path`
- `run_command`
- `output_root`
- `log_path`
- `comparison_png_path`

Additional required fields by `candidate_kind`:

### `source`

- `candidate_commit`
- `candidate_paths_file`

### `run_config`

- no candidate-commit fields required

Optional provenance fields allowed for any factory:

- `candidate_factory`
- `queue_item_path`
- `design_doc_path`
- `plan_doc_path`
- `implementation_session_log_path`
- `review_log_path`
- `check_log_path`

These provenance fields are additive and must not be required by the existing
smoke/scored controller path.

## Controller Responsibilities After Factory Output

Once a valid `candidate_metadata.json` exists, the controller behavior does not
change:

1. validate the candidate package
2. run smoke
3. run scored candidate
4. classify:
   - `KEEP`
   - `DISCARD`
   - `TIMEOUT`
   - `CRASH`
   - `BLOCKED`
5. update accepted state if warranted
6. move queued workflow idea only after terminal outcome

This is the core boundary of the redesign:

- candidate production is pluggable
- candidate evaluation remains controller-owned

## Redesign Workflow

Proposed new subordinate workflow:

- `workflows/agent_orchestration/lines_256_redesign_candidate_impl_review.yaml`

This workflow is a candidate producer, not a study controller.

### Redesign Workflow Role

It may:

- read queued redesign idea content
- design a candidate
- write a plan for that candidate
- implement the candidate in the current checkout
- run targeted candidate-specific checks
- run a bounded review/fix loop
- finalize the candidate package

It may not:

- score candidates
- update accepted state
- append to `results.tsv`
- move queue files across terminal-outcome folders
- decide keep/discard

## Workflow Boundary Contract

The redesign workflow should expose typed workflow inputs and outputs rather
than relying on implicit prompt text or ad hoc context.

### Required Inputs

- `queue_item_path`
- `proposal_context_path`
- `candidate_metadata_output_path`
- `proposal_result_output_path`
- `candidate_paths_output_path`
- `design_doc_output_path`
- `plan_doc_output_path`
- `execution_session_log_output_path`
- `check_log_output_path`

### Optional Inputs

- `review_log_output_path`
- `max_review_cycles`

All path inputs should be typed `relpath` and controller-provided.

### Required Outputs

- `candidate_metadata_path`
- `proposal_result_path`

### Optional Outputs

- `design_doc_path`
- `plan_doc_path`
- `execution_session_log_path`
- `check_log_path`
- `review_log_path`

The controller should treat successful workflow completion plus valid
`candidate_metadata.json` as the success condition for redesign production.

## Internal Workflow Phases

Recommended first-tranche internal workflow shape:

1. `PrepareRedesignBrief`
- materialize a concise redesign brief from:
  - queued idea
  - accepted state
  - recent history
  - search summary
  - authoritative output paths

2. `DesignCandidate`
- provider step
- writes a short candidate-specific design doc

3. `PlanCandidateImplementation`
- provider step
- writes a concrete implementation plan for this candidate only

4. `ImplementCandidate`
- provider step
- makes source changes
- writes candidate package files to controller-owned paths

5. `RunTargetedChecks`
- command step
- runs narrow checks appropriate for the candidate

6. `ReviewCandidate`
- provider step
- decides whether the implementation is ready or needs revision

7. `FixCandidate`
- provider step
- bounded fix loop if review asks for revision

8. `FinalizeCandidatePackage`
- assert that required candidate files exist
- write `proposal_result.json`

This should reuse the same broad shape already used by:

- `workflows/examples/ptychopinn_backlog_plan_slice_impl_review_loop.yaml`
- `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`

## Reused Patterns

Patterns to reuse from existing workflow examples:

- file-based queue selection and task contracts
- explicit execute/check/review separation
- bounded review/fix cycles
- deterministic publication of step outputs
- relpath-based authoritative artifact paths

Patterns explicitly not reused:

- backlog workflow as the outer study loop
- review workflow owning acceptance/ledger decisions
- prompt text owning queue mutation or global controller state

## Resume Semantics

Resume remains controller-owned.

Required controller behavior:

- if redesign production fails before valid `candidate_metadata.json` exists:
  - session remains retryable in proposal state
- if redesign production succeeded and `candidate_metadata.json` exists:
  - resume reuses that metadata
  - resume does not rerun redesign workflow blindly

This matches the existing transactional proposal semantics already described in
`docs/studies/lines_256_controller_loop.md`.

## Queue Semantics

Queue lifecycle remains controller-owned.

During redesign production:

- queued item stays in `docs/workflow_queue/active/`

After terminal scored outcome:

- controller moves the queued item to:
  - `accepted/`
  - `discarded/`
  - `blocked/`
  - `crashed/`
  - `timed_out/`

The redesign workflow must not move queue items.

## Controller Integration Points

Minimal controller changes required:

1. resolve active queue item
2. read optional `candidate_factory` frontmatter
3. select `direct` or `redesign`
4. for `redesign`, call the subordinate workflow with typed path inputs
5. on success, continue with the normal smoke/scored path
6. on failure before metadata, persist proposal failure state and remain
   retryable

The controller should not add new downstream branching after proposal based on
factory type.

## Proposed Artifacts

Controller-owned iteration directory remains authoritative for:

- `candidate_context.json`
- `candidate_metadata.json`
- `proposal_attempt.json`
- `proposal_result.json`
- smoke log
- scored log
- candidate assessment

Redesign-produced supplemental artifacts should also land under the iteration
directory or controller-provided relpaths, for example:

- `design.md`
- `implementation_plan.md`
- `implementation_session.md`
- `review.md`
- `targeted_checks.log`

This keeps per-iteration provenance co-located.

## Verification Requirements

Implementation of this design should include:

- controller unit tests for:
  - default `direct` selection
  - queued `candidate_factory: redesign`
  - invalid factory token handling
- one integration-style test proving redesign-produced candidate metadata is
  accepted by the existing smoke/scored controller path without downstream
  special cases
- one resume test for redesign failure before metadata
- one resume test for redesign success with metadata reuse
- orchestrator dry-run validation of the new redesign workflow

## Rollout Plan

### Phase 1

- queue-only redesign factory
- one redesign workflow
- no free-form redesign selection by controller heuristics

### Phase 2

Optional later extension:

- allow controller-selected redesign production for non-queued exploration
- keep the same candidate package contract
- keep the controller as the sole owner of smoke/scored/acceptance logic

## Recommendation

Do not redesign the outer `lines_256` controller loop.

Redesign only the proposal/candidate-production boundary into:

- `direct_candidate_factory`
- `redesign_candidate_factory`

with a uniform candidate package contract and queue-only redesign selection in
phase 1.

That is the narrowest redesign that supports broader architecture experiments
without collapsing the study into one giant prompt or moving deterministic
study mechanics out of the controller.
