# `lines_256` Proposal Transaction Recovery Plan

## Goal

Make the v2 `lines_256` controller recover cleanly when a proposal agent fails after entering `proposal_running` but before writing `candidate_metadata.json`.

## Root Cause

The current controller treats proposal generation as atomic:

- it sets `proposal_running` before durable proposal metadata exists
- it immediately requires `candidate_metadata.json` after the provider returns
- resume treats missing proposal metadata as fatal instead of recoverable
- the proposal agent currently owns smoke execution side effects that should belong to deterministic controller code

This leaves the session unrecoverable when the provider exits mid-proposal, such as the observed `Selected model is at capacity` failure.

## Scope

Implement a proposal transaction model in the Python controller for the `lines_256` v2 study.

In scope:
- controller-owned proposal attempt/result artifacts
- recoverable resume semantics for partial proposal steps
- deterministic smoke ownership in the controller rather than the proposal prompt path
- targeted tests and one real resume validation

Out of scope:
- redesigning the full search policy
- changing keep/discard criteria
- rewriting the legacy v1 workflow

## Design

### 1. Add controller-owned proposal transaction artifacts

For each iteration add controller-authored files under the iteration root, for example:

- `proposal_attempt.json`
- `proposal_result.json`

`proposal_attempt.json` should capture:
- timestamp
- iteration
- prompt/log paths
- candidate metadata path expected for the proposal
- whether this is a normal proposal or debug proposal

`proposal_result.json` should capture:
- provider exit code
- completion status classification
- stdout/stderr or log excerpt summary
- whether validated candidate metadata was produced
- whether the step is retryable

These artifacts must be written by the controller, not the provider.

### 2. Make proposal phases transactional

Keep the current broad phase names if possible, but change their meaning:

- `proposal_running` means an attempt is in progress or was interrupted
- `proposal_complete` means validated `candidate_metadata.json` exists

Controller behavior:
- write `proposal_attempt.json`
- set phase to `proposal_running`
- launch provider
- validate `candidate_metadata.json`
- only then set `proposal_complete`

If provider exits before valid metadata exists:
- write `proposal_result.json`
- classify the outcome as transient/interrupted/fatal
- do not hard-exit solely because metadata is missing

### 3. Recover on resume instead of aborting

For `proposal_running` resume:

- if valid metadata exists, proceed as today
- if metadata is missing but proposal_result says transient/interrupted, reset to `proposal_pending` and rerun proposal
- if metadata is missing and no result exists, conservatively synthesize a transient failure result and return to `proposal_pending`
- only stop the session for an explicit non-retryable controller-detected blocker

This removes the current `Missing candidate metadata` fatal exit path as the default outcome.

### 4. Move smoke execution ownership back to the controller

The proposal provider should only:
- propose the candidate
- edit files for source candidates
- write proposal metadata and candidate paths

The controller should own:
- smoke launch
- smoke result persistence
- transition from proposal to scored-running

This keeps deterministic mechanics out of the prompt/provider path and reduces the chance of partially executed proposal-side effects.

### 5. Keep backward compatibility for existing sessions where feasible

Support sessions that do not yet have proposal transaction artifacts.

Resume policy for old sessions:
- if `proposal_running` and metadata exists, continue
- if `proposal_running` and metadata missing, treat as transient failed proposal and reset to `proposal_pending`

This should allow recovery of the currently stalled `20260331T015545Z` session without manual filesystem surgery.

## Tests

Add targeted controller tests for:

1. proposal provider exits nonzero before metadata exists
- controller writes proposal result
- session does not hard-exit on missing metadata
- resume resets to `proposal_pending` and retries

2. proposal provider exits zero but metadata missing
- classified as invalid/incomplete proposal
- recoverable on resume

3. proposal provider produces valid metadata
- phase advances to `proposal_complete`
- normal scored flow still works

4. backward-compatible recovery for legacy sessions
- old session at `proposal_running` with missing metadata resumes by resetting to `proposal_pending`

5. smoke ownership contract
- smoke is invoked by controller after proposal validation, not by the provider path

Use narrow pytest selectors first and run `pytest --collect-only` if new tests are added/renamed.

## Validation

1. Run narrow controller pytest selectors for proposal transaction and resume recovery.
2. Run the full `test_lines_256_session_controller.py` module.
3. Run an orchestrator dry-run for `workflows/agent_orchestration/lines_256_session_controller.yaml`.
4. Resume the real stalled session `20260331T015545Z` and verify it no longer exits with `Missing candidate metadata`.

## Implementation Order

1. Add proposal attempt/result helpers and JSON schema.
2. Refactor proposal execution into a transaction.
3. Update resume logic for `proposal_running` recovery.
4. Move smoke ownership into deterministic controller code.
5. Add/adjust tests.
6. Validate on the real stalled session.
7. Update `docs/studies/lines_256_controller_loop.md` and `docs/findings.md` if the controller contract changes.
