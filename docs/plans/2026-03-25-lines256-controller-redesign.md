# lines_256 Controller-Based Loop Redesign

## Problem

The current `lines_256` architecture loop is too brittle because it asks one YAML workflow to own too many responsibilities at once:

- autonomous candidate proposal
- smoke/scored run launching
- metric harvest
- ledger writes
- accepted-state tracking
- git rollback/checkpoint behavior
- crash/debug routing
- resume semantics
- live-checkout coexistence policy

In practice, this has produced failure modes that are much more about orchestration shape than about the study itself:

- baseline/scored runs succeed but a later inline harvest block fails
- parameter-only experiments are awkward because the loop assumes every credible candidate must become a source-changing commit
- git rollback behavior and human edits interact in fragile ways
- the study state is spread across many YAML steps, inline Python blocks, and prompt contracts

The current workflow is over-optimized for expressing the loop in DSL form and under-optimized for robustness.

## Actual Goals

The study needs to preserve these outcomes:

1. run a fresh comparable baseline for the current session
2. let an agent propose the next experiment autonomously
3. support both source-changing and parameter-only candidates
4. smoke-test candidates before spending the full scored budget
5. run one scored experiment per iteration under a fixed budget
6. record a durable experiment ledger and accepted state
7. keep or reject candidates deterministically
8. resume an interrupted session without ambiguity
9. avoid colliding with normal human development on `fno-stable`

Those are the real requirements. The study does **not** require that the full loop live inside one branch-heavy workflow YAML.

## Non-Goals

This redesign does not aim to:

- remove the legacy workflow immediately
- generalize a new orchestration framework for every study
- auto-merge winning changes back to `fno-stable`
- eliminate the agent from the loop
- redesign the scientific objective, metric, or baseline budget

## Options Considered

### 1. Keep patching the current YAML loop

Pros:
- lowest short-term migration cost
- no new controller script

Cons:
- keeps the current failure surface
- parameter-only candidates remain awkward
- hard to reason about resume, git, and harvest behavior together
- every new policy change tends to add more YAML and inline Python

Verdict: not recommended.

### 2. Thin workflow + Python session controller

Pros:
- keeps autonomous agent proposal in the loop
- moves deterministic study mechanics into normal Python
- supports `source` and `run_config` candidates naturally
- simpler testing and resume behavior
- keeps a small workflow entrypoint if orchestrator launch/resume/report is still useful

Cons:
- requires a study-specific controller script
- introduces a temporary period where legacy and new paths coexist

Verdict: recommended.

### 3. Pure Python study driver with no workflow wrapper

Pros:
- simplest runtime shape
- smallest orchestration surface

Cons:
- gives up the existing orchestrator wrapper entirely
- less consistent with other orchestrated study entrypoints

Verdict: plausible later, but not the recommended first redesign.

## Recommendation

Adopt a **thin workflow + Python session controller** design.

The workflow layer should become a minimal launch/resume wrapper, while one study-specific controller script owns the deterministic loop.

This keeps the agent in the loop for the judgment-heavy part, but removes the brittle deterministic mechanics from the DSL.

## High-Level Architecture

### 1. Dedicated run checkout

The new path should always run in a dedicated disposable checkout.

Rules:

- the study run checkout is for autonomous runtime state only
- `fno-stable` remains the human integration branch in a separate normal checkout
- winning source changes are ported back explicitly later
- urgent prompt/doc steering can still be applied in the run checkout as working-tree edits when necessary

This removes the biggest source of git-state collisions from the design.

### 2. Python session controller

Add a controller script, for example:

- `scripts/studies/lines_256_session_controller.py`

This script becomes the deterministic authority for:

- session initialization
- baseline launch and harvest
- proposal context assembly
- proposal validation
- smoke/scored run launching
- timeout/crash handling
- randomness-contract validation
- ledger append
- accepted-state updates
- git rollback for source-changing candidates
- resume/checkpoint behavior

### 3. Thin workflow wrapper

Keep an orchestrator entrypoint if desired, but make it small. For example:

- `workflows/agent_orchestration/lines_256_session_controller.yaml`

That wrapper should do only a few things:

- validate authoritative docs and required scripts
- call the controller script
- optionally publish a summary artifact

The loop itself should no longer be expressed as a large YAML control-flow graph.

## Candidate Model

The controller should treat the accepted object as an **accepted experiment state**, not just an accepted git ref.

### Accepted state

Suggested fields in accepted-state JSON:

- `accepted_ref`
- `accepted_candidate_kind`
- `accepted_run_command`
- `accepted_amp_ssim`
- `accepted_randomness_contract`
- `accepted_output_root`
- `accepted_comparison_png`
- `session_id`

### Candidate kinds

The controller should support two first-class candidate kinds:

#### `source`

- agent edits tracked source files
- agent runs a cheap smoke check and repairs obvious breakage
- controller validates changed paths
- controller creates or validates one candidate commit in the run checkout
- controller scores the candidate
- on rejection, controller resets to `accepted_ref`

#### `run_config`

- agent leaves tracked source files unchanged
- candidate changes only smoke/scored run parameters
- no candidate commit is created
- controller scores the candidate
- on rejection, controller performs no git rollback at all

This is the core simplification. Parameter-only experiments stop masquerading as fake source edits.

## Agent Boundary

The agent should keep exactly one responsibility:

- prepare one viable next candidate package

For proposal:

- read study docs
- read accepted state
- read a compact recent-history summary
- choose the next coherent hypothesis
- produce either a `source` or `run_config` candidate
- run a cheap smoke check
- repair obvious breakage
- emit one proposal artifact

The agent should **not** own:

- ledger appends
- metric parsing
- keep/discard logic
- timeout policy
- accepted-state updates
- session counters or loop termination
- git rollback/reset behavior

That deterministic control belongs in the controller.

## State and Artifact Layout

The new path should not share live session state with the legacy workflow.

Suggested isolation:

```text
state/lines_256_arch_improvement_v2/sessions/<session_id>/
  session.json
  accepted_state.json
  results.tsv
  proposal_context.json
  latest_candidate.json
  latest_smoke.json
  latest_scored_run.json
  summary.md

outputs/lines_256_arch_improvement_v2/sessions/<session_id>/
  baseline/
  candidates/<iteration_id>/
  comparison_pngs/
```

This avoids collisions with the legacy path's global state files such as:

- `state/lines_256_arch_improvement/accepted_state.json`
- `state/lines_256_arch_improvement/results.tsv`

## Controller Loop

The new controller loop should look like this:

1. validate study inputs
2. initialize session root
3. run fresh baseline
4. harvest baseline and write accepted state
5. repeat:
   - build compact proposal context
   - call proposal agent
   - validate proposal artifact
   - materialize candidate
   - run smoke if not already performed by the agent
   - run scored experiment
   - harvest outputs
   - append one ledger row
   - update accepted state on keep
   - rollback only when the candidate changed source and was not kept
   - optionally run one focused debug attempt only for true crash cases
6. write/update session summary

Resume should reload `session.json` and continue from the last completed phase, not infer state from branch ancestry.

## Logging and Recovery

The controller should make phase boundaries explicit.

Suggested phases:

- `init`
- `baseline`
- `proposal`
- `smoke`
- `score`
- `harvest`
- `accept_or_reject`
- `debug`
- `complete`

`session.json` should record:

- current phase
- current iteration
- current candidate kind
- current output/log paths
- last successful accepted state snapshot

That makes `resume` a normal state-machine restart, not a reconstruction exercise across many step-local artifacts.

## Parallel Rollout Plan

The new controller path should coexist with the current legacy workflow until it proves itself.

### Legacy path

Keep:

- `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml`

Treat it as the legacy implementation.

### New path

Add:

- `scripts/studies/lines_256_session_controller.py`
- optional thin workflow wrapper
- isolated `state/..._v2/` and `outputs/..._v2/` roots

### Isolation requirements

The two paths must not share:

- accepted-state files
- results ledger
- output roots
- tmux session names
- dedicated run checkouts

Parallel rollout is only safe if the new path is fully isolated from the legacy live state.

## Validation Criteria For Promotion

Do **not** deprecate the legacy workflow until the new path proves itself on real runs.

Required capabilities:

1. fresh baseline completes reliably
2. one source-changing candidate completes end-to-end
3. one parameter-only candidate completes end-to-end
4. keep/discard behavior is correct
5. timeout handling is non-terminal and deterministic
6. crash handling is non-terminal when appropriate and clearly terminal when blocked
7. resume works after an interrupted run
8. ledger and accepted-state outputs are coherent and session-local

Promotion criteria:

- several real sessions complete with fewer orchestration failures than the legacy path
- no shared-state collisions with the legacy implementation
- the new path handles the obvious higher-`fno_modes` parameter-only experiment cleanly

Only after that should the legacy path be marked deprecated.

## Testing Strategy

The redesign should be tested mostly as normal Python behavior rather than as giant workflow topology.

Tests should focus on:

- baseline harvest
- candidate kind routing
- keep/discard logic
- source rollback
- `run_config` no-op git handling
- timeout/crash decisions
- resume from phase checkpoints

Keep a smaller set of workflow tests only for:

- the thin wrapper entrypoint
- required input validation
- controller invocation contract

## Migration Slices

### Slice 1: controller skeleton

- add session root and state model
- run baseline through the controller
- harvest baseline

### Slice 2: proposal integration

- call the proposal agent
- support `source` and `run_config` proposal artifacts

### Slice 3: scored-run loop

- score one candidate
- ledger append
- keep/discard handling

### Slice 4: resume and crash/debug

- checkpoint phases
- add crash-debug attempt
- validate resume

### Slice 5: parallel rollout

- document legacy vs v2 usage
- run both paths in isolation
- compare robustness

## Open Questions

These do not block the redesign direction, but they should be answered during implementation:

1. Should the agent itself perform the smoke step, or should the controller rerun smoke as a deterministic confirmation?
2. Should the new path keep an orchestrator wrapper at all, or should the controller be launched directly by a runbook command first?
3. What is the smallest useful recent-history summary artifact for proposal quality without reintroducing prompt bloat?

## Summary

The right simplification is not to remove the agent. It is to remove the deterministic experiment loop from the large YAML workflow.

The recommended replacement is:

- dedicated run checkout
- study-specific Python session controller
- thin workflow wrapper
- isolated parallel rollout alongside the legacy path

That keeps autonomous experiment proposal while replacing the brittle orchestration shape with a more testable and robust controller.
