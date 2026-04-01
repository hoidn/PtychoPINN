# `lines_256` Controller Loop (V2)

This document describes the parallel v2 `lines_256` experiment path driven by
`scripts/studies/lines_256_session_controller.py`.

## Purpose

The v2 path keeps the autonomous proposal behavior but moves deterministic study
mechanics out of the legacy branch-heavy YAML loop and into a session-local
Python controller.

The goals are:
- keep baseline, scoring, keep/discard, timeout/crash, and resume behavior in
  ordinary Python instead of large inline-YAML script blocks
- support both source-changing and parameter-only (`run_config`) candidates
- isolate v2 state and outputs from the legacy loop while both paths coexist

## Rollout Status

This is the v2 parallel replacement path. The legacy workflow remains supported
during rollout, but the controller now owns a real iterative `start` / `resume`
loop instead of baseline-only scaffolding.

- Legacy doc: `docs/studies/lines_256_arch_improvement_loop.md`
- Legacy workflow: `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml`
- V2 controller script: `scripts/studies/lines_256_session_controller.py`
- V2 thin wrapper: `workflows/agent_orchestration/lines_256_session_controller.yaml`

## Run Isolation

Like the legacy loop, this study should run in a dedicated run checkout because
the `source` candidate path still has explicit rollback/checkpoint behavior.

Do not treat the human development checkout as the live controller runtime
surface.

The dedicated run checkout should stay on a named session branch:

- `lines256/session/<session_id>`

This branch is the operator-facing runtime handle only. Exact commit SHAs remain
the authoritative provenance for:

- `accepted_ref`
- resolved source candidate commits
- resume validation and scoring lineage

Detached checkouts are recoverable, but they are no longer the intended steady
state for a healthy session.

## Study-Input Reset Rule

Treat the authoritative `lines_256` train/test NPZ pair as part of the session
contract.

If the pair changes materially, for example:

- `set_phi=False -> True`
- a different probe source or probe-scaling policy
- a different normalized `probe_transform_pipeline`
- different train/test NPZ contents behind the same compatibility paths

then:

- do not bootstrap from an older accepted-state artifact
- do not resume older sessions rooted in the previous dataset semantics
- rerun a fresh baseline
- start a new session on the regenerated pair

Earlier accepted states remain historical evidence only across that dataset
boundary.

## Session Layout

The v2 controller stores session-local state and outputs under separate roots:

- state: `state/lines_256_arch_improvement_v2/sessions/<session_id>/`
- outputs: `outputs/lines_256_arch_improvement_v2/sessions/<session_id>/`

Key artifacts:
- `session.json`
- `accepted_state.json`
- `protected_local_paths.json`
- `results.tsv`
- `proposal_context.json`
- `iterations/<n>/proposal_attempt.json`
- `iterations/<n>/proposal_result.json`
- `baseline_run_result.json`
- `iterations/<n>/candidate_context.json`
- `iterations/<n>/candidate_metadata.json`
- `iterations/<n>/candidate_assessment.json`

`session.json` should persist both:

- `session_id`
- `session_branch`

so resume can reattach a detached-but-recoverable run checkout to the correct
session branch before continuing.

`proposal_context.json` is not just a recent-attempt dump. It should include:
- the accepted state
- a short recent-history window for local context
- a broader full-session search summary so the proposal agent can detect saturated local neighborhoods and underexplored hypothesis classes
- a soft `proposal_mode` / `proposal_mode_reason` signal so the controller can steer prompt selection without hard-rejecting proposals based on heuristic hypothesis-family classification
- any active queued workflow idea from `docs/workflow_queue/active/`, with queue priority taking precedence over free-form proposal selection
- the queued idea's requested `candidate_factory` when queue frontmatter opts into a non-default candidate-production path

## Workflow Idea Queue

The v2 controller can consume a dedicated experiment-idea queue separate from
`docs/backlog/`.

Queue roots:
- `docs/workflow_queue/active/`
- `docs/workflow_queue/accepted/`
- `docs/workflow_queue/discarded/`
- `docs/workflow_queue/blocked/`
- `docs/workflow_queue/crashed/`
- `docs/workflow_queue/timed_out/`

Queue rules:
- if `active/` contains one or more Markdown files, the controller selects the
  first file in lexicographic order as the highest-priority proposal direction
- the queue item remains in `active/` while the iteration is only being
  proposed, smoked, or scored
- after a terminal result, the controller moves that same queue item to the
  outcome-specific folder
- free-form proposal selection resumes only when `active/` is empty

Queue mutation is controller-owned. Prompts may read queued idea content as
guidance, but they should not move, delete, or refile queue items.

Queue items may optionally include YAML frontmatter. In phase 1, the only
controller-owned queue frontmatter key is:

```yaml
candidate_factory: redesign
```

If omitted, the controller defaults to `candidate_factory: direct`.

Phase-1 factory policy:
- queued ideas default to the direct prompt-driven proposal path
- queued ideas may explicitly opt into the redesign subworkflow with
  `candidate_factory: redesign`
- free-form non-queued proposals still use the direct factory only

## Artifact Boundary

Critical scored artifacts are:
- launcher result / exit status
- `metrics.json`
- `randomness_contract.json`

Optional reporting artifacts are:
- `visual_publication_status.json`
- `visuals/compare_amp_phase_probe.png`
- plain compare fallbacks and session-gallery copies

The controller must score from the critical artifacts first. Optional visual
publication can add warning context, but it must not by itself turn a scored run
into `CRASH`.

## Candidate Kinds

The v2 controller supports two candidate kinds:

1. `source`
- candidate includes a source change and candidate commit
- rejected candidates reset back to the recorded accepted ref

2. `run_config`
- candidate leaves tracked source unchanged
- candidate only changes smoke/scored commands
- rejected candidates perform no git reset

## Controller Responsibilities

The controller owns deterministic study behavior:
- initialize session-local state
- capture protected local tracked-path state for the run checkout
- run and harvest baseline
- build recent-history summary and proposal context
- select the proposal prompt mode (`exploit` or `explore`) from deterministic session signals, then invoke the proposal agent and validate its candidate package
- persist controller-owned proposal attempt/result artifacts so interrupted proposal steps can be resumed or retried without corrupting session state
- run the cheap smoke command after proposal metadata validation rather than requiring the provider step to own smoke execution
- run scored candidates
- execute scored candidate commands under a controller-owned subprocess env whose
  `PATH` resolves `python` to the controller runtime, while leaving persisted
  command strings as plain `python ...`
- set `PYTHONPATH` for controller-owned smoke/scored/debug child runs to the
  session repo root instead of inheriting ambient launcher state, so source
  candidates import from the session checkout on both `start` and `resume`
- classify `KEEP`, `DISCARD`, `TIMEOUT`, and `CRASH`
- attempt one focused crash-debug retry when warranted
- update accepted state
- resume from persisted phase

Crash-debug is reserved for real execution failures or missing core scored
artifacts. Optional visual-publication problems should persist as warnings
instead of consuming crash-debug budget.

Proposal resume is also transactional:
- `proposal_running` means the provider step may have been interrupted before durable candidate metadata existed
- if `candidate_metadata.json` is missing after a proposal attempt, the controller records a retryable `proposal_result.json`, resets the session to `proposal_pending`, and lets a later resume continue instead of aborting with a missing-metadata invariant failure
- if proposal metadata already exists during `proposal_running`, the controller should reuse it and run the deterministic smoke gate rather than re-running the provider blindly

Proposal-mode selection is intentionally soft:
- the controller may choose an `explore` prompt after a local discard streak
- the controller should not reject a `READY` proposal merely because a heuristic family classifier thinks the idea is still “too local”
- exploratory versus exploitative judgment remains prompt-owned, not controller-policed

Candidate-factory selection is separate from prompt mode:
- `direct` factory: current prompt-driven proposal path for `run_config` and
  narrow `source` candidates
- `redesign` factory: queue-only subordinate workflow
  `workflows/agent_orchestration/lines_256_redesign_candidate_impl_review.yaml`
  for broader redesign candidates

Both factories must converge on the same downstream candidate package surface:
- `candidate_metadata.json`
- `proposal_result.json`

The controller remains the sole owner of:
- smoke execution
- scored execution
- keep/discard/timeout/crash classification
- accepted-state mutation
- queue movement after terminal outcome

## Timeout Semantics

Scored-run timeouts are first-class persisted outcomes, not controller-fatal
exceptions.

When a scored candidate exceeds the controller budget, v2 should still write:
- the scored candidate log
- `candidate_run_result.json`
- `candidate_assessment.json`

The controller then records the candidate as `TIMEOUT` and continues or stops
according to the normal session rules instead of dying in the timeout path.

## CLI Surface

Primary entrypoints:
- `python scripts/studies/lines_256_session_controller.py start --repo-root .`
- `python scripts/studies/lines_256_session_controller.py resume --session-root state/lines_256_arch_improvement_v2/sessions/<session_id>`

Both entrypoints keep persisted proposal/run commands as plain `python ...`,
but the controller executes smoke/scored child commands under a session-shaped
runtime env whose authoritative import root is the session repo root.

Useful options:
- `--mode baseline-only`
- `--max-iterations <n>`
- `--bootstrap-accepted-state <path>`

`--bootstrap-accepted-state` lets the v2 controller seed a fresh session from
an existing accepted-state artifact instead of rerunning baseline immediately.

Prompts remain task-local and should only prepare candidate packages or crash
repairs.
