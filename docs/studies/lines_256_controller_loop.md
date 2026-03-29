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

## Study-Input Reset Rule

Treat the authoritative `lines_256` train/test NPZ pair as part of the session
contract.

If the pair changes materially, for example:

- `set_phi=False -> True`
- a different probe source or probe-scaling policy
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
- `baseline_run_result.json`
- `iterations/<n>/candidate_context.json`
- `iterations/<n>/candidate_metadata.json`
- `iterations/<n>/candidate_assessment.json`

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
- invoke the proposal agent and validate its candidate package
- run scored candidates
- classify `KEEP`, `DISCARD`, `TIMEOUT`, and `CRASH`
- attempt one focused crash-debug retry when warranted
- update accepted state
- resume from persisted phase

Crash-debug is reserved for real execution failures or missing core scored
artifacts. Optional visual-publication problems should persist as warnings
instead of consuming crash-debug budget.

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

Useful options:
- `--mode baseline-only`
- `--max-iterations <n>`
- `--bootstrap-accepted-state <path>`

`--bootstrap-accepted-state` lets the v2 controller seed a fresh session from
an existing accepted-state artifact instead of rerunning baseline immediately.

Prompts remain task-local and should only prepare candidate packages or crash
repairs.
