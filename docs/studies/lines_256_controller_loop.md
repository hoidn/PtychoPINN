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

This is the v2 validation path. The legacy workflow remains supported during the
parallel rollout.

- Legacy doc: `docs/studies/lines_256_arch_improvement_loop.md`
- Legacy workflow: `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml`
- V2 controller script: `scripts/studies/lines_256_session_controller.py`
- V2 thin wrapper: `workflows/agent_orchestration/lines_256_session_controller.yaml`

## Run Isolation

Like the legacy loop, this study should run in a dedicated run checkout because
the `source` candidate path still has explicit rollback/checkpoint behavior.

Do not treat the human development checkout as the live controller runtime
surface.

## Session Layout

The v2 controller stores session-local state and outputs under separate roots:

- state: `state/lines_256_arch_improvement_v2/sessions/<session_id>/`
- outputs: `outputs/lines_256_arch_improvement_v2/sessions/<session_id>/`

Key artifacts:
- `session.json`
- `accepted_state.json`
- `results.tsv`
- `proposal_context.json`
- `baseline_run_result.json`

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
- run and harvest baseline
- build recent-history summary and proposal context
- run scored candidates
- classify `KEEP`, `DISCARD`, `TIMEOUT`, and `CRASH`
- update accepted state
- resume from persisted phase

Prompts remain task-local and should only prepare candidate packages or crash
repairs.
