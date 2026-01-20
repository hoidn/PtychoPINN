# Minimal Design: Combined Orchestrator (Router-Driven)

## Goals
- Provide a single entrypoint that can route and execute both `supervisor.md` and `main.md` prompts.
- Support both local (single-machine) combined loops and sync-via-git workflows.
- Reuse existing supervisor/loop logic via shared runner utilities.
- Preserve `sync/state.json` semantics and router allowlist/actor gating.

## Proposed Components
- `scripts/orchestration/runner.py` (new shared module)
  - `PromptExecutor`: injectable callable for running a prompt (used by tests to avoid external processes).
  - `RouterContext`: prompt map, allowlist, review cadence, router mode/output, prompts_dir.
  - `select_prompt(state, ctx)` wrapper around `select_prompt_with_mode`.
  - `run_turn(role, state, ctx, exec, logger)`:
    - resolve prompt path
    - execute prompt
    - log decision
    - return updated state + `last_prompt` annotation
- `scripts/orchestration/orchestrator.py` (new entrypoint)
  - CLI supports `--mode combined|role` (default combined for local use) and `--role galph|ralph` for sync-via-git.
  - Combined mode runs two turns per iteration (galph then ralph) using shared runner.
  - Sync mode runs only when `state.expected_actor == role` to avoid cross-machine contention.
- Wrapper: `orchestrator.sh` (parallels `supervisor.sh`/`loop.sh`).

## State & Router Semantics
- Combined/local mode:
  - Runs both actors sequentially on a single machine (galph then ralph).
  - Applies review cadence only once per iteration (galph turn only).
  - Router override is allowed only on the first actor (galph); ralph uses deterministic routing.
  - Increments `iteration` only after the ralph turn succeeds (matching existing semantics).
- Sync-via-git mode:
  - Reuses existing git pull/push guards and state stamping.
  - Role-gated execution: only runs when `expected_actor` matches the configured role.
  - Failure stops immediately with a descriptive error message.

## Logging & Artifacts
- Preserve current log naming and summary conventions under `logs_dir` (galph/ralph prefixes).
- Router decisions logged via existing `log_router_decision` helper.

## Testability Hooks
- `PromptExecutor` injection allows unit tests to stub prompt execution.
- Runner returns structured results for state transitions to enable deterministic assertions.
