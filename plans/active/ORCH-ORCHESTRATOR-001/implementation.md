# Implementation Plan (Phased)

## Initiative
- ID: ORCH-ORCHESTRATOR-001
- Title: Combined orchestrator entrypoint + shared runner refactor
- Owner: user + Codex
- Spec Owner: scripts/orchestration/README.md
- Status: in_progress (Phase C tests complete; full-suite regression gate pending)

## Goals
- Add a single orchestrator entrypoint that can drive both supervisor and main prompts using the router.
- Refactor shared logic out of supervisor/loop to enable reuse without changing behavior.
- Preserve sync-via-git semantics, logging conventions, and router/allowlist enforcement.

## Phases Overview
- Phase A — Design & Prep: finalize interfaces and test strategy; seed failing tests.
- Phase B — Implementation: shared runner refactor + orchestrator entrypoint.
- Phase C — Verification & Docs: tests, collect-only proof, and documentation updates.

## Exit Criteria
1. New orchestrator runs in combined local mode and produces supervisor/main prompt sequence via router.
2. Sync-via-git mode works with role gating (no cross-machine contention) and preserves state semantics.
3. Review cadence runs only once per iteration in combined mode (no double reviewer runs).
4. Router override is applied only on the first actor (galph); ralph uses deterministic routing.
5. Failures stop immediately with a descriptive error message (no retries).
6. Supervisor/loop behavior remains unchanged when router is disabled.
7. **Test coverage verified:**
   - All cited selectors collect >0 tests (`pytest --collect-only`)
   - All cited selectors pass
   - No regression in existing test suite (full suite green or known-skip documented)
   - Test registry synchronized: `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` updated
   - Logs saved to `plans/active/ORCH-ORCHESTRATOR-001/reports/<timestamp>/`

## Compliance Matrix (Mandatory)
- [x] **Spec Constraint:** `scripts/orchestration/README.md` — state contract, router cadence, logging
- [x] **Fix-Plan Link:** `docs/fix_plan.md` — ORCH-ORCHESTRATOR-001
- [x] **Finding/Policy ID:** PYTHON-ENV-001 (docs/DEVELOPER_GUIDE.md)
- [x] **Test Strategy:** `plans/active/ORCH-ORCHESTRATOR-001/test_strategy.md`

## Spec Alignment
- **Normative Spec:** `scripts/orchestration/README.md`
- **Key Clauses:** State file fields + handoff semantics; router cadence/allowlist; logging conventions.

## Testing Integration

**Principle:** Every checklist item that adds or modifies observable behavior MUST specify its test artifact.

**Format for checklist items:**
```
- [ ] <ID>: <implementation task>
      Test: <pytest selector> | N/A: <justification>
```

**Complex testing needs:** See `plans/active/ORCH-ORCHESTRATOR-001/test_strategy.md`.

## Architecture / Interfaces (minimal design)
- **Key Data Types / Protocols:**
  - `OrchestrationState` (state.json), `RouterDecision`, `PromptExecutor` (stubbed in tests), `OrchestratorMode` (combined|role).
- **Boundary Definitions:**
  - `[orchestrator] -> [runner] -> [router] -> [prompt executor] -> [state writer]`.
- **Sequence Sketch (Happy Path):**
  - Combined mode: load state -> select prompt (router) -> run prompt -> update state -> repeat for second actor -> persist state.
  - Sync mode: load state -> if expected_actor matches role -> select + run prompt -> update state -> push.
- **Data-Flow Notes:**
  - `sync/state.json` + `orchestration.yaml` → router → prompt path → subprocess execution → updated state + logs.

## Context Priming (read before edits)
- Primary docs/specs to re-read: `scripts/orchestration/README.md`, `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`
- Required findings/case law: PYTHON-ENV-001 (use `python` from PATH)
- Related telemetry/attempts: ORCH-ROUTER-001 artifacts in `.artifacts/orch-router-001/`
- Data dependencies to verify: `orchestration.yaml`, `sync/state.json`, `prompts/` (supervisor/main/reviewer/router), `logs/`

## Phase A — Design & Prep
### Checklist
- [x] A0: Draft minimal design + interfaces (runner/orchestrator) and confirm review cadence behavior.
      Test: N/A — design artifact (`plans/active/ORCH-ORCHESTRATOR-001/design.md`)
- [x] A1: Create test strategy + link in fix plan.
      Test: N/A — planning artifact (`plans/active/ORCH-ORCHESTRATOR-001/test_strategy.md`)
- [x] A2: Add failing tests for combined-mode ordering and review cadence (TDD nucleus).
      Test: `scripts/orchestration/tests/test_orchestrator.py::test_combined_sequence`
      Evidence: `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T032929Z/pytest_orchestrator.log`

### Dependency Analysis (Required for Refactors)
- **Touched Modules:** `scripts/orchestration/supervisor.py`, `scripts/orchestration/loop.py`, `scripts/orchestration/router.py`, new `scripts/orchestration/runner.py`, new `scripts/orchestration/orchestrator.py`
- **Circular Import Risks:** Avoid importing supervisor/loop inside runner; runner should depend on router/state/config only.
- **State Migration:** Preserve existing state fields; no schema changes.

### Notes & Risks
- Review cadence currently fires on both actors per iteration; combined mode must avoid double reviewer runs.

## Phase B — Implementation
### Checklist
- [x] B1: Extract shared prompt execution + router selection into `runner.py`; update supervisor/loop to call shared helper.
      Test: `scripts/orchestration/tests/test_router.py::test_router_deterministic`, `scripts/orchestration/tests/test_orchestrator.py::test_router_disabled_uses_actor_prompts`
      Evidence: `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T032929Z/pytest_orchestrator.log`
- [x] B2: Implement `orchestrator.py` CLI with combined mode + role-gated sync mode; add wrapper script.
      Test: `scripts/orchestration/tests/test_orchestrator.py::test_combined_sequence`
      Evidence: `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T032929Z/pytest_orchestrator.log`
- [x] B3: Enforce galph-only review cadence + router override (ralph deterministic only).
      Test: `scripts/orchestration/tests/test_orchestrator.py::test_review_cadence_single`, `scripts/orchestration/tests/test_orchestrator.py::test_router_override_galph_only`
      Evidence: `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T032929Z/pytest_orchestrator.log`

### Notes & Risks
- Ensure sync-via-git guardrails mirror supervisor/loop (pull/push + dirty-tree handling).

## Phase C — Verification & Docs
### Checklist
- [x] C1: Complete orchestrator tests + collect-only verification.
      Test: `pytest --collect-only scripts/orchestration/tests/test_orchestrator.py -v`
      Evidence: `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T034858Z/pytest_collect_orchestrator.log`
- [x] C2: Run orchestrator test module.
      Test: `pytest scripts/orchestration/tests/test_orchestrator.py -v`
      Evidence: `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T034858Z/pytest_orchestrator.log`
- [x] C3: Update docs (`scripts/orchestration/README.md`, `docs/index.md`, `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`).
      Test: N/A — documentation-only

### Notes & Risks
- Keep tests CPU-only and avoid invoking external LLM binaries.

## Artifacts Index
- Reports root: `plans/active/ORCH-ORCHESTRATOR-001/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
