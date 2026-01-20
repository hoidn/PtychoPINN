# Implementation Plan (Phased)

## Initiative
- ID: ORCH-ORCHESTRATOR-001
- Title: Combined orchestrator entrypoint + shared runner refactor + combined auto-commit
- Owner: user + Codex
- Spec Owner: scripts/orchestration/README.md
- Status: in_progress (Phase E — sync router review cadence parity)

## Goals
- Add a single orchestrator entrypoint that can drive both supervisor and main prompts using the router.
- Refactor shared logic out of supervisor/loop to enable reuse without changing behavior.
- Preserve sync-via-git semantics, logging conventions, and router/allowlist enforcement.
- Support local-only auto-commit in combined mode using supervisor whitelist/limits without failing on dirty state.

## Phases Overview
- Phase A — Design & Prep: finalize interfaces and test strategy; seed failing tests.
- Phase B — Implementation: shared runner refactor + orchestrator entrypoint.
- Phase C — Verification & Docs: tests, collect-only proof, and documentation updates.
- Phase D — Combined Auto-Commit: local-only auto-commit with dry-run/no-git support.

## Exit Criteria
1. New orchestrator runs in combined local mode and produces supervisor/main prompt sequence via router.
2. Sync-via-git mode works with role gating (no cross-machine contention) and preserves state semantics.
3. Review cadence runs only once per iteration in combined mode (no double reviewer runs).
4. Router override is applied only on the first actor (galph); ralph uses deterministic routing.
5. Failures stop immediately with a descriptive error message (no retries).
6. Supervisor/loop behavior remains unchanged when router is disabled.
7. Combined mode can auto-commit doc/meta, report, and tracked outputs locally using supervisor defaults; dirty non-whitelist paths only warn, no hard failures; `--commit-dry-run`/`--no-git` prevent git ops.
8. **Test coverage verified:**
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
- **Key Clauses:** State file fields + handoff semantics; router cadence/allowlist; logging conventions; doc/meta auto-commit whitelist defaults.

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
- Primary docs/specs to re-read: `scripts/orchestration/README.md`, `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`, `prompts/git_hygiene.md`
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

## Phase D — Combined Auto-Commit (local-only)
### Checklist
- [x] D0: Update test strategy for combined auto-commit (best-effort + dry-run/no-git).
      Test: N/A — planning artifact (`plans/active/ORCH-ORCHESTRATOR-001/test_strategy.md`)
- [x] D1: Add shared auto-commit helpers for doc whitelist + tracked outputs with dry-run + best-effort modes; preserve supervisor/loop semantics.
      Test: `scripts/orchestration/tests/test_orchestrator.py::test_combined_autocommit_after_turns`, `scripts/orchestration/tests/test_orchestrator.py::test_combined_autocommit_no_git`, `scripts/orchestration/tests/test_orchestrator.py::test_combined_autocommit_dry_run`
      Evidence: `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T051859Z/pytest_orchestrator.log`
- [x] D2: Add combined-mode CLI flags and defaults (`--auto-commit-*`, `--commit-dry-run`, `--no-git`) aligned with supervisor config.
      Test: `scripts/orchestration/tests/test_orchestrator.py::test_combined_autocommit_flag_plumbing`
      Evidence: `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T051859Z/pytest_orchestrator.log`
- [x] D3: Wire combined mode to run auto-commit after galph + ralph; warnings only on dirty non-whitelist paths.
      Test: `scripts/orchestration/tests/test_orchestrator.py::test_combined_autocommit_best_effort`
      Evidence: `plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T051859Z/pytest_orchestrator.log`
- [x] D4: Document combined auto-commit + flags (`scripts/orchestration/README.md`, `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`).
      Test: N/A — documentation-only

### Notes & Risks
- Ensure combined auto-commit never stages `logs/` or `tmp/`.
- Keep git ops local-only; no pulls/pushes in combined mode.
- Avoid regressions in supervisor/loop auto-commit behavior when sharing helpers.

## Phase E — Sync Review Cadence Parity
### Checklist
- [ ] E1: Add combined-mode regression tests proving reviewer cadence fires once per iteration and `last_prompt_actor` toggles appropriately when router review iterations hit.
      Test: `scripts/orchestration/tests/test_orchestrator.py::test_combined_review_last_prompt_actor`
- [ ] E2: Update orchestration testing docs (`docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`) with the new combined review cadence selector.
      Test: `pytest --collect-only scripts/orchestration/tests/test_orchestrator.py -v`
- [ ] E3: Run the new orchestrator test plus the sync router review module to confirm supervisor/loop + combined entrypoints stay in parity.
      Test: `pytest scripts/orchestration/tests/test_orchestrator.py::test_combined_review_last_prompt_actor -v`, `pytest scripts/orchestration/tests/test_sync_router_review.py -v`

### Notes & Risks
- Combined mode shares router context with supervisor/loop; missing `last_prompt_actor` annotations would reintroduce duplicate reviewer runs.
- Ensure new tests keep prompt execution fully stubbed (no external CLI).

## Artifacts Index
- Reports root: `plans/active/ORCH-ORCHESTRATOR-001/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
