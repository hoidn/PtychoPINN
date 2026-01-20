# Test Strategy: DOC-HYGIENE-20260120

**Initiative:** DOC-HYGIENE-20260120  
**Phase:** Phase 0 / Pre-Implementation  
**Date:** 2026-01-20  
**Status:** Draft

---

## 1. Framework Selection & Compatibility

### Test Framework
- **Primary Framework:** pytest
- **Rationale:** Orchestration code already ships pytest suites under `scripts/orchestration/tests/`. Adding supervisor `--no-git` coverage with pytest keeps parity with existing router/orchestrator tests and leverages pytest fixtures/monkeypatch for git stubs.

### Compatibility Constraints
- [x] Framework supports parametrization/fixtures (pytest)
- [x] Matches current orchestration test pattern (module-local tests under scripts/orchestration/tests/)
- [x] No unittest / pytest mixing required
- [x] Git operations will be stubbed with monkeypatch; no real repo mutations

**Decision:**
```
Extend scripts/orchestration/tests/ with a new test_supervisor.py module.
Use pytest + monkeypatch to stub git bus helpers (safe_pull, add, commit, push_with_rebase, etc.)
and assert that "--no-git" suppresses them while still running prompt execution paths.
Existing selectors (test_router.py, test_agent_dispatch.py, test_orchestrator.py) re-run
to guard config + combined-mode behavior after edits.
```

---

## 2. CI/CD Constraint Analysis

### Environment Availability
- **Development Environment:**
  - Python 3.10+
  - pytest available
  - Hardware: CPU-only (sufficient)
  - Git present locally, but tests will stub git commands so they succeed even without git.
- **CI Environment:**
  - Python 3.10+
  - pytest available
  - Hardware: CPU-only
  - Git CLI available but not required at runtime because tests stub git helpers.

### Optional Dependency Handling
- No GPU/torch dependencies involved.
- Tests should monkeypatch git helper functions exported by `scripts.orchestration.git_bus` to avoid touching the real repo.

**Constraints:**
- [x] Skip markers not required (tests CPU-only)
- [x] All selectors run in CI (no optional dependency gates)

---

## 3. Test Tier Definitions

### Unit Tests
- **Scope:** Supervisor CLI argument handling + git guard logic, config loader reading orchestration.yaml.
- **Location:** `scripts/orchestration/tests/test_supervisor.py`, `scripts/orchestration/tests/test_router.py`.
- **Dependencies:** monkeypatched git helpers / fake configs, no filesystem mutation beyond temp dirs.
- **Execution time:** < 1s per test.
- **Run frequency:** Every commit / before MR.

### Integration Tests
- **Scope:** Combined orchestrator behavior already covered by `test_orchestrator.py`; re-run to verify no regressions.
- **Location:** `scripts/orchestration/tests/test_orchestrator.py`.
- **Dependencies:** Local temp dirs, no real git operations when `--no-git` is set.
- **Execution time:** ~10s total.
- **Run frequency:** Upon orchestrator/supervisor changes.

### Smoke Tests
- Not applicable. Full CLI loops are out of scope for this hygiene initiative.

**Tier Strategy:**
```
Add targeted unit tests for supervisor --no-git gating and keep existing orchestrator/router
tests as integration coverage. No new smoke tests; CLI end-to-end runs still require
manual supervision and are covered by higher-level initiatives.
```

---

## 4. Execution Proof Requirements

- pytest logs stored under `plans/active/DOC-HYGIENE-20260120/reports/<timestamp>/`:
  - `pytest_supervisor_no_git.log`
  - `pytest_orchestrator.log`
  - `pytest_router.log` (if rerun)
- Collect-only proof for new selector: `pytest --collect-only scripts/orchestration/tests/test_supervisor.py -q`.
- Ensure tests emit PASS (not SKIP) on dev + CI; any intentionally skipped tests require explicit justification (none expected).

---

## 5. Mock/Stub Strategy

| Dependency | Reason | Mock Strategy |
|------------|--------|---------------|
| `scripts.orchestration.git_bus.safe_pull/add/commit/push_with_rebase` | Need to prove "--no-git" bypasses git operations without touching real repo | monkeypatch these functions with fakes that record calls; assert they are not invoked when `--no-git` set |
| `scripts.orchestration.autocommit.autocommit_reports` + `_supervisor_autocommit_*` | Avoid real staging/commits while verifying gating | monkeypatch to record invocations; assert gating when `--no-git` |
| `tee_run` | Avoid launching actual prompts during tests | monkeypatch to record arguments / simulate RC=0 |

Mocking principles:
- Only mock external side-effectful helpers (git, file I/O). Leave supervisor control flow intact.
- Use pytest `monkeypatch` fixture to restore originals automatically.

---

## 6. Test Execution Plan

### Phase A (Doc alignment)
- No test impact; documentation-only edits.

### Phase B (orchestration.yaml)
1. Add file + doc references.
2. Run `pytest scripts/orchestration/tests/test_router.py::test_router_config_loads -v`.
3. Run `pytest scripts/orchestration/tests/test_agent_dispatch.py -v` (sanity).

### Phase C (supervisor guard)
1. Add/modify tests in `scripts/orchestration/tests/test_supervisor.py`.
2. Run collect-only: `pytest --collect-only scripts/orchestration/tests/test_supervisor.py -q`.
3. Run targeted suite: `pytest scripts/orchestration/tests/test_supervisor.py -v`.
4. Re-run orchestrator regression: `pytest scripts/orchestration/tests/test_orchestrator.py::test_combined_autocommit_no_git -v`.

Each run's log saved under the current reports timestamp.

---

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tests accidentally invoke real git operations | Could dirty the repo or fail on hosts without git | Monkeypatch git bus helpers and assert they were not called; use temp dirs for config fixtures |
| Supervisor CLI output depends on prompts that invoke external CLIs | Tests would hang waiting for LLM CLI | Monkeypatch `tee_run` to simulate the prompt execution path with a no-op |
| YAML config drift vs README | Tests pass but docs mislead reviewers | Phase B checklist includes README/docs/index updates + manual review |

---

## 8. Success Criteria

- New supervisor tests pass locally and in CI.
- Existing router/orchestrator/agent dispatch tests remain green (no regressions).
- pytest logs + collect-only evidence archived under the initiative reports hub.
- Docs/test registries updated to mention the new selector(s).
