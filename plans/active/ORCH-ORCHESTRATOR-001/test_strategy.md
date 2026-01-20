# Test Strategy: Combined Orchestrator

**Initiative:** ORCH-ORCHESTRATOR-001
**Phase:** Phase 0 / Pre-Implementation
**Date:** 2026-01-20
**Status:** Active

---

## 1. Framework Selection & Compatibility

### Test Framework
- **Primary Framework:** pytest
- **Rationale:** Existing orchestration tests already use pytest under `scripts/orchestration/tests/`.

### Compatibility Constraints
- [x] Framework supports parametrization (if needed)
- [x] Compatible with existing test suite patterns
- [x] No mixing of unittest.TestCase + pytest.parametrize
- [x] Fixture strategy defined (pytest fixtures)

**Decision:**
```
Use pytest in scripts/orchestration/tests/test_orchestrator.py with tmp_path fixtures and stubbed prompt execution.
```

---

## 2. CI/CD Constraint Analysis

### Environment Availability
- **Development Environment:**
  - Python version: PATH python (per PYTHON-ENV-001)
  - Available frameworks: pytest
  - Hardware: CPU

- **CI Environment:**
  - Python version: PATH python
  - Available frameworks: pytest
  - Hardware: CPU only
  - Gaps from dev: none expected

### Optional Dependency Handling
- **Pattern to use:** none (no optional dependencies)

**Constraints:**
- [x] CI limitations documented
- [x] Skip markers defined for unavailable dependencies (not needed)
- [x] Tests run in both dev and CI (with appropriate skips)

---

## 3. Test Tier Definitions

### Unit Tests
- **Scope:** Combined orchestrator state transitions, galph-only review cadence, galph-only router override, allowlist/actor gating.
- **Location:** `scripts/orchestration/tests/test_orchestrator.py`
- **Dependencies:** tmp_path + stubbed PromptExecutor only.
- **Execution time:** <1 second per test
- **Run frequency:** Every commit

### Integration Tests
- **Scope:** Orchestrator CLI wiring (argument parsing, log path selection) without external process calls.
- **Location:** `scripts/orchestration/tests/test_orchestrator.py`
- **Dependencies:** Temporary config + state files, mocked prompt executor.
- **Execution time:** <1 second per test
- **Run frequency:** Pre-merge, CI

### Smoke Tests
- **Scope:** None (avoid external LLM or git operations).

**Tier Strategy:**
```
Unit + light integration tests in a single module with full stubbing of prompt execution and git I/O.
```

---

## 4. Execution Proof Requirements

### PASSED vs SKIPPED Criteria

**What constitutes PASSED:**
- Test executed successfully
- Assertions evaluated (state transitions, router decisions, log outputs)

**Acceptable SKIP reasons:**
- None anticipated

**UNACCEPTABLE SKIP reasons:**
- Missing dependency without explicit skip
- External process requirements

### Artifact Requirements
- [ ] pytest execution log at `plans/active/ORCH-ORCHESTRATOR-001/reports/<timestamp>/pytest_orchestrator.log`
- [ ] Test summary in `summary.md` with pass/fail counts

---

## 5. Mock/Stub Strategy

### Dependencies Requiring Mocks
| Dependency | Reason | Mock Strategy |
|------------|--------|---------------|
| Prompt execution | Avoid external LLM calls | Inject PromptExecutor stub |
| Git sync | Avoid modifying repo state | Stub sync helpers or guard paths |

### Mock Implementation
```python
# Use a stub PromptExecutor that returns deterministic success.
```

**Mocking Principles:**
- [x] Mock external dependencies only (not internal routing logic)
- [x] Test real code paths where possible
- [x] Document what is mocked and why
- [x] Ensure mocks match real API contracts

---

## 6. Test Execution Plan

### Phase A: Red Tests (TDD)
1. Write failing tests for combined mode ordering + review cadence handling.
2. Run: `pytest scripts/orchestration/tests/test_orchestrator.py -v`
3. Expected: FAILED before implementation

### Phase B: Implementation
1. Implement shared runner + orchestrator entrypoint
2. Run: `pytest scripts/orchestration/tests/test_orchestrator.py -v`
3. Expected: PASSED

### Phase C: Validation
1. Run collect-only: `pytest --collect-only scripts/orchestration/tests/test_orchestrator.py -v`
2. Save logs to `plans/active/ORCH-ORCHESTRATOR-001/reports/<timestamp>/`

---

## 7. Risk Mitigation

### Identified Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Review cadence runs twice per iteration | Medium | Gate review cadence to galph in combined mode |
| Router override applied on ralph | Low | Assert ralph uses deterministic routing only |
| Tests depend on external processes | Medium | Use PromptExecutor stubs + tmp_path fixtures |

---

## 8. Success Criteria

- [ ] All tests PASS on CPU
- [ ] No external process calls in tests
- [ ] Collection proof captured
- [ ] Test logs archived

---

## 9. Approval

**Reviewed by:** TBD
**Date:** TBD
**Status:** Draft

**Notes:**
```
Pending plan approval and selector confirmation.
```

---

## References

- Test harness compatibility: `docs/DEVELOPER_GUIDE.md` Section 4.3
- Pytest selectors: `scripts/orchestration/tests/test_router.py`
- CI configuration: `.github/workflows/` or similar
