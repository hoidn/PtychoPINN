# Test Strategy: Router Prompt Orchestration

**Initiative:** ORCH-ROUTER-001
**Phase:** Phase 0 / Pre-Implementation
**Date:** 2026-01-20
**Status:** Draft

---

## 1. Framework Selection & Compatibility

### Test Framework
- **Primary Framework:** pytest
- **Rationale:** Existing tooling tests in tests/tools use pytest-only patterns.

### Compatibility Constraints
- [ ] Framework supports parametrization (if needed)
- [ ] Compatible with existing test suite patterns
- [ ] No mixing of unittest.TestCase + pytest.parametrize
- [ ] Fixture strategy defined (pytest fixtures)

**Decision:**
```
Use pytest in tests/tools/test_orchestration_router.py. No unittest.TestCase usage.
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
- **Pattern to use:** none (no optional deps)

**Constraints:**
- [ ] CI limitations documented
- [ ] Skip markers defined for unavailable dependencies
- [ ] Tests run in both dev and CI (with appropriate skips)

---

## 3. Test Tier Definitions

### Unit Tests
- **Scope:** Router decision logic, allowlist validation, override precedence.
- **Location:** tests/tools/test_orchestration_router.py
- **Dependencies:** Temporary files (state.json, fake prompts) only.
- **Execution time:** <1 second per test
- **Run frequency:** Every commit

### Integration Tests
- **Scope:** End-to-end router dispatch with temp state + config + prompt output.
- **Location:** tests/tools/test_orchestration_router.py
- **Dependencies:** Temporary files only (no external binaries)
- **Execution time:** <1 second per test
- **Run frequency:** Pre-merge, CI

### Smoke Tests
- **Scope:** None required (router does not invoke external processes in tests).

**Tier Strategy:**
```
Unit and light integration tests in a single module. No slow or external-process tests.
```

---

## 4. Execution Proof Requirements

### PASSED vs SKIPPED Criteria

**What constitutes PASSED:**
- Test executed successfully
- Assertions evaluated (routing decisions, errors, logs)

**Acceptable SKIP reasons:**
- None anticipated

**UNACCEPTABLE SKIP reasons:**
- Missing dependency without explicit skip
- Skipping due to external binaries or network calls

### Artifact Requirements
- [ ] pytest execution log at .artifacts/orch-router-001/pytest_router.log
- [ ] Test summary in summary.md with pass/fail counts

---

## 5. Mock/Stub Strategy

### Dependencies Requiring Mocks
| Dependency | Reason | Mock Strategy |
|------------|--------|---------------|
| File system paths | Avoid touching real repo files | tmp_path fixtures |
| time/iteration | Deterministic routing | fixed values in state.json |

### Mock Implementation
```python
# Use tmp_path to create state.json and prompt files.
```

**Mocking Principles:**
- [ ] Mock external dependencies only (not internal logic)
- [ ] Test real code paths where possible
- [ ] Document what is mocked and why
- [ ] Ensure mocks match real API contracts

---

## 6. Test Execution Plan

### Phase A: Red Tests (TDD)
1. Write failing tests for deterministic routing + override
2. Run: pytest tests/tools/test_orchestration_router.py -v
3. Expected: FAILED before implementation

### Phase B: Implementation
1. Implement router logic
2. Run: pytest tests/tools/test_orchestration_router.py -v
3. Expected: PASSED

### Phase C: Validation
1. Run collect-only: pytest --collect-only tests/tools/test_orchestration_router.py -v
2. Save logs to .artifacts/orch-router-001/

---

## 7. Risk Mitigation

### Identified Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Router tests invoke external binaries | Medium | Keep tests file-based only |
| Routing rules drift from README | Medium | Update README in Phase A and assert behavior in tests |

---

## 8. Success Criteria

- [ ] All router tests PASS on CPU
- [ ] No external process calls in tests
- [ ] Collect-only proves test discoverability
