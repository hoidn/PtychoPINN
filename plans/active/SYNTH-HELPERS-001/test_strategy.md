# Test Strategy: Script-level synthetic simulation helpers

**Initiative:** SYNTH-HELPERS-001  
**Phase:** Phase A / Pre-Implementation  
**Date:** 2026-01-13  
**Status:** Approved

---

## 1. Framework Selection & Compatibility

### Test Framework
- **Primary Framework:** pytest
- **Rationale:** Matches existing suite patterns and fixtures.

### Compatibility Constraints
- [x] Framework supports parametrization (if needed)
- [x] Compatible with existing test suite patterns
- [x] No mixing of unittest.TestCase + pytest.parametrize
- [x] Fixture strategy defined (pytest fixtures)

**Decision:**
```
Use pytest-only modules under tests/scripts to match existing CLI/test helper coverage.
```

---

## 2. CI/CD Constraint Analysis

### Environment Availability
- **Development Environment:**
  - Python version: 3.11+
  - Available frameworks: pytest, tensorflow
  - Hardware: CPU

- **CI Environment:**
  - Python version: 3.10+
  - Available frameworks: pytest, tensorflow
  - Hardware: CPU
  - Gaps from dev: none expected for this scope

### Optional Dependency Handling
- **Pattern to use:** None (tests do not require optional deps)
- **Implementation:** N/A

**Constraints:**
- [x] CI limitations documented
- [x] Skip markers defined for unavailable dependencies (not needed)
- [x] Tests run in both dev and CI (with appropriate skips)

---

## 3. Test Tier Definitions

### Unit Tests
- **Scope:** Helper functions (object/probe creation, nongrid simulation, splitting)
- **Location:** `tests/scripts/test_synthetic_helpers.py`
- **Dependencies:** NumPy, TensorFlow (via simulation path)
- **Execution time:** < 5 seconds
- **Run frequency:** Every commit touching helpers

### Smoke Tests
- **Scope:** CLI `--help` for study/simulation scripts, pipeline import
- **Location:** `tests/scripts/test_synthetic_helpers_cli_smoke.py`
- **Dependencies:** Subprocess invocation only
- **Execution time:** < 5 seconds
- **Run frequency:** Every commit touching scripts

**Tier Strategy:**
```
Unit tests validate helper behavior; CLI smoke tests ensure entrypoints remain callable.
```

---

## 4. Execution Proof Requirements

### PASSED vs SKIPPED Criteria

**What constitutes PASSED:**
- Test executed successfully
- Assertions evaluated
- Helper outputs and CLI exit codes verified

**Acceptable SKIP reasons:**
- None expected for this scope

**UNACCEPTABLE SKIP reasons:**
- Missing TensorFlow dependency
- Import failures in helper modules

### Artifact Requirements
- [x] pytest execution log at `plans/active/SYNTH-HELPERS-001/reports/<timestamp>/pytest_synthetic_helpers.log`
- [x] pytest collect-only log at `plans/active/SYNTH-HELPERS-001/reports/<timestamp>/pytest_collect.log`
- [x] Static analysis log at `plans/active/SYNTH-HELPERS-001/reports/<timestamp>/ruff_check.log`

---

## 5. Mock/Stub Strategy

### Dependencies Requiring Mocks
| Dependency | Reason | Mock Strategy |
|------------|--------|---------------|
| CLI scripts | Avoid heavy runs | Use `--help` only |

### Mock Implementation
```
No additional mocking required beyond CLI help calls.
```

---

## 6. Test Execution Plan

### Phase A: Red Tests (TDD)
1. Author helper tests and CLI smoke tests
2. Verify selectors collect >0 tests

### Phase B: Implementation
1. Implement helper module + refactors
2. Run: `pytest tests/scripts/test_synthetic_helpers.py -v`
3. Run: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v`

### Phase C: Validation
1. Run collect-only on the new selectors
2. Save logs to `plans/active/SYNTH-HELPERS-001/reports/<timestamp>/`

---

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| TensorFlow simulation path slow | Medium | Use small N/object sizes in tests |
| CLI smoke tests accidentally execute heavy runs | Medium | Use `--help` only |

---

## 8. Success Criteria

- [x] Helper tests pass on CPU
- [x] CLI help smoke tests pass
- [x] Collect-only confirms test discovery
- [x] Logs archived under reports directory

---

## 9. Approval

**Reviewed by:** Codex  
**Date:** 2026-01-13  
**Status:** Approved

**Notes:**
```
Aligned with TESTING_GUIDE and scripts/ helper coverage requirements.
```

---

## References

- `docs/TESTING_GUIDE.md`
- `docs/development/TEST_SUITE_INDEX.md`
