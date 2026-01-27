# Test Strategy: Grid Lines Workflow

**Initiative:** GRID-LINES-WORKFLOW-001
**Phase:** Phase 0 / Pre-Implementation
**Date:** 2026-01-27
**Status:** Draft

---

## 1. Framework Selection & Compatibility

### Test Framework
- **Primary Framework:** pytest
- **Rationale:** Matches existing repo conventions and supports simple unit tests for helper functions.

### Compatibility Constraints
- [x] Framework supports parametrization (if needed)
- [x] Compatible with existing test suite patterns
- [x] No mixing of unittest.TestCase + pytest.parametrize
- [x] Fixture strategy defined (simple pytest fixtures / tmp_path)

**Decision:**
```
Use pytest with small, deterministic unit tests only. No integration or heavy simulation tests
in this initiative; those are handled by manual workflow runs documented in the plan.
```

---

## 2. CI/CD Constraint Analysis

### Environment Availability
- **Development Environment:**
  - Python version: PATH `python` (per PYTHON-ENV-001)
  - Available frameworks: pytest, tensorflow, torch (mandatory)
  - Hardware: CPU (GPU not required for unit tests)

- **CI Environment:**
  - Python version: PATH `python`
  - Available frameworks: pytest, tensorflow, torch
  - Hardware: CPU only
  - Gaps from dev: None expected for unit tests

### Optional Dependency Handling
- **Pattern to use:** None (torch is mandatory per POLICY-001)
- **Implementation:** N/A

**Constraints:**
- [x] CI limitations documented
- [x] Skip markers defined for unavailable dependencies (not needed here)
- [x] Tests run in both dev and CI (with appropriate skips)

---

## 3. Test Tier Definitions

### Unit Tests
- **Scope:** Probe scaling, dataset path layout, stitching helper logic
- **Location:** `tests/test_grid_lines_workflow.py`
- **Dependencies:** NumPy, pytest
- **Execution time:** <1s per test
- **Run frequency:** Every commit

### Integration Tests
- **Scope:** None for this initiative (manual workflow runs only)
- **Location:** N/A
- **Dependencies:** N/A
- **Execution time:** N/A
- **Run frequency:** N/A

### Smoke Tests
- **Scope:** None (manual workflow runs captured separately)
- **Location:** N/A
- **Dependencies:** N/A
- **Execution time:** N/A
- **Run frequency:** N/A

**Tier Strategy:**
```
Unit tests only. Heavy end-to-end runs are executed manually via the CLI after implementation
and recorded as workflow artifacts, not as automated tests.
```

---

## 4. Execution Proof Requirements

### PASSED vs SKIPPED Criteria

**What constitutes PASSED:**
- Test executed successfully
- All assertions evaluated
- Expected behavior verified
- Not just import checks

**Acceptable SKIP reasons:**
- [ ] Hardware unavailable (GPU tests on CPU-only CI)
- [ ] External service unavailable (integration tests)
- [ ] Long-running tests (>5 min benchmarks)
- [ ] Optional dependency unavailable (not applicable here)

**UNACCEPTABLE SKIP reasons:**
- Missing optional dependency without proper pattern
- Framework incompatibility
- “Tests don’t work in CI” without fixing root cause

### Artifact Requirements
- [ ] pytest execution log stored under `.artifacts/` (no timestamped report dirs)
- [ ] Test summary in `plans/active/GRID-LINES-WORKFLOW-001/summary.md`
- [ ] Explicit justification for each SKIPPED test in summary (if any)
- [ ] Proof assertions actually executed (check log output)

---

## 5. Mock/Stub Strategy

### Dependencies Requiring Mocks
| Dependency | Reason | Mock Strategy |
|------------|--------|---------------|
| None | Unit tests are pure NumPy logic | N/A |

### Mock Implementation
```python
# No mocks needed for current test scope
```

**Mocking Principles:**
- [x] Mock external dependencies only (not internal logic)
- [x] Test real code paths where possible
- [x] Document what is mocked and why
- [x] Ensure mocks match real API contracts

---

## 6. Test Execution Plan

### Phase A: Red Tests (TDD)
1. Write failing tests first for new helpers
2. Verify tests fail for the right reason
3. Run: `pytest tests/test_grid_lines_workflow.py -v`
4. Expected: FAILED (not SKIPPED, not ERROR)

### Phase B: Implementation
1. Implement minimal code to pass tests
2. Run: `pytest tests/test_grid_lines_workflow.py -v`
3. Expected: PASSED (not SKIPPED)

### Phase C: Validation
1. Run focused unit suite: `pytest tests/test_grid_lines_workflow.py -v`
2. Verify no regressions in that scope
3. Save logs to `.artifacts/` and link from summary

---

## 7. Risk Mitigation

### Identified Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Unit tests too narrow | Medium | Add coverage for stitch helper and probe scaling edge cases |
| Manual runs not documented | Medium | Record CLI runs + outputs in summary |

---

## 8. Success Criteria

- [ ] All unit tests PASS in development environment
- [ ] Tests run in CI (no skips expected)
- [ ] Test execution log proves assertions ran
- [ ] No unittest/pytest mixing issues
- [ ] Coverage sufficient for new helper functions

---
