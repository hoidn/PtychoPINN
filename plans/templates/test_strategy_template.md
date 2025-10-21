# Test Strategy: [Initiative Name]

**Initiative:** [INITIATIVE-ID]
**Phase:** Phase 0 / Pre-Implementation
**Date:** [YYYY-MM-DD]
**Status:** [Draft/Approved]

---

## 1. Framework Selection & Compatibility

### Test Framework
- **Primary Framework:** [pytest / unittest / other]
- **Rationale:** [Why this framework was chosen]

### Compatibility Constraints
- [ ] Framework supports parametrization (if needed)
- [ ] Compatible with existing test suite patterns
- [ ] No mixing of unittest.TestCase + pytest.parametrize
- [ ] Fixture strategy defined (pytest fixtures / unittest setUp)

**Decision:**
```
[Document framework selection decision and any compatibility concerns]
```

---

## 2. CI/CD Constraint Analysis

### Environment Availability
- **Development Environment:**
  - Python version: [e.g., 3.10+]
  - Available frameworks: [e.g., pytest, torch, tensorflow]
  - Hardware: [CPU/GPU/TPU]

- **CI Environment:**
  - Python version: [e.g., 3.10+]
  - Available frameworks: [e.g., pytest, tensorflow only]
  - Hardware: [CPU only]
  - Gaps from dev: [e.g., PyTorch unavailable, no GPU]

### Optional Dependency Handling
- **Pattern to use:** [torch-optional harness / skip markers / mock]
- **Implementation:**
  ```python
  # Example pattern for handling optional dependencies
  try:
      import torch
      TORCH_AVAILABLE = True
  except ImportError:
      TORCH_AVAILABLE = False

  @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
  def test_pytorch_feature():
      ...
  ```

**Constraints:**
- [ ] CI limitations documented
- [ ] Skip markers defined for unavailable dependencies
- [ ] Tests run in both dev and CI (with appropriate skips)

---

## 3. Test Tier Definitions

### Unit Tests
- **Scope:** [Individual functions/classes in isolation]
- **Location:** `tests/unit/` or `tests/[module]/test_*.py`
- **Dependencies:** Minimal (mocks/stubs for external dependencies)
- **Execution time:** <1 second per test
- **Run frequency:** Every commit

### Integration Tests
- **Scope:** [Component interactions, data flow]
- **Location:** `tests/integration/` or `tests/[module]/test_*_integration.py`
- **Dependencies:** [Real components, test data]
- **Execution time:** 1-10 seconds per test
- **Run frequency:** Pre-merge, CI

### Smoke Tests
- **Scope:** [End-to-end critical paths]
- **Location:** `tests/smoke/` or marked with `@pytest.mark.smoke`
- **Dependencies:** [Full system, datasets]
- **Execution time:** >10 seconds per test
- **Run frequency:** Pre-release, nightly

**Tier Strategy:**
```
[Explain which tiers apply to this initiative and why]
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
  - Marker: `@pytest.mark.gpu`
- [ ] External service unavailable (integration tests)
  - Marker: `@pytest.mark.requires_service`
- [ ] Long-running tests (>5 min benchmarks)
  - Marker: `@pytest.mark.slow`
- [ ] Optional dependency unavailable (documented pattern)
  - Skip condition: `@pytest.mark.skipif(not TORCH_AVAILABLE, ...)`

**UNACCEPTABLE SKIP reasons:**
- Missing optional dependency without proper torch-optional pattern
- Framework incompatibility (design error)
- "Tests don't work in CI" without fixing root cause

### Artifact Requirements
- [ ] pytest execution log at `plans/active/<initiative>/reports/<timestamp>/pytest.log`
- [ ] Test summary in `summary.md` with counts: X passed, Y skipped, Z failed
- [ ] Explicit justification for each SKIPPED test in summary
- [ ] Proof assertions actually executed (check log output)

---

## 5. Mock/Stub Strategy

### Dependencies Requiring Mocks
| Dependency | Reason | Mock Strategy |
|------------|--------|---------------|
| [e.g., GPU] | Unavailable in CI | Use CPU fallback in tests |
| [e.g., External API] | Not available in test | Mock with `unittest.mock` |
| [e.g., Large dataset] | Too large for CI | Use small test fixtures |

### Mock Implementation
```python
# Example mock strategy
from unittest.mock import Mock, patch

@patch('module.external_dependency')
def test_with_mock(mock_dep):
    mock_dep.return_value = expected_value
    result = function_under_test()
    assert result == expected
```

**Mocking Principles:**
- [ ] Mock external dependencies only (not internal logic)
- [ ] Test real code paths where possible
- [ ] Document what is mocked and why
- [ ] Ensure mocks match real API contracts

---

## 6. Test Execution Plan

### Phase A: Red Tests (TDD)
1. **Write failing tests first** for new functionality
2. Verify tests fail for the right reason
3. Run: `pytest tests/[module]/test_new_feature.py -v`
4. Expected: FAILED (not SKIPPED, not ERROR)

### Phase B: Implementation
1. Implement minimal code to pass tests
2. Run: `pytest tests/[module]/test_new_feature.py -v`
3. Expected: PASSED (not SKIPPED)

### Phase C: Validation
1. Run full test suite: `pytest tests/`
2. Verify no regressions
3. Check coverage: `pytest --cov=module tests/`
4. Save logs to `plans/active/<initiative>/reports/<timestamp>/`

---

## 7. Risk Mitigation

### Identified Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| [e.g., Tests skip in CI due to PyTorch] | High - false confidence | Implement torch-optional pattern from start |
| [e.g., Framework incompatibility] | High - requires refactor | Validate pytest-only approach before implementation |
| [e.g., Slow tests block CI] | Medium - CI timeout | Mark as @pytest.mark.slow, run separately |

---

## 8. Success Criteria

- [ ] All tests PASS (not SKIP) in development environment
- [ ] Tests run in CI (with documented skips for env limitations)
- [ ] Test execution log proves assertions ran
- [ ] Zero test refactoring iterations needed
- [ ] No unittest/pytest mixing issues
- [ ] Coverage meets project standards (>80% for new code)

---

## 9. Approval

**Reviewed by:** [Supervisor/Engineer]
**Date:** [YYYY-MM-DD]
**Status:** [Approved / Needs Revision]

**Notes:**
```
[Any review comments or required changes]
```

---

## References

- Test harness compatibility: `docs/DEVELOPER_GUIDE.md` Section 4.3
- Torch-optional pattern: `tests/conftest.py`
- Pytest markers: `pytest.ini`
- CI configuration: `.github/workflows/` or similar
