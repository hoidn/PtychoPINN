# Test Strategy: GPU-Only Execution

**Initiative:** FIX-DEVICE-TOGGLE-001
**Phase:** Phase 0 / Pre-Implementation
**Date:** 2026-01-20
**Status:** Draft

---

## 1. Framework Selection & Compatibility

### Test Framework
- **Primary Framework:** pytest
- **Rationale:** Existing CLI/config tests are pytest-based and already cover execution config behavior.

### Compatibility Constraints
- [x] Framework supports parametrization
- [x] Compatible with existing test suite patterns
- [x] No mixing of unittest.TestCase + pytest.parametrize
- [x] Fixture strategy defined (pytest fixtures)

**Decision:**
```
Use pytest for all updates. Reuse existing CLI/config test modules and fixtures.
```

---

## 2. CI/CD Constraint Analysis

### Environment Availability
- **Development Environment:**
  - Python version: project default
  - Available frameworks: pytest, torch
  - Hardware: GPU required (CUDA)

- **CI Environment:**
  - Python version: project default
  - Available frameworks: pytest, torch
  - Hardware: GPU preferred; CPU-only CI must skip GPU-only execution tests

### Optional Dependency Handling
- **Pattern to use:** GPU availability guard / skip markers
- **Implementation:**
  ```python
  import torch
  import pytest

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
  def test_gpu_only_behavior():
      ...
  ```

**Constraints:**
- [x] CI limitations documented
- [x] Skip markers defined for missing CUDA
- [x] Tests run in dev + CI with documented skips

---

## 3. Test Tier Definitions

### Unit Tests
- **Scope:** CLI parsing + execution config validation (no actual training)
- **Location:** `tests/torch/test_cli_train_torch.py`, `tests/torch/test_cli_inference_torch.py`, `tests/torch/test_cli_shared.py`, `tests/torch/test_execution_config_defaults.py`
- **Dependencies:** torch import; mock `torch.cuda.is_available()` where needed
- **Execution time:** <1 second per test
- **Run frequency:** Every commit

### Integration Tests
- **Scope:** Training/inference workflows that actually run
- **Location:** `tests/torch/test_integration_workflow_torch.py` (GPU-only)
- **Dependencies:** CUDA GPU, fixture dataset
- **Execution time:** 1-10 seconds per test
- **Run frequency:** Pre-merge or GPU-enabled CI

### Smoke Tests
- **Scope:** End-to-end CLI flow (train + inference)
- **Location:** existing CLI smoke scripts/tests (if retained)
- **Dependencies:** CUDA GPU
- **Execution time:** >10 seconds
- **Run frequency:** On-demand

**Tier Strategy:**
```
Update unit tests to remove CPU/auto toggles and assert GPU-only defaults.
Gate integration tests on CUDA availability using skip markers or cuda_gpu_env fixture.
```

---

## 4. Execution Proof Requirements

### PASSED vs SKIPPED Criteria

**What constitutes PASSED:**
- Tests run successfully with assertions executed
- GPU-only enforcement verified (no CPU fallback path)

**Acceptable SKIP reasons:**
- [x] CUDA unavailable (GPU-only tests)
  - Marker: @pytest.mark.skipif(not torch.cuda.is_available(), ...)

**UNACCEPTABLE SKIP reasons:**
- Missing torch installation (POLICY-001)
- Silent CPU fallback without explicit skip or error

### Artifact Requirements
- [ ] pytest execution log at `plans/active/FIX-DEVICE-TOGGLE-001/reports/<timestamp>/pytest.log`
- [ ] Test summary in `summary.md` with counts: X passed, Y skipped, Z failed
- [ ] Explicit justification for each SKIPPED test in summary

---

## 5. Mock/Stub Strategy

### Dependencies Requiring Mocks
| Dependency | Reason | Mock Strategy |
|------------|--------|---------------|
| `torch.cuda.is_available()` | Simulate GPU presence in unit tests | `monkeypatch`/`patch` return True |

### Mock Implementation
```python
from unittest.mock import patch

@patch('torch.cuda.is_available', return_value=True)
def test_gpu_only_config_defaults(mock_available):
    ...
```

**Mocking Principles:**
- [x] Mock external environment checks only
- [x] Avoid mocking core model logic

---

## 6. Test Execution Plan

### Phase A: Red Tests (TDD)
1. Add failing tests asserting CLI rejects deprecated flags (`--device`, `--accelerator`, `--torch-accelerator`).
2. Run: `pytest tests/torch/test_cli_train_torch.py -v`
3. Expected: FAIL (before implementation)

### Phase B: Implementation
1. Implement GPU-only behavior + remove toggles.
2. Run: `pytest tests/torch/test_cli_train_torch.py -v`
3. Expected: PASS

### Phase C: Validation
1. Run targeted selectors (see plan).
2. Archive logs under `plans/active/FIX-DEVICE-TOGGLE-001/reports/<timestamp>/`.

---

## 7. Risk Mitigation

### Identified Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU-only CI cannot execute GPU-only tests | Medium | Use CUDA availability skip markers for integration tests; unit tests patch CUDA availability |
| Downstream scripts expect flags | Medium | Update docs + provide migration note in spec/workflow docs |
| Lightning accelerator string mismatch | Low | Normalize to `'gpu'` in trainer invocation and `'cuda'` in torch device mapping |

---

## 8. Success Criteria

- [ ] Unit tests updated and PASS in dev
- [ ] GPU-only integration tests PASS or SKIP with documented reason
- [ ] Logs captured and archived per plan
- [ ] Docs/test registry updated if selectors change

---

## 9. Approval

**Reviewed by:** TBD
**Date:** TBD
**Status:** Draft

---

## References

- Test harness compatibility: `docs/DEVELOPER_GUIDE.md` Section 4.3
- Torch-optional pattern: `tests/conftest.py`
- Pytest markers: `pytest.ini`
