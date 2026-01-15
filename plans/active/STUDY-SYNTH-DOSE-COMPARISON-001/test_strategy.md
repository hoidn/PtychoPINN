# Test Strategy: Dose Response Study Inference Fix

**Initiative:** STUDY-SYNTH-DOSE-COMPARISON-001
**Phase:** Phase 0 / Pre-Implementation
**Date:** 2026-01-13
**Status:** Draft

---

## 1. Framework Selection & Compatibility

### Test Framework
- **Primary Framework:** pytest
- **Rationale:** Existing script-level tests use pytest with fixtures and monkeypatch.

### Compatibility Constraints
- [x] Framework supports parametrization (if needed)
- [x] Compatible with existing test suite patterns
- [x] No mixing of unittest.TestCase + pytest.parametrize
- [x] Fixture strategy defined (pytest fixtures)

**Decision:**
```
Use pytest for a lightweight script-level regression test that relies on monkeypatch
to avoid heavy model training/inference.
```

---

## 2. CI/CD Constraint Analysis

### Environment Availability
- **Development Environment:**
  - Python version: 3.10+
  - Available frameworks: pytest, tensorflow
  - Hardware: CPU

- **CI Environment:**
  - Python version: 3.10+
  - Available frameworks: pytest, tensorflow
  - Hardware: CPU
  - Gaps from dev: None expected for this test

### Optional Dependency Handling
- **Pattern to use:** Mock/stub heavy inference helpers
- **Implementation:**
  ```python
  monkeypatch.setattr(nbutils, "reconstruct_image", fake_reconstruct_image)
  monkeypatch.setattr(tf_helper, "reassemble_position", fake_reassemble_position)
  ```

**Constraints:**
- [x] CI limitations documented
- [x] Skip markers defined for unavailable dependencies
- [x] Tests run in both dev and CI (with appropriate skips)

---

## 3. Test Tier Definitions

### Unit Tests
- **Scope:** Script-level inference workflow logic in isolation
- **Location:** `tests/scripts/test_dose_response_study.py`
- **Dependencies:** Mocked model loader, reconstruction, stitching
- **Execution time:** <1 second
- **Run frequency:** Every commit touching `dose_response_study.py`

**Tier Strategy:**
```
Only unit-level coverage is required because the goal is to validate group-count
selection and inference plumbing without running the full training workflow.
```

---

## 4. Execution Proof Requirements

### PASSED vs SKIPPED Criteria

**What constitutes PASSED:**
- Test executed successfully
- Assertions on group-count capping and reconstruction output shape

**Acceptable SKIP reasons:**
- None expected for this unit test

### Artifact Requirements
- [x] pytest execution log stored under `.artifacts/`
- [x] Test summary referenced in `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/summary.md`
- [x] Explicit justification for each SKIPPED test in summary (if any)

---

## 5. Mock/Stub Strategy

### Dependencies Requiring Mocks
| Dependency | Reason | Mock Strategy |
| --- | --- | --- |
| `load_inference_bundle_with_backend` | Avoid loading real model bundles | Return dummy model + params dict |
| `loader.load` | Avoid heavy container creation | Return dummy container |
| `nbutils.reconstruct_image` | Avoid TensorFlow inference | Return dummy patches + offsets |
| `tf_helper.reassemble_position` | Avoid stitching | Return dummy array |

### Mock Implementation
```python
def fake_reconstruct_image(container, diffraction_to_obj=None):
    return np.zeros((1, N, N, 1)), container.global_offsets
```

**Mocking Principles:**
- [x] Mock external dependencies only (not internal logic)
- [x] Test real code paths where possible
- [x] Document what is mocked and why
- [x] Ensure mocks match real API contracts

---

## 6. Test Execution Plan

### Phase A: Red Tests (TDD)
1. Add a failing test verifying group-count capping for inference
2. Run: `pytest tests/scripts/test_dose_response_study.py -v`
3. Expected: FAILED (before implementation)

### Phase B: Implementation
1. Update inference workflow to cap groups and use bundle loader
2. Run: `pytest tests/scripts/test_dose_response_study.py -v`
3. Expected: PASSED

### Phase C: Validation
1. Run test collection: `pytest tests/scripts/ --collect-only`
2. Archive logs to `.artifacts/`

---

## 7. Risk Mitigation

### Identified Risks
| Risk | Impact | Mitigation |
| --- | --- | --- |
| TensorFlow import overhead | Low | Use mocks and avoid real inference |
| API drift in inference helpers | Medium | Patch by module path in tests |

---

## 8. Success Criteria

- [ ] New test passes in development environment
- [ ] Test collection log proves discovery
- [ ] Logs stored under `.artifacts/` and referenced in ledger
