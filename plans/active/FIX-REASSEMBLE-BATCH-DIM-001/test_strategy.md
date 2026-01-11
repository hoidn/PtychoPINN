# Test Strategy: Preserve Batch Dimension in Batched Reassembly

**Initiative:** FIX-REASSEMBLE-BATCH-DIM-001
**Phase:** Phase 0 / Pre-Implementation
**Date:** 2026-01-11
**Status:** Approved

---

## 1. Framework Selection & Compatibility

### Test Framework
- **Primary Framework:** pytest
- **Rationale:** Existing suite uses pytest; avoid unittest mixing.

### Compatibility Constraints
- [x] Framework supports parametrization if needed
- [x] Compatible with existing test suite patterns
- [x] No mixing of unittest.TestCase + pytest.parametrize
- [x] Fixture strategy defined (pytest fixtures)

**Decision:**
```
Use pytest with existing TF-based tests. Keep the regression test co-located in tests/study.
```

---

## 2. CI/CD Constraint Analysis

### Environment Availability
- **Development Environment:**
  - Python version: 3.11
  - Available frameworks: pytest, tensorflow
  - Hardware: CPU/GPU (GPU optional)

- **CI Environment:**
  - Python version: 3.11
  - Available frameworks: pytest, tensorflow
  - Hardware: CPU only
  - Gaps from dev: GPU unavailable

### Optional Dependency Handling
- **Pattern to use:** none (TensorFlow required by tests)

**Constraints:**
- [x] CI limitations documented
- [x] Skip markers defined for unavailable dependencies (not needed)

---

## 3. Test Tier Definitions

### Unit/Regression Tests
- **Scope:** Batch-preserving reassembly behavior
- **Location:** `tests/study/test_dose_overlap_comparison.py`
- **Dependencies:** TensorFlow
- **Execution time:** Seconds to a minute
- **Run frequency:** This change only

**Tier Strategy:**
```
Use the existing regression test in tests/study to validate batch preservation under batched reassembly.
```

---

## 4. Execution Proof Requirements

### PASSED vs SKIPPED Criteria

**What constitutes PASSED:**
- Test executed successfully
- Assertions for batch dimension evaluated

**Acceptable SKIP reasons:**
- None expected

### Artifact Requirements
- [x] pytest execution log at `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/pytest_reassemble_batch.log`
- [x] Summary in `plans/active/FIX-REASSEMBLE-BATCH-DIM-001/summary.md`

---

## 5. Mock/Stub Strategy

### Dependencies Requiring Mocks
| Dependency | Reason | Mock Strategy |
|------------|--------|---------------|
| None | Test uses synthetic tensors | Not needed |

---

## 6. Test Execution Plan

### Phase B: Implementation
1. Run: `pytest tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split -v`
2. Expected: PASSED

---

## 7. Risk Mitigation

### Identified Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Large tensor allocation | Medium | Keep input sizes as defined; rely on batched path |

---

## 8. Success Criteria

- [ ] Test passes in dev environment
- [ ] pytest log archived
- [ ] Summary updated with results

---

## 9. Approval

**Reviewed by:** Codex
**Date:** 2026-01-11
**Status:** Approved
