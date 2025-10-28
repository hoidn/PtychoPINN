# Test Strategy: EXPORT-PTYCHODUS-PRODUCT-001

**Initiative:** EXPORT-PTYCHODUS-PRODUCT-001
**Phase:** Phase 0 / Pre-Implementation
**Date:** 2025-10-28
**Status:** Draft

---

## 1. Framework Selection & Compatibility

### Test Framework
- Primary Framework: pytest
- Rationale: Matches repository conventions; supports parametrization and concise fixtures; no unittest mixing.

### Compatibility Constraints
- [x] Parametrization supported
- [x] Compatible with existing patterns
- [x] No mixing of unittest.TestCase + pytest.parametrize
- [x] Pytest fixtures will be used for small synthetic data

**Decision:**
```
Use pytest for unit and smoke tests. Keep tests CPU-only and independent of GPU/torch. Avoid
framework mixing to comply with repo guidance.
```

---

## 2. CI/CD Constraint Analysis

### Environment Availability
- Development Environment:
  - Python: 3.10+
  - Frameworks: pytest, numpy, h5py, tensorflow available (as in repo)
  - Hardware: CPU

- CI Environment:
  - Python: 3.10+
  - Frameworks: pytest, tensorflow (torch not required for this initiative)
  - Hardware: CPU only
  - Gaps: No GPU; torch may be unavailable (not needed)

### Optional Dependency Handling
- Pattern: Not required (no torch dependency). Keep tests pure numpy/h5py/tensorflow.

**Constraints:**
- [x] CI limitations documented (CPU only)
- [x] No torch dependency introduced
- [x] Tests run in CI and dev without skips

---

## 3. Test Tier Definitions

### Unit Tests
- Scope: Export/import functions; dtype/shape/unit conversions; attribute presence
- Location: `tests/io/test_ptychodus_product_io.py`
- Dependencies: numpy, h5py; tensorflow for RawData construction if needed
- Execution time: <1s per test
- Run frequency: Every commit

### Integration/Smoke Tests
- Scope: Export a small synthetic RawData to HDF5, read with ptychodus H5 reader, validate fields
- Location: `tests/io/test_ptychodus_product_io.py::test_smoke_roundtrip`
- Dependencies: h5py, ptychodus reader module import
- Execution time: ~1–3s
- Run frequency: CI and pre-merge

**Tier Strategy:**
```
Focus on unit tests for mapping correctness and a single smoke test for interop with ptychodus
HDF5 reader. No large datasets or GPU required.
```

---

## 4. Execution Proof Requirements

### PASSED vs SKIPPED Criteria

**PASSED:** Tests execute assertions validating conversions, shapes, and key presence. No import-only tests.

**Acceptable SKIP reasons:** None expected.

**UNACCEPTABLE:** Skipping due to optional dependency misdesign; framework incompatibility.

### Artifact Requirements
- [x] pytest execution log at `plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/<timestamp>/pytest.log`
- [x] Test summary `summary.md` with pass/skip counts
- [x] Proof that assertions executed (log capture)

---

## 5. Mock/Stub Strategy

### Dependencies Requiring Mocks
| Dependency | Reason | Mock Strategy |
|------------|--------|---------------|
| Large datasets | Keep tests fast | Use tiny synthetic arrays |

### Mock Implementation
```
Use small synthetic arrays; no external API calls; no heavy mocking needed.
```

**Mocking Principles:**
- [x] Mock only where necessary; prefer real code paths
- [x] Keep fixtures minimal and deterministic

---

## 6. Test Execution Plan

### Phase A: Red Tests (TDD)
1. Write failing tests for exporter/importer mapping and interop
2. Verify failures are mapping/interop-related

### Phase B: Implementation
1. Implement minimal code to satisfy RED tests
2. Re-run targeted tests until GREEN

### Phase C: Validation
1. Run full test subset: `pytest tests/io/test_ptychodus_product_io.py -vv`
2. Save logs to `plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/<timestamp>/pytest.log`
3. Record summary.md

---

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pixel size defaults inaccurate | Low | Annotate defaults in comments; allow overrides later |
| Coord unit mismatch | Medium | Enforce pixels→meters per plan; document behavior |
| Reader interop mismatch | Medium | Validate via ptychodus H5 reader in smoke test |

---

## 8. Success Criteria

- [ ] All tests PASS (not SKIP) locally
- [ ] Interop smoke test passes with ptychodus H5 reader
- [ ] Logs and summary stored under initiative reports

---

## 9. Approval

**Reviewed by:** Supervisor/Engineer
**Date:** 2025-10-28
**Status:** Needs Revision → Approved (upon review)

**Notes:**
```
OK to proceed with dummy pixel sizes when metadata absent; coords treated as pixels; HDF5 only; no losses.
```

---

## References

- Data contracts: `specs/data_contracts.md`
- Ptychodus reader: `ptychodus/src/ptychodus/plugins/h5_product_file.py`
- TF data pipeline: `ptycho/raw_data.py`, `ptycho/loader.py`

