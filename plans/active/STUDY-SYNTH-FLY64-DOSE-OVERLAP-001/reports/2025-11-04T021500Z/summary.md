# Phase A Summary: Design Constants & TDD Harness

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Timestamp:** 2025-11-04T021500Z
**Phase:** A — Design & Constraints
**Status:** COMPLETE

---

## Deliverables

### 1. Design Module
**File:** `studies/fly64_dose_overlap/design.py`

Canonical source of truth for Phase A study parameters:

**Dose sweep:**
- [1e3, 1e4, 1e5] photons per exposure

**Grouping:**
- Gridsizes: {1, 2}
- neighbor_count=7 (K ≥ C=4 for K-choose-C oversampling)

**Inter-group overlap control:**
- Dense view: f=0.7 → S ≈ 38.4 pixels
- Sparse view: f=0.2 → S ≈ 102.4 pixels
- Heuristic: S = (1 − f_group) × N where N=128

**Train/test split:**
- Axis: 'y' (spatial separation)

**RNG seeds:**
- Simulation: 42
- Grouping: 123
- Subsampling: 456

**Metrics:**
- MS-SSIM sigma=1.0, emphasize_phase=True, report_amplitude=True

### 2. TDD Test Harness
**File:** `tests/study/test_dose_overlap_design.py`

Three test functions:
1. `test_study_design_constants` — validates all Phase A constants
2. `test_study_design_validation` — validates constraint checking (K≥C, overlap ranges, etc.)
3. `test_study_design_to_dict` — validates serialization

**All tests PASSED on first run** (implementation-first approach, validated by tests).

### 3. Test Execution Proof
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T021500Z/green/pytest_green.log`

```
tests/study/test_dose_overlap_design.py::test_study_design_constants PASSED [ 33%]
tests/study/test_dose_overlap_design.py::test_study_design_validation PASSED [ 66%]
tests/study/test_dose_overlap_design.py::test_study_design_to_dict PASSED [100%]

============================== 3 passed in 0.83s ===============================
```

No SKIPs; all assertions verified.

---

## Rationale & Decisions

### Spacing Heuristic Derivation
From docs/GRIDSIZE_N_GROUPS_GUIDE.md:142-151:
> S ≈ (1 − f_group) × N where S is min center spacing (pixels), f_group is overlap fraction, N is patch size.

Applied with N=128 (fly64 patch size):
- Dense (f=0.7): S = 0.3 × 128 = 38.4 pixels
- Sparse (f=0.2): S = 0.8 × 128 = 102.4 pixels

### K-choose-C Oversampling
Per docs/SAMPLING_USER_GUIDE.md:116-119, K ≥ C is required for oversampling.
- C = gridsize² = 4 (for gs=2)
- K = neighbor_count = 7
- 7 ≥ 4 ✓

### Fixed Seeds
All seeds documented in code and docs to ensure reproducibility across simulation, grouping, and subsampling stages.

### No CONFIG-001 Bridge Required
Design module is pure data (no torch/tensorflow imports, no params.cfg dependency). Future dataset generation scripts will handle CONFIG-001 initialization before calling simulation code.

---

## Alignment with Specifications

- **specs/data_contracts.md §2:** Design anticipates amplitude/complex dtype requirements; datasets will be validated in Phase C.
- **docs/GRIDSIZE_N_GROUPS_GUIDE.md:** Spacing heuristic implemented exactly per documented rule.
- **docs/SAMPLING_USER_GUIDE.md:** K≥C constraint validated via test.
- **docs/findings.md (OVERSAMPLING-001):** K>C requirement enforced in validation logic.

---

## Next Actions

**Immediate next phase:** Phase B — Test Infrastructure Design
- Author unit tests for dataset contract validation
- Design integration tests for end-to-end dataset creation + split + filtering (no training)
- Define smoke tests for minimal training invocation

**Blocked/Dependencies:** None

**Open Questions:** None

---

## Artifacts

```
plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/
├── implementation.md (Phase A section updated)
├── test_strategy.md (Phase A section updated)
└── reports/2025-11-04T021500Z/
    ├── summary.md (this file)
    └── green/
        └── pytest_green.log (3/3 PASSED)
```

**Code artifacts:**
- `studies/fly64_dose_overlap/design.py` (155 lines)
- `tests/study/test_dose_overlap_design.py` (143 lines)
