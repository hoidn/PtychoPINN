<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** MS-SSIM Implementation Correction

**Core Technologies:** Python, NumPy, scikit-image, SciPy

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

*   **`docs/initiatives/plan_ms_ssim_correction.md`** (The high-level R&D Plan)
    *   **`implementation_ms_ssim_correction.md`** (This file - The Phased Implementation Plan)
        *   `phase_1_checklist_ms_ssim_correction.md` (Detailed checklist for Phase 1)
        *   `phase_2_checklist_ms_ssim_correction.md` (Detailed checklist for Phase 2)

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** To replace the flawed MS-SSIM function with a mathematically correct and validated implementation, ensuring all perceptual metrics are reliable and MS-SSIM values are always ≤ SSIM values.

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Function Replacement and Unit Testing**

**Goal:** To replace the incorrect ms_ssim function in ptycho/evaluation.py with a correct version and to create a new unit test suite to validate it.

**Deliverable:** A modified ptycho/evaluation.py with the corrected ms_ssim function and a new tests/test_evaluation.py file with passing unit tests for MS-SSIM.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_1_checklist_ms_ssim_correction.md`

**Key Tasks Summary:**
*   Replace the flawed ms_ssim logic with the standard algorithm using luminance, contrast, and structure components
*   Implement proper downsampling with Gaussian filtering between scales
*   Create a new test file tests/test_evaluation.py
*   Add unit tests to verify ms_ssim(image, image) ≈ 1.0 and ms_ssim ≤ ssim for typical cases
*   Test with known reference values against the literature

**Success Test:** All tasks in the Phase 1 checklist are marked as done. The new unit tests in tests/test_evaluation.py pass successfully when run from the project root.

---

### **Phase 2: Integration, Validation, and Documentation**

**Goal:** To validate the corrected MS-SSIM metric within the full comparison workflow and update project documentation.

**Deliverable:** A new comparison_metrics.csv file with corrected MS-SSIM values showing MS-SSIM ≤ SSIM, and updated documentation reflecting the change.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_2_checklist_ms_ssim_correction.md`

**Key Tasks Summary:**
*   Re-run the scripts/compare_models.py workflow with the exact same inputs that produced the original faulty metrics
*   Inspect the new comparison_metrics.csv to confirm MS-SSIM values are now correct and reasonable (MS-SSIM ≤ SSIM)
*   Update the ptycho/evaluation.py docstrings for the ms_ssim and eval_reconstruction functions to reflect the corrected implementation
*   Update docs/PROJECT_STATUS.md to mark the initiative as complete
*   Verify no regressions in other evaluation metrics

**Success Test:** All tasks in the Phase 2 checklist are marked as done. The generated comparison_metrics.csv shows that for both PtychoPINN and Baseline models, the ms_ssim value is less than or equal to the ssim value.

---

## 📝 **PHASE TRACKING**

- [ ] **Phase 1:** Core Function Replacement and Unit Testing (see `phase_1_checklist_ms_ssim_correction.md`)
- [ ] **Phase 2:** Integration, Validation, and Documentation (see `phase_2_checklist_ms_ssim_correction.md`)

**Current Phase:** Phase 1: Core Function Replacement and Unit Testing
**Next Milestone:** A corrected ms_ssim function in ptycho/evaluation.py and a new, passing unit test suite in tests/test_evaluation.py.

---

## 🔬 **TECHNICAL SPECIFICATIONS**

**MS-SSIM Formula to Implement:**
```
MS-SSIM = (l_M)^α_M * ∏(c_j * s_j)^β_j
```

Where:
- `l_M` is the luminance at the highest scale
- `c_j, s_j` are contrast and structure at scale j
- `α_M, β_j` are the standard weights [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

**Key Requirements:**
- Must use proper component separation via `structural_similarity(..., full=True)`
- Must implement 2x downsampling with Gaussian filtering between scales
- Must ensure MS-SSIM ≤ SSIM for all valid inputs
- Must handle edge cases (small images, identical images, etc.)

---

## 🎯 **VALIDATION CRITERIA**

**Unit Test Requirements:**
1. Self-similarity: `ms_ssim(img, img) ≈ 1.0`
2. Bounded property: `ms_ssim(img1, img2) ≤ ssim(img1, img2)`
3. Symmetry: `ms_ssim(img1, img2) = ms_ssim(img2, img1)`
4. Range: `0 ≤ ms_ssim(img1, img2) ≤ 1`

**Integration Test Requirements:**
1. All existing model comparison workflows continue to work
2. MS-SSIM values in comparison_metrics.csv are ≤ corresponding SSIM values
3. No regressions in other evaluation metrics (MAE, PSNR, etc.)