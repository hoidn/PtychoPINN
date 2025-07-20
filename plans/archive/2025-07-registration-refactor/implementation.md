<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Add Robust Image Registration to Evaluation Pipeline

**Core Technologies:** Python, NumPy, SciPy, scikit-image, Cross-correlation, Image Processing

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

*   **`docs/refactor/plan_registration.md`** (The high-level R&D Plan)
    *   **`docs/refactor/implementation_registration.md`** (This file - The Phased Implementation Plan)
        *   `docs/refactor/phase_1_checklist.md` (Detailed checklist for Phase 1)
        *   `docs/refactor/phase_2_checklist.md` (Detailed checklist for Phase 2)
        *   `docs/refactor/phase_3_checklist.md` (Detailed checklist for Phase 3)

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** To create a robust image registration system that automatically aligns reconstructions before evaluation, eliminating spurious metric differences caused by translational shifts.

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Registration Module Development**

**Goal:** To create a standalone, well-tested image registration module with cross-correlation-based alignment functions.

**Deliverable:** A complete `ptycho/image/registration.py` module containing `find_translation_offset()` and `apply_shift_and_crop()` functions with comprehensive unit tests.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] docs/refactor/phase_1_checklist.md`

**Key Tasks Summary:**
*   Create `ptycho/image/registration.py` with core registration functions
*   Implement `find_translation_offset()` using phase cross-correlation
*   Implement `apply_shift_and_crop()` for image shifting and cropping
*   Add comprehensive docstrings and type hints
*   Create unit tests covering known shifts, edge cases, and pre-aligned images
*   Verify functions work with complex-valued images

**Success Test:** All tasks in `phase_1_checklist.md` are marked as done. Unit tests pass, demonstrating correct offset detection and image alignment for known test cases.

**Duration:** 1-2 days

---

### **Phase 2: Integration with Evaluation Pipeline**

**Goal:** To integrate the registration module into the existing comparison workflow, ensuring all reconstructions are properly aligned before metric calculation.

**Deliverable:** A modified `scripts/compare_models.py` that automatically registers and aligns PtychoPINN and Baseline reconstructions against ground truth before evaluation.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] docs/refactor/phase_2_checklist.md`

**Key Tasks Summary:**
*   Import registration functions into `scripts/compare_models.py`
*   Add registration step after reconstruction reassembly
*   Implement workflow: register PINN → GT, register Baseline → GT
*   Apply shifts and crop all images to common overlapping region
*   Ensure aligned images are passed to `eval_reconstruction`
*   Add logging to track detected offsets and alignment results

**Success Test:** All tasks in `phase_2_checklist.md` are marked as done. Running `compare_models.py` on problematic datasets produces different FRC values for PtychoPINN vs Baseline, and visual inspection shows proper alignment.

**Duration:** 1 day

---

### **Phase 3: Validation and Documentation**

**Goal:** To thoroughly validate the registration system works correctly and document the new capability for future users.

**Deliverable:** Comprehensive validation results demonstrating the registration system resolves alignment issues, plus updated documentation explaining the registration workflow.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] docs/refactor/phase_3_checklist.md`

**Key Tasks Summary:**
*   Run integration tests on datasets with known alignment issues
*   Verify that previously identical FRC values are now differentiated
*   Create before/after comparison showing registration effectiveness
*   Update `DEVELOPER_GUIDE.md` with registration workflow documentation
*   Add examples and usage patterns for the registration module
*   Document when registration should/shouldn't be used

**Success Test:** All tasks in `phase_3_checklist.md` are marked as done. Documentation is complete, and validation results demonstrate the registration system successfully resolves the identical FRC issue.

**Duration:** 1 day

---

## 📝 **PHASE TRACKING**

- ✅ **Phase 1:** Core Registration Module Development (see `docs/refactor/phase_1_checklist.md`)
- ✅ **Phase 2:** Integration with Evaluation Pipeline (see `docs/refactor/phase_2_checklist.md`)
- ✅ **Phase 3:** Validation and Documentation (see `docs/refactor/phase_3_checklist.md`)

**Current Phase:** 🎉 **ALL PHASES COMPLETE** 🎉  
**Status:** Registration system is fully operational and validated. The core "identical FRC issue" has been resolved.