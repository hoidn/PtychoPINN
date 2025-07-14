<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Evaluation Pipeline Enhancements

**Core Technologies:** Python, NumPy, scikit-image, Pandas

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

*   **`docs/refactor/eval_enhancements/plan_eval_enhancements.md`** (The high-level R&D Plan)
    *   **`implementation_eval_enhancements.md`** (This file - The Phased Implementation Plan)
        *   `phase_1_checklist.md` (Detailed checklist for Phase 1)
        *   `phase_2_checklist.md` (Detailed checklist for Phase 2)
        *   `phase_3_checklist.md` (Detailed checklist for Phase 3 - NEW)
        *   `phase_4_checklist.md` (Detailed checklist for Phase 4)

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** To create an enhanced evaluation pipeline with SSIM and MS-SSIM metrics, configurable phase alignment methods, and transparent FRC calculations while maintaining backwards compatibility.

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Evaluation Function Enhancement**

**Goal:** To update the `ptycho/evaluation.py` module with new SSIM metric integration, configurable phase alignment, and enhanced FRC functionality.

**Deliverable:** A modified `eval_reconstruction` function that includes SSIM calculation, switchable phase alignment methods (`'plane'` and `'mean'`), and enhanced FRC with configurable smoothing.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_1_checklist.md`

**Key Tasks Summary:**
*   Add SSIM metric calculation using `skimage.metrics.structural_similarity`
*   Implement plane-fitting phase alignment method using `numpy.linalg.lstsq`
*   Add `phase_align_method` parameter with `'plane'` (default) and `'mean'` options
*   Enhance FRC function with `frc_sigma` parameter and raw curve return
*   Ensure backwards compatibility by preserving all existing metric keys

**Success Test:** All tasks in `phase_1_checklist.md` are marked as done. Unit tests pass: SSIM self-comparison returns 1.0, plane-fitting correctly removes known planes, and FRC self-comparison returns all 1s.

**Duration:** 2 days

---

### **Phase 2: Integration and Script Updates**

**Goal:** To update the `scripts/compare_models.py` script to handle and report all new metrics while maintaining existing functionality.

**Deliverable:** A modified `compare_models.py` script that correctly processes and saves SSIM, enhanced FRC data, and phase alignment metrics to the CSV output.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_2_checklist.md`

**Key Tasks Summary:**
*   Update CSV writing logic to handle new metric keys
*   Ensure proper handling of raw FRC curve data
*   Add logging for new metrics during comparison runs
*   Verify backwards compatibility with existing CSV parsing scripts

**Success Test:** All tasks in `phase_2_checklist.md` are marked as done. Running `compare_models.py` completes without errors and produces a CSV file containing all expected metrics including `ssim`, `frc`, and legacy metrics.

**Duration:** 1 day

---

### **Phase 3: MS-SSIM Metric Integration**

**Goal:** To implement the Multi-Scale Structural Similarity (MS-SSIM) metric, using the correct pre-processing pipeline established in the previous phases.

**Deliverable:** An updated `eval_reconstruction` function that also calculates and returns the MS-SSIM for both amplitude and phase.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_3_checklist.md`

**Key Tasks Summary:**
*   Implement a helper function for MS-SSIM calculation in `ptycho/evaluation.py`.
*   Ensure the function uses the same pre-processed (unwrapped, plane-corrected) phase data as the other metrics.
*   Add a shared debugging utility function to save PNG visualizations of pre-processed images (following DRY principles).
*   Add debugging capability to save PNG visualizations of pre-processed phase images just before MS-SSIM calculation.
*   Add debugging capability to save PNG visualizations of pre-processed phase images just before FRC calculation (if FRC preprocessing differs from other metrics).
*   Update `eval_reconstruction` to call this new function and include `ms_ssim` in its return dictionary.
*   Update `scripts/compare_models.py` to handle and save the new `ms_ssim` metric.

**Success Test:** All tasks in `phase_3_checklist.md` are marked as done. The output `comparison_metrics.csv` now includes a valid `ms_ssim` row.

**Duration:** 1 day

---

### **Phase 4: Final Validation and Performance Testing**

**Goal:** To thoroughly validate the fully-enhanced evaluation pipeline through comprehensive testing and performance verification.

**Deliverable:** A fully validated evaluation system with documented performance characteristics and comprehensive test coverage.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_4_checklist.md`

**Key Tasks Summary:**
*   Run comprehensive integration tests with real model comparison data, verifying all new metrics (SSIM, MS-SSIM).
*   Validate performance requirements (≤10% runtime increase).
*   Verify all success criteria from the R&D plan are met.
*   Create documentation for new parameters and usage patterns in `DEVELOPER_GUIDE.md`.

**Success Test:** All tasks in `phase_4_checklist.md` are marked as done. Full end-to-end comparison runs complete successfully, performance benchmarks pass, and all R&D plan success criteria are verified.

**Duration:** 1 day

---

## 📝 **PHASE TRACKING**

- ✅ **Phase 1:** Core Evaluation Function Enhancement (see `phase_1_checklist.md`)
- ✅ **Phase 2:** Integration and Script Updates (see `phase_2_checklist.md`)
- ✅ **Phase 3:** MS-SSIM Metric Integration (see `phase_3_checklist.md`)
- ✅ **Phase 4:** Final Validation and Performance Testing (see `phase_4_checklist.md`)

**Current Phase:** All phases complete ✅
**Final Deliverable:** A fully validated evaluation system with SSIM/MS-SSIM metrics, configurable phase alignment, and debug visualization capabilities.