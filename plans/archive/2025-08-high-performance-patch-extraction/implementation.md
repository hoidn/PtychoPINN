<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** High-Performance Patch Extraction Refactoring
**Initiative Path:** `plans/active/high-performance-patch-extraction/`

---
## Git Workflow Information
**Feature Branch:** feature/high-performance-patch-extraction
**Baseline Branch:** feature/simulation-workflow-unification
**Baseline Commit Hash:** 9a67d07a3b0c0b14403f2d50b78e574de4f7aadc
**Last Phase Commit Hash:** 4dccc17
---

**Created:** 2025-08-03
**Core Technologies:** Python, TensorFlow, NumPy

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

- **`plan.md`** - The high-level R&D Plan
  - **`implementation.md`** - This file - The Phased Implementation Plan
    - `phase_1_checklist.md` - Detailed checklist for Phase 1
    - `phase_2_checklist.md` - Detailed checklist for Phase 2
    - `phase_3_checklist.md` - Detailed checklist for Phase 3
    - `phase_final_checklist.md` - Checklist for the Final Phase

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** Replace the slow iterative patch extraction loop with a high-performance batched implementation using the existing XLA-compatible translation engine, achieving 10-100x speedup while maintaining bit-for-bit identical output.

**Total Estimated Duration:** 3-4 days

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Batched Implementation**

**Goal:** To implement the new batched patch extraction function that replaces the for loop with a single batched translation call.

**Deliverable:** A new `_get_image_patches_batched` function in `ptycho/raw_data.py` with basic unit tests demonstrating functionality.

**Estimated Duration:** 1 day

**Key Tasks:**
- Extract the current loop-based implementation into `_get_image_patches_iterative()` function
- Implement `_get_image_patches_batched()` using `tf_helper.translate` with batched operations
- Add the configuration parameter `use_batched_patch_extraction` to `ModelConfig`
- Create basic unit tests to verify the batched implementation works

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** New unit tests in `tests/test_raw_data.py` pass, demonstrating the batched function produces valid output.

---

### **Phase 2: Feature Flag Integration & Dispatcher**

**Goal:** To integrate the feature flag system that allows safe switching between implementations.

**Deliverable:** Modified `get_image_patches` function that acts as a dispatcher between old and new implementations based on configuration.

**Estimated Duration:** 0.5 days

**Key Tasks:**
- Modify `get_image_patches` to check the `use_batched_patch_extraction` configuration parameter
- Ensure proper configuration flow from dataclass to legacy params system
- Add logging to indicate which implementation is being used
- Test the dispatcher with both flag states

**Dependencies:** Requires Phase 1 completion

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:** The `get_image_patches` function correctly routes to either implementation based on configuration setting.

---

### **Phase 3: Comprehensive Equivalence Testing**

**Goal:** To create a rigorous test suite that proves the new implementation is numerically identical to the old one across all edge cases.

**Deliverable:** A comprehensive test file `tests/test_patch_extraction_equivalence.py` with performance benchmarks.

**Estimated Duration:** 1 day

**Key Tasks:**
- Create parameterized tests for various configurations (N, gridsize, batch sizes, dtypes)
- Test edge cases (border coordinates, single patches, empty batches)
- Implement performance benchmarking with timing measurements
- Add memory usage profiling to verify efficiency claims
- Set up tolerance testing with `np.testing.assert_allclose` at 1e-6

**Dependencies:** Requires Phase 2 completion

**Implementation Checklist:** `phase_3_checklist.md`

**Success Test:** All equivalence tests pass with tolerance atol=1e-6, and performance improvement of at least 10x is demonstrated.

---

### **Final Phase: Validation & Documentation**

**Goal:** Enable the new implementation by default, update documentation, and prepare for legacy code removal.

**Deliverable:** Updated configuration defaults, comprehensive documentation, and a cleanup plan.

**Estimated Duration:** 0.5-1 day

**Key Tasks:**
- Run full test suite to ensure no regressions
- Update default value of `use_batched_patch_extraction` to `True`
- Update `DEVELOPER_GUIDE.md` with information about the new high-performance implementation
- Update docstrings in `raw_data.py` to reflect the new approach
- Document the deprecation timeline for the iterative implementation
- Create a follow-up issue for removing the legacy code after stabilization period

**Dependencies:** All previous phases complete

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** All project tests pass with the new implementation as default, and documentation accurately reflects the changes.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [x] **Phase 1:** Core Batched Implementation - 100% complete
- [x] **Phase 2:** Feature Flag Integration & Dispatcher - 100% complete
- [x] **Phase 3:** Comprehensive Equivalence Testing - 100% complete
- [ ] **Final Phase:** Validation & Documentation - 0% complete

**Current Phase:** Final Phase: Validation & Documentation
**Overall Progress:** ████████████░░░░ 75%

---

## 🚀 **GETTING STARTED**

1. **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist.
2. **Begin Implementation:** Follow the checklist tasks in order.
3. **Track Progress:** Update task states in the checklist as you work.
4. **Request Review:** Run `/complete-phase` when all Phase 1 tasks are done to generate a review request.

---

## ⚠️ **RISK MITIGATION**

**Potential Blockers:**
- **Risk:** The batched tensor operations may have different memory layout requirements than expected.
  - **Mitigation:** Start with small test cases and gradually increase batch sizes while monitoring memory usage.
- **Risk:** XLA compilation might introduce subtle numerical differences.
  - **Mitigation:** The 1e-6 tolerance in tests should catch meaningful differences; we can adjust if needed based on findings.

**Rollback Plan:**
- **Git:** Each phase will be a separate, reviewed commit on the feature branch, allowing for easy reverts.
- **Feature Flag:** The `use_batched_patch_extraction` flag allows immediate rollback to the iterative implementation if issues arise in production.