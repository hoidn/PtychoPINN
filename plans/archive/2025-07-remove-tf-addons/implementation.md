<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Remove TensorFlow Addons Dependency
**Initiative Path:** `plans/active/remove-tf-addons-dependency/`

---
## Git Workflow Information
**Feature Branch:** feature/remove-tf-addons-dependency
**Baseline Branch:** feature/tike-comparison-integration
**Baseline Commit Hash:** 1080ab7620cd781b49564d09040464bbe5a02d32
**Last Phase Commit Hash:** 1080ab7620cd781b49564d09040464bbe5a02d32
---

**Created:** 2025-07-27
**Core Technologies:** Python, TensorFlow, NumPy

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

- **`plan.md`** - The high-level R&D Plan
  - **`implementation.md`** - This file - The Phased Implementation Plan
    - `phase_1_checklist.md` - Detailed checklist for Phase 1
    - `phase_final_checklist.md` - Checklist for the Final Phase

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** Replace tensorflow-addons dependency with native TensorFlow implementation while maintaining numerical equivalence and functionality.

**Total Estimated Duration:** 2 days

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Implementation and Validation**

**Goal:** To implement the native TensorFlow translation function and validate it produces identical results to the tensorflow-addons version.

**Deliverable:** Updated `ptycho/tf_helper.py` with new `translate_core()` function and comprehensive test file `tests/test_tf_helper.py` proving numerical equivalence.

**Estimated Duration:** 1 day

**Key Tasks:**
- Implement `translate_core()` function using `tf.raw_ops.ImageProjectiveTransformV3`
- Update the existing `translate()` function to use the new implementation
- Create `tests/test_tf_helper.py` with direct comparison tests
- Test edge cases: zero shifts, integer shifts, sub-pixel shifts
- Verify complex tensor support through `@complexify_function` decorator

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** 
- `python -m unittest tests/test_tf_helper.py` passes with all tests green
- Numerical comparison shows results match within 1e-6 tolerance

---

### **Final Phase: Integration Testing and Dependency Removal**

**Goal:** Validate the complete implementation works in the full system, remove the tensorflow-addons dependency, and ensure all documentation is updated.

**Deliverable:** A fully tested PtychoPINN system without tensorflow-addons dependency, with all tests passing and documentation updated.

**Estimated Duration:** 1 day

**Key Tasks:**
- Run the full project test suite: `python -m unittest discover -s tests`
- Execute end-to-end verification: `ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 512 --output_dir verification_run_no_addons`
- Remove 'tensorflow-addons' from `setup.py` install_requires
- Uninstall tensorflow-addons and reinstall project: `pip uninstall tensorflow-addons && pip install -e .`
- Update any documentation that references tensorflow-addons
- Run performance benchmarks to ensure no regression

**Dependencies:** Requires Phase 1 completion

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** 
- All project tests pass without tensorflow-addons installed
- Training workflow completes successfully
- Performance benchmarks show no significant regression

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [ ] **Phase 1:** Core Implementation and Validation - 0% complete
- [ ] **Final Phase:** Integration Testing and Dependency Removal - 0% complete

**Current Phase:** Phase 1: Core Implementation and Validation
**Overall Progress:** ░░░░░░░░░░░░░░░░ 0%

---

## 🚀 **GETTING STARTED**

1. **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist.
2. **Begin Implementation:** Follow the checklist tasks in order.
3. **Track Progress:** Update task states in the checklist as you work.
4. **Request Review:** Run `/complete-phase` when all Phase 1 tasks are done to generate a review request.

---

## ⚠️ **RISK MITIGATION**

**Potential Blockers:**
- **Risk:** The coordinate system differences between TensorFlow and TFA might cause subtle bugs.
  - **Mitigation:** Extensive testing with visual inspection of translated images during development.
- **Risk:** Performance regression with the native implementation.
  - **Mitigation:** Early benchmarking in Phase 1, with optimization if needed.

**Rollback Plan:**
- **Git:** Each phase will be a separate, reviewed commit on the feature branch, allowing for easy reverts.
- **Dependency:** The tensorflow-addons package can be quickly reinstalled if critical issues arise.