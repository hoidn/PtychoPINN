<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Simulation Workflow Unification
**Initiative Path:** `plans/examples/2025-08-simulation-workflow-unification/`

---
## Git Workflow Information
**Feature Branch:** feature/simulation-workflow-unification
**Baseline Branch:** feature/2x2study
**Baseline Commit Hash:** bd0dc5b66b4128d75284203f62e6134d74626192
**Last Phase Commit Hash:** bd0dc5b66b4128d75284203f62e6134d74626192
---

**Created:** 2025-08-02
**Core Technologies:** Python, NumPy, TensorFlow, ptychography simulation

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

- **`plan.md`** - The high-level R&D Plan
  - **`implementation.md`** - This file - The Phased Implementation Plan
    - `phase_1_checklist.md` - Detailed checklist for Phase 1
    - `phase_2_checklist.md` - Detailed checklist for Phase 2
    - `phase_final_checklist.md` - Checklist for the Final Phase

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** Fix the gridsize > 1 crash in simulate_and_save.py by refactoring it to use explicit, modular orchestration instead of the monolithic RawData.from_simulation method.

**Total Estimated Duration:** 3 days

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Refactoring - Replace Monolithic Function**

**Goal:** To refactor `scripts/simulation/simulate_and_save.py` to use explicit orchestration of modular functions instead of the monolithic `RawData.from_simulation` method.

**Deliverable:** A refactored `simulate_and_save.py` script that explicitly orchestrates coordinate grouping, patch extraction, and diffraction simulation, fixing the gridsize > 1 crash.

**Estimated Duration:** 1 day

**Key Tasks:**
- Analyze the current `simulate_and_save.py` implementation and identify all usages of `RawData.from_simulation`.
- Implement the new orchestration workflow:
  - Load inputs (`objectGuess`, `probeGuess`) from NPZ files
  - Use `ptycho.raw_data.group_coords()` for coordinate generation and grouping
  - Use `ptycho.raw_data.get_image_patches()` for patch extraction
  - Handle format conversions between Channel and Flat formats using `tf_helper` functions
  - Use `ptycho.diffsim.illuminate_and_diffract()` for simulation
  - Assemble and save results according to data contracts
- Ensure the refactored script maintains all existing command-line arguments and functionality.
- Add debug logging to trace the data flow and tensor shapes throughout the pipeline.

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** Running `python scripts/simulation/simulate_and_save.py --input-file datasets/fly/fly001_transposed.npz --output-file test_sim.npz --gridsize 2` completes without errors and produces a valid NPZ file.

---

### **Phase 2: Integration Testing & Validation**

**Goal:** To create a comprehensive test suite that validates the refactored simulation pipeline for both gridsize=1 and gridsize > 1 cases.

**Deliverable:** A new test file `tests/simulation/test_simulate_and_save.py` with comprehensive integration tests covering all validation scenarios from the R&D plan.

**Estimated Duration:** 1 day

**Key Tasks:**
- Create the test directory structure `tests/simulation/` if it doesn't exist.
- Implement integration tests for:
  - Gridsize=1 regression test (verify single-channel output shapes)
  - Gridsize=2 correctness test (verify multi-channel output shapes)
  - Probe override functionality test
  - Data contract compliance verification
  - Content sanity checks (non-zero values, correct data types)
- Add performance benchmarks to ensure no significant regression.
- Create visual validation scripts using `visualize_dataset.py` for manual inspection.
- Run the full test suite and ensure all tests pass.

**Dependencies:** Requires Phase 1 completion.

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:** Running `pytest tests/simulation/test_simulate_and_save.py -v` shows all tests passing with 100% success rate.

---

### **Final Phase: Deprecation, Documentation & Cleanup**

**Goal:** To add deprecation warnings to the legacy method, update all documentation, and ensure the solution is production-ready.

**Deliverable:** Complete documentation updates, deprecation warnings in place, and all success criteria from the R&D plan verified.

**Estimated Duration:** 1 day

**Key Tasks:**
- Add `DeprecationWarning` to `RawData.from_simulation` method with guidance to use the new approach.
- Search codebase for any other usages of `RawData.from_simulation` and document findings.
- Update `scripts/simulation/CLAUDE.md` with the new architecture and usage examples.
- Update `scripts/simulation/README.md` with clear documentation of the changes.
- Update the main `CLAUDE.md` if necessary to reflect the unified simulation workflow.
- Verify all success criteria from the R&D plan are met:
  - No crashes with gridsize > 1
  - Data contract compliance
  - Performance benchmarks acceptable
  - All tests passing
- Create a migration guide for any downstream users of the deprecated method.

**Dependencies:** All previous phases complete.

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** All R&D plan success criteria are verified as complete, documentation is updated, and deprecation warnings are properly displayed when using the legacy method.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [ ] **Phase 1:** Core Refactoring - Replace Monolithic Function - 0% complete
- [ ] **Phase 2:** Integration Testing & Validation - 0% complete
- [ ] **Final Phase:** Deprecation, Documentation & Cleanup - 0% complete

**Current Phase:** Phase 1: Core Refactoring - Replace Monolithic Function
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
- **Risk:** The modular functions (`group_coords`, `get_image_patches`) may have undocumented assumptions or bugs.
  - **Mitigation:** Add comprehensive debug logging and validate intermediate results against known-good data.
- **Risk:** Format conversions between Channel and Flat formats may introduce subtle bugs.
  - **Mitigation:** Create unit tests specifically for the format conversion functions and validate with known test cases.
- **Risk:** Performance regression due to explicit orchestration overhead.
  - **Mitigation:** Profile both old and new implementations early in Phase 1, optimize if necessary.

**Rollback Plan:**
- **Git:** Each phase will be a separate, reviewed commit on the feature branch, allowing for easy reverts.
- **Legacy Path:** The original `RawData.from_simulation` remains available (with deprecation warning) as a fallback.