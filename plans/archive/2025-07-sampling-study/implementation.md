<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Spatially-Biased Randomized Sampling Study

**Core Technologies:** Python, NumPy, Bash scripting, NPZ file format

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

*   **`docs/sampling/plan_sampling_study.md`** (The high-level R&D Plan)
    *   **`implementation_sampling_study.md`** (This file - The Phased Implementation Plan)
        *   `phase_1_checklist_sampling_study.md` (Detailed checklist for Phase 1)
        *   `phase_final_checklist_sampling_study.md` (Checklist for the Final Phase)

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** To create a reusable shuffling tool that enables statistically valid generalization studies on random samples from specific spatial regions of datasets.

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Shuffle Dataset Tool Creation**

**Goal:** To create and validate the core `shuffle_dataset_tool.py` script that can randomize NPZ datasets while preserving data relationships.

**Deliverable:** A working `scripts/tools/shuffle_dataset_tool.py` that correctly shuffles per-scan arrays in unison while preserving global arrays.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_1_checklist_sampling_study.md`

**Success Test:** All tasks in the Phase 1 checklist are marked as done. The shuffling tool passes unit tests with synthetic data, correctly reordering per-scan arrays while preserving global arrays and maintaining data relationships.

---

### **Final Phase: Validation & Documentation**

**Goal:** To validate the complete workflow with a real dataset and update all relevant documentation.

**Deliverable:** A fully tested and documented shuffling workflow, with a completed fly64 generalization study demonstrating the new capability.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_final_checklist_sampling_study.md`

**Key Tasks Summary:**
*   Execute the complete `convert -> split -> shuffle -> run_study` workflow on the fly64 dataset
*   Verify the generalization study completes successfully with randomized top-half data
*   Update `scripts/tools/README.md` with documentation for the new shuffling tool
*   Update `scripts/studies/QUICK_REFERENCE.md` with the new workflow example
*   Update `PROJECT_STATUS.md` to mark the initiative as complete

**Success Test:** All tasks in the final phase checklist are marked as done. The fly64 generalization study produces valid results with properly shuffled data, and all documentation is updated.

---

## 📝 **PHASE TRACKING**

- [✅] **Phase 1:** Shuffle Dataset Tool Creation (see `phase_1_checklist_sampling_study.md`)
- [ ] **Final Phase:** Validation & Documentation (see `phase_final_checklist_sampling_study.md`)

**Current Phase:** Final Phase: Validation & Documentation
**Next Milestone:** A fully tested and documented shuffling workflow, with a completed fly64 generalization study demonstrating the new capability.