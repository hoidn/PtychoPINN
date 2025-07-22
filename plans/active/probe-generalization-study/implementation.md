<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Probe Generalization Study
**Initiative Path:** `plans/active/probe-generalization-study/`

---
## Git Workflow Information
**Feature Branch:** feature/probe-generalization-study
**Baseline Branch:** devel
**Baseline Commit Hash:** 973d684ad172fe987067b48693f6804ad74facfa
**Last Phase Commit Hash:** 973d684ad172fe987067b48693f6804ad74facfa
---

**Created:** 2025-07-22
**Core Technologies:** Python, NumPy, TensorFlow, scikit-image, ptychographic simulation

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

**Overall Goal:** To systematically evaluate the impact of probe function variations (idealized vs. experimental) on PtychoPINN model performance across different overlap constraints through a controlled 2x2 experimental study.

**Total Estimated Duration:** 3 days + compute time

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Housekeeping & Workflow Verification**

**Goal:** To perform targeted code cleanup and verify that the existing synthetic 'lines' workflow functions correctly for both gridsize=1 and gridsize=2, adding a new unit test for this capability.

**Deliverable:** A cleaner codebase with a new unit test in `tests/test_simulation.py` that confirms the successful generation of 'lines' datasets for both grid sizes.

**Estimated Duration:** 1 day

**Key Tasks:**
- **Housekeeping:** Execute Phase 1 of the previously defined "Codebase Housekeeping" plan (centralize tests, archive example plans, remove legacy scripts).
- **Verification:** Run the `scripts/simulation/run_with_synthetic_lines.py` script for gridsize=1 and gridsize=2 to confirm it generates valid, trainable datasets.
- **Testing:** Add a new unit test to `tests/test_simulation.py` that automates this verification.

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** The new unit test passes, and `python -m unittest discover -s tests` runs successfully.

---

### **Phase 2: Experimental Probe Integration**

**Goal:** To create a reusable workflow for simulating data using the experimental probe from the fly64 dataset.

**Deliverable:** A new `.npz` file, `simulation_input_experimental_probe.npz`, containing a synthetic 'lines' object and the experimental probeGuess from `datasets/fly64/fly001_64_train_converted.npz`.

**Estimated Duration:** 0.5 days

**Key Tasks:**
- Write a small helper script to load the probeGuess from the fly64 dataset.
- Generate a standard synthetic 'lines' objectGuess.
- Save both arrays into a new .npz file that can be fed into `simulate_and_save.py`.
- Run a small test simulation using this new input file to verify it works.

**Dependencies:** Requires Phase 1 completion.

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:** The command `python scripts/simulation/simulate_and_save.py --input-file simulation_input_experimental_probe.npz` runs successfully.

---

### **Phase 3: Automated 2x2 Study Execution**

**Goal:** To automate and execute the full 2x2 probe generalization study, training all four model configurations.

**Deliverable:** A new orchestration script, `scripts/studies/run_probe_generalization_study.sh`, and the completed training outputs for all four experimental arms.

**Estimated Duration:** 1 day (plus compute time)

**Key Tasks:**
- Create the `run_probe_generalization_study.sh` script.
- The script will execute four separate runs of `run_comparison.sh`, correctly configuring the gridsize and the input data (simulated with either the default or experimental probe).
- Each run will have a distinct output directory (e.g., `probe_study/ideal_gs1`, `probe_study/exp_gs2`).

**Dependencies:** Requires Phase 2 completion.

**Implementation Checklist:** `phase_3_checklist.md`

**Success Test:** The script completes all four training and comparison runs without error, and each output directory contains a valid `comparison_metrics.csv` file.

---

### **Final Phase: Results Aggregation & Documentation**

**Goal:** To analyze the results from the four experiments, generate the final comparison report and plot, and document the findings.

**Deliverable:** The final `2x2_comparison_report.md` and `2x2_comparison_plot.png` artifacts, with the initiative archived.

**Estimated Duration:** 0.5 days

**Key Tasks:**
- Write a Python script to parse the four `comparison_metrics.csv` files.
- Generate the summary table and the 2x2 visualization plot.
- Write a brief analysis of the results in the markdown report.
- Update `docs/PROJECT_STATUS.md` to move this initiative to the "Completed" section.

**Dependencies:** All previous phases complete.

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** All R&D plan success criteria are met, and the final artifacts are generated correctly.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [ ] **Phase 1:** Housekeeping & Workflow Verification - 0% complete
- [ ] **Phase 2:** Experimental Probe Integration - 0% complete
- [ ] **Phase 3:** Automated 2x2 Study Execution - 0% complete
- [ ] **Final Phase:** Results Aggregation & Documentation - 0% complete

**Current Phase:** Phase 1: Housekeeping & Workflow Verification
**Overall Progress:** ░░░░░░░░░░░░░░░░ 0%

---

## 🚀 **GETTING STARTED**

1.  **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist.
2.  **Begin Implementation:** Follow the checklist tasks in order.
3.  **Track Progress:** Update task states in the checklist as you work.
4.  **Request Review:** Run `/complete-phase` when all Phase 1 tasks are done to generate a review request.

---

## ⚠️ **RISK MITIGATION**

**Potential Blockers:**
- **Risk:** The synthetic 'lines' workflow may fail on one of the grid sizes due to post-refactor changes.
  - **Mitigation:** Phase 1 verification will catch this early, allowing fixes before the main study.
- **Risk:** The experimental probe from fly64 dataset may be incompatible with the simulation pipeline.
  - **Mitigation:** Phase 2 includes validation step with test simulation before proceeding to full study.
- **Risk:** Training runs may fail due to memory constraints or other system issues.
  - **Mitigation:** The orchestration script will include error handling and resume capability.

**Rollback Plan:**
- **Git:** Each phase will be a separate, reviewed commit on the feature branch, allowing for easy reverts.
- **Incremental Validation:** Each phase produces a testable deliverable, preventing error propagation.
- **Existing Infrastructure:** Study leverages proven `run_comparison.sh` framework, minimizing risk of fundamental failures.