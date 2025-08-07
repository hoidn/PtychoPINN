<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Generalization Test with Decoupled Probe Simulation
**Initiative Path:** `plans/active/probe-parameterization-study/`

---
## Git Workflow Information
**Feature Branch:** feature/probe-parameterization-study
**Baseline Branch:** feature/remove-tf-addons-dependency
**Baseline Commit Hash:** c58e466b83b208e7427a639bdb9ea6e862a861bc
**Last Phase Commit Hash:** 1287b38495b65bec3fc560524806441ae197f131
---

**Created:** 2025-08-01
**Core Technologies:** Python, NumPy, TensorFlow, scikit-image

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

**Overall Goal:** To create a robust, decoupled simulation workflow that enables advanced studies of probe generalization, culminating in a comprehensive 2x2 study that validates the new tools and provides scientific insights.

**Total Estimated Duration:** 3 days

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Utilities and Hybrid Probe Generation**

**Goal:** To build the foundational, standalone tools required for the study: the hybrid probe generator and the core helper functions for probe loading and validation. This phase is self-contained and fully testable.

**Deliverable:** A new script `scripts/tools/create_hybrid_probe.py`, a new module `ptycho/workflows/simulation_utils.py` with helper functions, a complete set of unit tests, and a Jupyter notebook for visual validation.

**Estimated Duration:** 1 day

**Key Tasks:**
- Create the new module `ptycho/workflows/simulation_utils.py`
- Implement and unit test the `load_probe_from_source` and `validate_probe_object_compatibility` helper functions
- Create the new script `scripts/tools/create_hybrid_probe.py` and implement the robust probe mixing algorithm
- Add a comprehensive suite of unit tests for `create_hybrid_probe.py`
- Create a validation notebook (`notebooks/validate_hybrid_probe.ipynb`) to visually compare the original and hybrid probes
- Update `scripts/tools/CLAUDE.md` with documentation for the new tool

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** 
- All unit tests pass
- The `create_hybrid_probe.py` script successfully generates a `hybrid_probe.npy` file from known sources
- The validation notebook visually confirms the hybrid probe has the correct amplitude and phase characteristics

---

### **Phase 2: Enhance Simulation Script and Validate Decoupling**

**Goal:** To integrate the new probe-loading logic into the main simulation script and validate that the object and probe sources can be successfully decoupled for both gridsize=1 and gridsize=2.

**Deliverable:** An enhanced `scripts/simulation/simulate_and_save.py` script with a new `--probe-file` argument, and a new integration test.

**Estimated Duration:** 1 day

**Key Tasks:**
- Modify `scripts/simulation/simulate_and_save.py` to add the optional `--probe-file` argument
- Integrate the `load_probe_from_source` and `validate_probe_object_compatibility` helpers from Phase 1
- Add a new integration test (`tests/test_decoupled_simulation.py`) that:
  - Verifies the probe override works correctly with both `.npy` and `.npz` probe sources
  - Validates that the data pipeline (coordinate generation, Y patch creation) remains physically consistent for both gridsize=1 and gridsize=2
  - Tests error cases (probe larger than object, invalid probe format)
- Update `scripts/simulation/CLAUDE.md` and `README.md` with the new capability

**Dependencies:** Phase 1 must be complete

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:** The new integration test passes, confirming that a simulation run with an external probe produces a valid, trainable dataset that adheres to all data contracts.

---

### **Phase 3: 2x2 Study Orchestration and Execution**

**Goal:** To automate and execute the full 2x2 probe generalization study, which serves as the final, comprehensive integration test of all new components.

**Deliverable:** A new master script `scripts/studies/run_2x2_probe_study.sh` and the completed training and evaluation outputs for all four experimental arms.

**Estimated Duration:** 0.5 days (plus compute time for the study)

**Key Tasks:**
- Create the `run_2x2_probe_study.sh` script with:
  - Robust error handling and checkpointing (skip completed steps)
  - A `--quick-test` flag for rapid validation (fewer images, fewer epochs)
  - An option for parallel execution (`--parallel-jobs`)
  - Progress tracking with estimated completion times
- Execute the script in "quick test" mode to validate the orchestration logic
- Run the full study
- Update `scripts/studies/CLAUDE.md` with documentation for the new study script

**Dependencies:** Phase 1 and Phase 2 must be complete. Requires sufficient compute resources (GPU with ~8GB memory recommended) and disk space (~20GB).

**Implementation Checklist:** `phase_3_checklist.md`

**Success Test:** The `run_2x2_probe_study.sh` script completes a full run without errors. The output directory contains four subdirectories, each with a trained model and an evaluation report.

---

### **Final Phase: Results Aggregation and Documentation**

**Goal:** To analyze the results from the four experiments, generate the final comparison report, and update all relevant project documentation.

**Deliverable:** The final `2x2_study_report.md`, updated documentation, and the initiative archived.

**Estimated Duration:** 0.5 days

**Key Tasks:**
- Write a script to parse the four `comparison_metrics.csv` files and generate the summary table for the `2x2_study_report.md`
- Create side-by-side visualization comparing all four reconstructions
- Update `docs/COMMANDS_REFERENCE.md` and `docs/TOOL_SELECTION_GUIDE.md` with the new capabilities
- Archive all intermediate artifacts (generated probes, simulated datasets) to `probe_study_artifacts/` subdirectory
- Create a brief "lessons learned" document if any significant issues were encountered
- Update `docs/PROJECT_STATUS.md` to move this initiative to the "Completed" section

**Dependencies:** All previous phases complete

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** All R&D plan success criteria are met (PSNR > 20 dB, < 3 dB degradation, smaller gap for gridsize=2), the final report is generated correctly, and the documentation is up-to-date.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [x] **Phase 1:** Core Utilities and Hybrid Probe Generation - 100% complete
- [x] **Phase 2:** Enhance Simulation Script and Validate Decoupling - 100% complete
- [x] **Phase 3:** 2x2 Study Orchestration and Execution - 100% complete
- [ ] **Final Phase:** Results Aggregation and Documentation - 0% complete

**Current Phase:** Final Phase: Results Aggregation and Documentation
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
- **Risk:** The hybrid probe algorithm proves to be physically unsound or creates NaN/Inf values.
  - **Mitigation:** The validation notebook (Phase 1) will allow for visual inspection and numerical validation. If it fails, pivot to using the full experimental probe from fly64 as "Probe B" instead of the hybrid.

- **Risk:** Training runs in the 2x2 study fail or take excessively long.
  - **Mitigation:** The orchestration script's `--quick-test` mode will catch configuration issues early. The script will include checkpointing to allow resuming a failed study. Consider using `@memoize_simulated_data` decorator for expensive operations.

- **Risk:** Memory/disk space exhaustion during the full study.
  - **Mitigation:** Add disk space check before starting. Consider cleanup of intermediate files between runs. Monitor GPU memory usage during quick test.

**Rollback Plan:**
- **Git:** Each phase will be a separate, reviewed commit on the feature branch, allowing for easy reverts.
- **Feature Flag:** The `--probe-file` flag in the simulation script allows the new code path to be optional, maintaining backward compatibility.