# Phased Implementation Plan: Probe Parameterization Study (Corrected)

**Project:** Probe Parameterization Study  
**Initiative Path:** `plans/archive/2025-08-probe-parameterization/`  
**Created:** 2025-08-01  
**Status:** Archived (Post-mortem refactoring plan)

## Git Workflow Information
- **Feature Branch:** `feature/probe-parameterization-study-cleanup`
- **Baseline Branch:** `main` (or `develop`)
- **Baseline Commit Hash:** *(To be filled with the hash of the last commit before starting this cleanup)*
- **Last Phase Commit Hash:** *(To be filled)*

**Core Technologies:** Python, NumPy, TensorFlow, scikit-image, Bash

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the refactoring and finalization of the successful Probe Parameterization Study. It replaces any previous implementation plans for this initiative.

- **`plan.md`** (Revised) - The high-level R&D Plan
- **`implementation.md`** - This file - The Phased Implementation Plan  
- **`phase_1_checklist.md`** - Detailed checklist for Phase 1
- **`phase_2_checklist.md`** - Detailed checklist for Phase 2
- **`phase_final_checklist.md`** - Checklist for the Final Phase

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** To refactor the successful but ad-hoc scripts from the completed probe study into robust, reusable, and well-documented tools. This plan formalizes the two-stage (prepare/execute) architecture that proved successful, fixes the underlying gridsize configuration bug, and cleans up the codebase.

**Total Estimated Duration:** 2-3 days

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Refactor for Reusability and Modularity**

**Goal:** To extract the core logic from the successful experimental scripts into general-purpose, reusable tools and modules, following the modular design of the original plan.

**Deliverable:** A new helper module `ptycho/workflows/simulation_utils.py`, a new standalone tool `scripts/tools/create_hybrid_probe.py`, a revised `simulate_and_save.py`, and a comprehensive suite of unit tests for all new components.

**Estimated Duration:** 1 day

**Key Tasks:**
1. **Create Helper Module:** Create `ptycho/workflows/simulation_utils.py` and implement the `load_probe_from_source` and `validate_probe_object_compatibility` functions.
2. **Create Hybrid Probe Tool:** Create the standalone tool `scripts/tools/create_hybrid_probe.py`, ensuring it is well-documented and uses the new helper module.
   - The tool must enforce that both source probes have identical dimensions and raise a ValueError if they do not match. It should not perform any automatic resizing.
   - The tool's primary use case will be to combine the amplitude from a synthetic probe with the phase from an experimental probe.
3. **Enhance Simulation Script:** Revise `scripts/simulation/simulate_and_save.py` to add the `--probe-file` argument, integrating the new helper functions for probe loading and validation.
4. **Add Unit Tests:** Create `tests/workflows/test_simulation_utils.py` and `tests/tools/test_create_hybrid_probe.py` to ensure the new tools are robust and reliable.
5. **Improve Subsampling:** Apply the performance patch to `ptycho/raw_data.py` to integrate the more efficient "sample-then-group" logic for gridsize > 1.

**Dependencies:** None (first phase)  
**Implementation Checklist:** `phase_1_checklist.md`  
**Success Test:** All new unit tests pass. The `create_hybrid_probe.py` tool and the enhanced `simulate_and_save.py` script function correctly as standalone components.

---

### **Phase 2: Implement the Two-Stage Workflow**

**Goal:** To create the new, robust two-stage workflow by implementing dedicated scripts for data preparation and experiment execution, thereby permanently fixing the gridsize configuration bug.

**Deliverable:** Two new scripts: `scripts/studies/prepare_2x2_study.py` for data generation and `scripts/studies/run_2x2_study.sh` for experiment execution.

**Estimated Duration:** 1 day

**Key Tasks:**
1. **Create Preparation Script:** Implement `scripts/studies/prepare_2x2_study.py`. This script will use the tools from Phase 1 to:
   - Generate the `default_probe.npy` and `hybrid_probe.npy`.
   - Loop through all four experimental conditions.
   - For each condition, call `simulate_and_save.py` twice (with different seeds) to generate independent `train_data.npz` and `test_data.npz` files.

2. **Create Execution Script:** Implement `scripts/studies/run_2x2_study.sh`. This script will:
   - Take the study directory as input.
   - Loop through the four subdirectories created by the preparation script.
   - For each subdirectory, launch `ptycho_train` and `compare_models.py` in isolated subprocesses, passing the correct `--gridsize` parameter.

3. **Enhance compare_models.py:** Add the `detect_gridsize_from_model_dir` function as a safety measure to ensure the evaluation step is robust against any misconfiguration.

**Dependencies:** Phase 1 must be complete.  
**Implementation Checklist:** `phase_2_checklist.md`  
**Success Test:** The new two-script workflow runs successfully in `--quick-test` mode. A check of the logs confirms that each training run was initiated with the correct gridsize parameter.

---

### **Final Phase: Validation, Documentation, and Cleanup**

**Goal:** To validate the new workflow end-to-end, update all relevant documentation to reflect the new best practices, and clean up all obsolete files from the previous attempt.

**Deliverable:** A fully validated and documented workflow, an updated `PROJECT_STATUS.md`, and a clean repository state.

**Estimated Duration:** 0.5 days

**Key Tasks:**
1. **End-to-End Validation:** Run the full `prepare_2x2_study.py` followed by `run_2x2_study.sh` on a small scale to verify the entire pipeline.

2. **Update Documentation:**
   - Update `docs/DEVELOPER_GUIDE.md` to explain the gridsize bug and the necessity of the two-stage, process-isolated workflow.
   - Update `docs/COMMANDS_REFERENCE.md` and `docs/TOOL_SELECTION_GUIDE.md` to include the new, permanent tools.
   - Create a "Lessons Learned" document summarizing the findings of the initiative.

3. **Code Cleanup:** Delete all obsolete scripts and modules from the previous attempt (e.g., the old monolithic study script, temporary helpers).

4. **Project Archival:** Update `docs/PROJECT_STATUS.md` to reflect the completion of the initiative and archive the planning documents.

**Dependencies:** All previous phases must be complete.  
**Implementation Checklist:** `phase_final_checklist.md`  
**Success Test:** The new workflow is fully documented, the old files are removed, and the `PROJECT_STATUS.md` is up-to-date. The project is left in a clean, robust state, ready for future studies.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- **Phase 1:** Refactor for Reusability and Modularity - 0% complete
- **Phase 2:** Implement the Two-Stage Workflow - 0% complete  
- **Final Phase:** Validation, Documentation, and Cleanup - 0% complete

**Current Phase:** Phase 1: Refactor for Reusability and Modularity  
**Overall Progress:** ░░░░░░░░░░░░░░░░ 0%

---

## 🚀 **GETTING STARTED**

1. **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist.
2. **Begin Implementation:** Follow the checklist tasks in order.
3. **Track Progress:** Update task states in the checklist as you work.
4. **Request Review:** Run `/complete-phase` when all Phase 1 tasks are done to generate a review request.