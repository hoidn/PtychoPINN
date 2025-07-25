<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Tike Comparison Integration
**Initiative Path:** `plans/active/tike-comparison-integration/`

---
## Git Workflow Information
**Feature Branch:** feature/tike-comparison-integration
**Baseline Branch:** docstrings
**Baseline Commit Hash:** d9d813131120ae61acffdc725bbca47f3db1587e
**Last Phase Commit Hash:** 785435b72b95d0d0f4de8f9f7f04d8bb5e9b6e6f
---

**Created:** 2025-01-25
**Completed:** 2025-07-25
**Core Technologies:** Python, Tike, NumPy, Matplotlib, Pandas

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

- **`plan.md`** - The high-level R&D Plan
  - **`implementation.md`** - This file - The Phased Implementation Plan
    - `phase_1_checklist.md` - Detailed checklist for Phase 1
    - `phase_2_checklist.md` - Detailed checklist for Phase 2
    - `phase_3_checklist.md` - Checklist for Phase 3
    - `phase_4_checklist.md` - Checklist for Phase 4

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** To integrate the Tike iterative reconstruction algorithm as a third arm in comparison studies, providing a traditional algorithm baseline against which to benchmark the ML models.

**Total Estimated Duration:** 3.5 days

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Standalone Tike Reconstruction Script**

**Goal:** To create a robust, reusable, and well-documented command-line script that performs a Tike reconstruction on a given dataset and saves the output in a standardized .npz format with rich metadata.

**Deliverable:** A new script `scripts/reconstruction/run_tike_reconstruction.py` and a sample `tike_reconstruction.npz` artifact generated from a standard test dataset.

**Estimated Duration:** 1 day

**Key Tasks:**
- Create the new directory structure: `scripts/reconstruction/`.
- Implement the script with a clear command-line interface (`--input-npz`, `--output-dir`, `--iterations`).
- Integrate with the project's standard data loading and centralized logging systems.
- Implement the core Tike reconstruction logic, making it configurable.
- Implement the data contract for the output .npz file, ensuring it contains `reconstructed_object`, `reconstructed_probe`, and a `metadata` dictionary with timing, version, and parameter information.

**Dependencies:** None (first phase).

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** The command `python scripts/reconstruction/run_tike_reconstruction.py --input-npz <test_data.npz> --output-dir <tike_output>` runs successfully, producing an .npz file that conforms to the data contract.

---

### **Phase 2: Integration into the Comparison Workflow**

**Goal:** To modify the existing `scripts/compare_models.py` script to optionally accept the Tike reconstruction artifact from Phase 1 and seamlessly include it in the evaluation and visualization.

**Deliverable:** An updated `scripts/compare_models.py` script capable of generating a 2x4 comparison plot and a CSV file with metrics for all three models (PtychoPINN, Baseline, Tike).

**Estimated Duration:** 1 day

**Key Tasks:**
- Add an optional `--tike_recon_path` argument to `compare_models.py`.
- Implement logic to load the `reconstructed_object` from the Tike .npz file if the path is provided.
- **Crucially, apply the same `ptycho.image.registration` function to the Tike reconstruction** to ensure a fair, pixel-aligned comparison against the ground truth.
- Modify the plotting function to generate a 2x4 grid.
- Update the CSV saving logic to include results for the 'tike' model type and to report computation times for all models.
- Ensure full backward compatibility when the `--tike_recon_path` argument is not used.

**Dependencies:** Requires a valid `tike_reconstruction.npz` artifact from Phase 1.

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:**
1. Running `compare_models.py` with the `--tike_recon_path` argument produces the correct 2x4 plot and a three-model CSV.
2. Running `compare_models.py` *without* the new argument produces the original 2x3 plot and a two-model CSV, demonstrating no regressions.

---

### **Phase 3: Validation & Documentation**

**Goal:** To validate the complete three-way comparison workflow and update all relevant project documentation to reflect the new capability.

**Deliverable:** A fully validated and documented three-way comparison system, ready for production use.

**Estimated Duration:** 0.5 days

**Key Tasks:**
- Run the full, end-to-end workflow: Tike reconstruction followed by the three-way comparison.
- Verify that the metrics are plausible and that the registration is applied correctly to the Tike output.
- Update `docs/MODEL_COMPARISON_GUIDE.md` with instructions for the new workflow.
- Update `docs/COMMANDS_REFERENCE.md` with the new script and arguments.
- Create a `README.md` for the new `scripts/reconstruction/` directory.
- Update `docs/PROJECT_STATUS.md` to move this initiative to the "Completed" section.

**Dependencies:** All previous phases must be complete.

**Implementation Checklist:** `phase_3_checklist.md`

**Success Test:** All success criteria from the R&D plan are met, and the documentation is updated and accurate.

---

### **Phase 4: Full Integration into Generalization Study Script**

**Goal:** To enhance `run_complete_generalization_study.sh` to fully automate the three-way comparison, including on-the-fly test set subsampling and Tike reconstruction for each trial.

**Deliverable:** An updated `run_complete_generalization_study.sh` script that can execute the entire three-way study with a single command.

**Estimated Duration:** 1 day

**Key Tasks:**
- Create a new `subsample_dataset_tool.py` to generate reproducible test sets for Tike.
- Add an `--add-tike-arm` flag to the main study script.
- Integrate the Tike reconstruction step into the main trial loop.
- Modify `compare_models.py` to handle the dual-test-set evaluation logic correctly.

**Dependencies:** All previous phases must be complete.

**Implementation Checklist:** `phase_4_checklist.md`

**Success Test:** A single command (`./run_complete_generalization_study.sh --add-tike-arm ...`) successfully executes the entire three-way comparison study.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [x] **Phase 1:** Standalone Tike Reconstruction Script - 100% complete
- [x] **Phase 2:** Integration into the Comparison Workflow - 100% complete
- [x] **Phase 3:** Validation & Documentation - 100% complete
- [x] **Phase 4:** Full Integration into Generalization Study Script - 100% complete

**Current Phase:** All Phases Complete
**Overall Progress:** ████████████████ 100%

---

## 🚀 **GETTING STARTED**

1. **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist.
2. **Begin Implementation:** Follow the checklist tasks in order.
3. **Track Progress:** Update task states in the checklist as you work.
4. **Request Review:** Run `/complete-phase` when all Phase 1 tasks are done to generate a review request.

---

## ⚠️ **RISK MITIGATION**

**Potential Blockers:**
- **Risk:** Tike algorithm may have dependency conflicts with the existing environment.
  - **Mitigation:** Phase 1 isolates Tike-specific logic, allowing for focused dependency resolution without impacting the main codebase.
- **Risk:** The Tike reconstruction output might have a different coordinate system or alignment than the ML models.
  - **Mitigation:** Phase 2 explicitly includes a task to apply the project's standard registration (`ptycho.image.registration`) to the Tike output, ensuring a fair comparison.
- **Risk:** Long computation times for Tike could slow down development and testing.
  - **Mitigation:** The modular design allows the Tike reconstruction to be run once and reused. The script will have a configurable `--iterations` parameter to allow for quick, low-quality runs during testing.

**Rollback Plan:**
- **Git:** Each phase will be a separate, reviewed commit on the feature branch, allowing for easy reverts.
- **Feature Flag:** The optional `--tike_recon_path` argument allows the new code to be disabled by simply omitting it, maintaining full backward compatibility.