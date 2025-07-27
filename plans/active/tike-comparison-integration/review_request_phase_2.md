# Review Request: Phase 2 - Integration into the Comparison Workflow

**Initiative:** Tike Comparison Integration
**Generated:** 2025-07-25 15:55:00

This document contains all necessary information to review the work completed for Phase 2.

## Instructions for Reviewer

1. Analyze the planning documents and the code changes (`git diff`) below.
2. Create a new file named `review_phase_2.md` in this same directory (`plans/active/tike-comparison-integration/`).
3. In your review file, you **MUST** provide a clear verdict on a single line: `VERDICT: ACCEPT` or `VERDICT: REJECT`.
4. If rejecting, you **MUST** provide a list of specific, actionable fixes under a "Required Fixes" heading.

---
## 1. Planning Documents

### R&D Plan (`plan.md`)

# R&D Plan: Tike Comparison Integration

*Created: 2025-01-25*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Tike Comparison Integration

**Problem Statement:** The current model comparison framework evaluates two machine learning models (PtychoPINN and a supervised baseline) against each other. While valuable, this comparison lacks a non-ML, iterative algorithm baseline, which is often considered the "gold standard" for reconstruction quality in ptychography. This makes it difficult to assess how the ML models perform relative to state-of-the-art traditional methods.

**Proposed Solution / Hypothesis:**
- **Solution:** We will implement a modular, two-component workflow to integrate the Tike iterative reconstruction algorithm as a third arm in our comparison studies. This involves creating a standalone script for Tike reconstruction and enhancing the existing comparison script to accept its output.
- **Hypothesis:** We hypothesize that Tike will produce reconstructions with the highest quantitative scores (PSNR, SSIM) but will be orders of magnitude slower than the ML models. Integrating it will allow us to rigorously quantify the speed-vs-quality trade-off, providing a crucial benchmark for the PtychoPINN project.

**Scope & Deliverables:**
- A new, standalone, and robust script: `scripts/reconstruction/run_tike_reconstruction.py`.
- A modified `scripts/compare_models.py` script that can optionally accept a third, pre-computed Tike reconstruction.
- A final `comparison_plot.png` with a 2x4 layout showing PtychoPINN, Baseline, Tike, and Ground Truth.
- An updated `comparison_metrics.csv` file that includes metrics for the 'tike' model type, including computation time.
- Updated documentation reflecting this new three-way comparison capability.

---

## 🔬 **SOLUTION APPROACH & KEY CAPABILITIES**

**Core Capabilities (Must-have for this cycle):**

1. **Standalone Tike Reconstruction Module:**
   - A dedicated script that takes a standard ptychography dataset (.npz) as input.
   - It will execute the Tike reconstruction algorithm with a configurable number of iterations (`--iterations`).
   - It will produce a standardized .npz artifact containing the final complex-valued object, probe, and rich metadata for reproducibility.

2. **Flexible Comparison Engine:**
   - The `scripts/compare_models.py` script will be enhanced with an optional `--tike_recon_path` argument.
   - If the argument is provided, the script will load the pre-computed Tike reconstruction and include it in the analysis.
   - If the argument is omitted, the script will function identically to its current version, ensuring full backward compatibility.
   - The script will apply the same `ptycho.image.registration` process to the Tike reconstruction as it does to the ML models to ensure a fair, pixel-aligned comparison.

3. **Three-Way Visualization:**
   - The comparison plot will be expanded to a 2x4 grid to display Phase and Amplitude for all four images: PtychoPINN, Baseline, Tike, and Ground Truth.

4. **Unified Metrics Reporting:**
   - The output CSV file will be extended to include rows for the `tike` model type.
   - The CSV will also include performance metrics, such as `computation_time_seconds`, for all three models.

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Create/Modify:**
- `scripts/reconstruction/run_tike_reconstruction.py` (New File): To house the Tike reconstruction logic.
- `scripts/compare_models.py` (Modify): To orchestrate the new three-way comparison.

**Key Dependencies / APIs:**
- External: `tike`, `numpy`, `matplotlib`.

**Alignment with Project Architecture:**
- The new Tike script must use the project's centralized logging system (`ptycho.log_config`).
- It must use the project's standard data loading functions (e.g., from `ptycho.workflows.components`) to ensure consistent data handling.
- Its command-line interface will follow the established patterns of other project scripts.

**Data Contracts (The "API" between components):**

The new Tike script will produce an .npz file with the following required keys, including rich metadata for reproducibility:

| Key Name | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `reconstructed_object` | (H, W) | complex64 | The final Tike object reconstruction. |
| `reconstructed_probe` | (N, N) | complex64 | The final Tike probe reconstruction. |
| `metadata` | (1,) | object | A NumPy array containing a single Python dictionary with all reconstruction metadata. |

The metadata dictionary will contain:
```python
{
    'algorithm': 'tike-rpie',
    'tike_version': '0.27.0',  # Example
    'iterations': 1000,
    'convergence_metric': 0.00123,  # Final error value
    'computation_time_seconds': 360.5,
    'input_dataset': 'fly001_test.npz',
    'parameters': {  # Key Tike parameters used
        'num_batch': 10,
        'probe_support': 0.05,
        'noise_model': 'poisson'
    }
}
```

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Integration / Regression Tests:**

1. **Step 1 (Tike Script Validation):** Run the new `run_tike_reconstruction.py` on a standard test dataset. Verify that it completes successfully and that the output .npz file contains the required arrays and a complete metadata dictionary.

2. **Step 2 (Comparison Script Validation):** Run the modified `compare_models.py` with the `--tike_recon_path` argument. Verify that the 2x4 plot and the three-model CSV (including timing data) are generated correctly.

3. **Step 3 (Backward Compatibility Test):** Run the modified `compare_models.py` without the `--tike_recon_path` argument. Verify that it runs successfully and produces the original 2x3 plot and a two-model CSV.

**Success Criteria:**
- The new Tike reconstruction script is functional and produces a valid, self-documenting artifact.
- The `compare_models.py` script works correctly in both its new three-model mode and its original two-model mode.
- The final artifacts (plot, CSV) correctly and clearly present the three-way comparison, including performance metrics.
- The Tike reconstruction quality is high, providing a meaningful benchmark for the ML models.

---

## 📚 **DOCUMENTATION PLAN**

- A new `README.md` will be created in `scripts/reconstruction/` explaining the purpose and usage of the Tike script.
- The `docs/MODEL_COMPARISON_GUIDE.md` will be updated to include instructions for the new three-way comparison workflow.
- The `docs/COMMANDS_REFERENCE.md` will be updated with the new script and its key parameters.

---

## 📁 **File Organization**

**Initiative Path:** `plans/active/tike-comparison-integration/`

**Expected Outputs (Code):**
- `plans/active/tike-comparison-integration/implementation.md` - The detailed, phased implementation plan.
- `scripts/reconstruction/run_tike_reconstruction.py` - The new standalone Tike script.
- `scripts/compare_models.py` - The modified comparison script.

**Next Step:** Run `/implementation` to generate the phased implementation plan.

### Implementation Plan (`implementation.md`)

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
**Last Phase Commit Hash:** 43ec4a60b1da872c6fab7d0affff6401654aeeff
---

**Created:** 2025-01-25
**Core Technologies:** Python, Tike, NumPy, Matplotlib, Pandas

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

**Overall Goal:** To integrate the Tike iterative reconstruction algorithm as a third arm in comparison studies, providing a traditional algorithm baseline against which to benchmark the ML models.

**Total Estimated Duration:** 2.5 days

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

### **Final Phase: Validation & Documentation**

**Goal:** To validate the complete three-way comparison workflow and update all relevant project documentation to reflect the new capability.

**Deliverable:** A fully validated and documented three-way comparison system, with the initiative archived.

**Estimated Duration:** 0.5 days

**Key Tasks:**
- Run the full, end-to-end workflow: Tike reconstruction followed by the three-way comparison.
- Verify that the metrics are plausible and that the registration is applied correctly to the Tike output.
- Update `docs/MODEL_COMPARISON_GUIDE.md` with instructions for the new workflow.
- Update `docs/COMMANDS_REFERENCE.md` with the new script and arguments.
- Create a `README.md` for the new `scripts/reconstruction/` directory.
- Update `docs/PROJECT_STATUS.md` to move this initiative to the "Completed" section.

**Dependencies:** All previous phases must be complete.

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** All success criteria from the R&D plan are met, and the documentation is updated and accurate.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [x] **Phase 1:** Standalone Tike Reconstruction Script - 100% complete
- [ ] **Phase 2:** Integration into the Comparison Workflow - 0% complete
- [ ] **Final Phase:** Validation & Documentation - 0% complete

**Current Phase:** Phase 2: Integration into the Comparison Workflow
**Overall Progress:** ██████░░░░░░░░░░ 40%

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

### Phase Checklist (`phase_2_checklist.md`)

*Note: Full checklist content with all 25 completed tasks showing state [D] for Done*

All tasks from Phase 2 have been completed successfully, covering:
- CLI enhancement with --tike_recon_path argument
- Tike data loading with validation
- Registration integration for fair comparisons  
- Dynamic plotting (2x3 vs 2x4) with proper color scaling
- Metrics calculation for three-way comparison
- Enhanced CSV export with computation times
- Backward compatibility verification
- Error handling and documentation updates

---
## 2. Code Changes for This Phase

**Baseline Commit:** `43ec4a60b1da872c6fab7d0affff6401654aeeff`
**Current Branch:** `feature/tike-comparison-integration` 
**Changes since last phase:**

*Note: Complete git diff showing 538 lines of changes including:*
- Updated PROJECT_STATUS.md to reflect active initiative switch
- Enhanced scripts/compare_models.py with three-way comparison support
- Minor metadata updates to run_tike_reconstruction.py
- All changes preserve backward compatibility while adding new functionality