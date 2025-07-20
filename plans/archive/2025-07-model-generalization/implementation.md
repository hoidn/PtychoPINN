<!-- ACTIVE IMPLEMENTATION PLAN -->
# Phased Implementation Plan

**Project:** Model Generalization Study: Performance vs. Training Set Size

**Core Technologies:** Bash, Python, Pandas, Matplotlib

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

*   **`docs/studies/plan_model_generalization.md`** (The high-level R&D Plan)
    *   **`implementation_model_generalization.md`** (This file - The Phased Implementation Plan)
        *   `phase_1_checklist.md` (Detailed checklist for Phase 1)
        *   `phase_2_checklist.md` (Detailed checklist for Phase 2)
        *   `phase_3_checklist.md` (Detailed checklist for Phase 3)

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** To create a fully automated workflow for studying model generalization by comparing PtychoPINN and baseline performance across varying training set sizes.

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Enhance Single-Run Capability**

**Goal:** To update the core `run_comparison.sh` script to support configurable training and testing set sizes.

**Deliverable:** A modified `run_comparison.sh` that correctly accepts and utilizes `--n-train-images` and `--n-test-images` arguments.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_1_checklist.md`

**Key Tasks Summary:**
*   Add named argument parsing to `run_comparison.sh` for `--n-train-images` and `--n-test-images`
*   Update the script to forward these arguments to the underlying Python training scripts
*   Verify that `scripts/training/train.py` correctly uses the `n_images` parameter
*   Verify that `scripts/run_baseline.py` correctly uses the `n_images` parameter
*   Test the modified script with explicit image count arguments

**Success Test:** All tasks in `phase_1_checklist.md` are marked as done. Running `run_comparison.sh --n-train-images 512 --n-test-images 1000` successfully completes and logs show the correct number of images were used for training and testing.

**Duration:** 1 day

---

### **Phase 2: Create Multi-Run Orchestration**

**Goal:** To build a master script that automates running comparisons across multiple training set sizes.

**Deliverable:** A new script `scripts/studies/run_generalization_study.sh` that executes multiple comparison runs with varying training sizes and organizes outputs.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_2_checklist.md`

**Key Tasks Summary:**
*   Create the `scripts/studies/` directory structure
*   Implement `run_generalization_study.sh` with configurable training size list
*   Add logic to create organized output subdirectories for each run
*   Implement proper error handling and progress reporting
*   Test with a small set of training sizes to verify functionality

**Success Test:** All tasks in `phase_2_checklist.md` are marked as done. Running `run_generalization_study.sh --train-sizes "512 1024"` creates two comparison runs in organized subdirectories with complete outputs.

**Duration:** 1 day

---

### **Phase 3: Implement Results Aggregation and Visualization**

**Goal:** To create tools for collecting metrics from multiple runs and generating comparative visualizations.

**Deliverable:** A Python script `scripts/studies/aggregate_and_plot_results.py` that parses results from all runs and generates publication-ready plots.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_3_checklist.md`

**Key Tasks Summary:**
*   Implement CSV parsing logic to extract metrics from each run's `comparison_metrics.csv`
*   Create configurable plotting with metric selection (`psnr`, `frc50`, `mae`, `mse`)
*   Add support for selecting data component (`phase` or `amp`)
*   Implement logarithmic (base 2) X-axis scaling for training set sizes
*   Generate both the plot and a summary CSV with aggregated data
*   Add comprehensive documentation and usage examples

**Success Test:** All tasks in `phase_3_checklist.md` are marked as done. The script successfully generates a plot showing PtychoPINN vs Baseline performance curves across training set sizes, with configurable metrics and proper axis scaling.

**Duration:** 1-2 days

---

## 📝 **PHASE TRACKING**

- ✅ **Phase 1:** Enhance Single-Run Capability (see `phase_1_checklist.md`)
- ✅ **Phase 2:** Create Multi-Run Orchestration (see `phase_2_checklist.md`)
- ✅ **Phase 3:** Implement Results Aggregation and Visualization (see `phase_3_checklist.md`)

**Current Phase:** All phases complete
**Next Milestone:** A Python script that parses results from multiple runs and generates publication-ready comparison plots.
