# Project Status & Initiative Tracker

**Last Updated:** 2025-07-22

This document provides a high-level overview of the major development initiatives for the PtychoPINN project. It tracks completed work and outlines the current active initiative.

---

## ✅ **Completed Initiatives**

### **Initiative: Model Generalization Study**
*   **Status:** ✅ Complete
*   **Goal:** To create a workflow for studying how model performance scales with training data size.
*   **Key Deliverables:**
    *   `scripts/run_comparison.sh` enhanced with `--n-train-images` and `--n-test-images` flags.
    *   `scripts/studies/run_generalization_study.sh` for orchestrating multi-run experiments.
    *   `scripts/studies/aggregate_and_plot_results.py` for visualizing results.
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/archive/2025-07-model-generalization/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/archive/2025-07-model-generalization/implementation.md</doc-ref>

### **Initiative: Image Registration System**
*   **Status:** ✅ Complete
*   **Goal:** To implement automatic image registration for fair model comparisons by detecting and correcting translational misalignments.
*   **Key Deliverables:**
    *   `ptycho/image/registration.py` with sub-pixel phase correlation
    *   Integration into `scripts/compare_models.py` with `--skip-registration` flag
    *   Unified NPZ file format for reconstruction data
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/archive/2025-07-registration-refactor/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/archive/2025-07-registration-refactor/implementation.md</doc-ref>
    *   **Context Document:** <doc-ref type="plan">plans/archive/2025-07-registration-refactor/context_priming_registration.md</doc-ref>

### **Initiative: Evaluation Enhancements**
*   **Status:** ✅ Complete
*   **Goal:** To enhance the model evaluation pipeline by adding the SSIM metric and implementing a fairer, more robust pre-preprocessing method for phase comparison.
*   **Key Deliverables:**
    *   SSIM and MS-SSIM metrics integrated into `ptycho/evaluation.py`
    *   Configurable phase alignment methods (plane-fitting and mean subtraction)
    *   Debug visualization capabilities with `--save-debug-images` flag
    *   Enhanced FRC with configurable smoothing
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/archive/2025-07-evaluation-enhancements/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/archive/2025-07-evaluation-enhancements/implementation.md</doc-ref>

---
*(Add more completed initiatives here as they are finished)*
---

### **Initiative: Statistical Generalization Study**
*   **Status:** ✅ Complete
*   **Goal:** To enhance the generalization study workflow to support multiple training trials per configuration, enabling robust statistical analysis (median, percentiles) of model performance.
*   **Key Deliverables:**
    *   Enhanced `run_complete_generalization_study.sh` with `--num-trials` argument
    *   Multi-trial directory structure (`train_SIZE/trial_N/`)
    *   Statistical aggregation in `aggregate_and_plot_results.py` with median/percentiles
    *   Updated documentation with multi-trial examples
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/archive/2025-07-statistical-generalization/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/archive/2025-07-statistical-generalization/implementation.md</doc-ref>

---

## 🚀 **Current Active Initiative**

### **Initiative: Probe Generalization Study**
**Path:** `plans/active/probe-generalization-study/`
**Branch:** `feature/probe-generalization-study` (baseline: devel)
**Started:** 2025-07-22
**Current Phase:** Phase 3: Automated 2x2 Study Execution
**Progress:** ████████░░░░░░░░ 50%
**Next Milestone:** Orchestration script and completed training outputs for all four experimental arms
**R&D Plan:** <doc-ref type="plan">plans/active/probe-generalization-study/plan.md</doc-ref>
**Implementation Plan:** <doc-ref type="plan">plans/active/probe-generalization-study/implementation.md</doc-ref>

**Goal:** To understand the impact of different probe functions (idealized vs. experimental) on PtychoPINN model performance across different overlap constraints (gridsize=1 vs. gridsize=2).

**Key Deliverables:**
- Verification of synthetic 'lines' dataset workflow for both gridsizes
- 2x2 experimental matrix comparing idealized/experimental probes with gridsize 1/2
- Quantitative comparison report with PSNR, SSIM, and FRC50 metrics
- Visualization plots for all four experimental conditions

---

## 📋 **Recently Completed Initiatives**

### **Initiative: Grouping-Aware Subsampling for Overlap-Based Training**
*   **Status:** ✅ Complete
*   **Goal:** To replace the spatially biased sequential subsampling with a "group-then-sample" strategy that ensures both physical coherence and spatial representativeness for overlap-based training (`gridsize > 1`).
*   **Key Deliverables:**
    *   Enhanced data loading pipeline in `ptycho/raw_data.py` with group-first sampling strategy ✅
    *   Automated caching mechanism for expensive neighbor-finding operations ✅
    *   Unified `--n-images` command-line argument with intelligent interpretation based on `gridsize` ✅
    *   Updated documentation explaining the new robust sampling behavior ✅
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">docs/initiatives/smart-subsampling/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">docs/initiatives/smart-subsampling/implementation.md</doc-ref>
    *   **Final Phase Checklist:** <doc-ref type="checklist">docs/initiatives/smart-subsampling/final_phase_checklist.md</doc-ref>

### **Initiative: Spatially-Biased Randomized Sampling Study**
*   **Status:** ✅ Complete
*   **Goal:** To enable generalization studies on random samples from specific spatial regions of datasets, rather than just the first N data points.
*   **Key Deliverables:**
    *   `scripts/tools/shuffle_dataset_tool.py` for randomizing dataset order ✅
    *   Updated documentation in `scripts/tools/README.md` and `scripts/studies/QUICK_REFERENCE.md`
    *   Complete generalization study on top half of fly64 dataset
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/archive/2025-07-sampling-study/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/archive/2025-07-sampling-study/implementation.md</doc-ref>

### **Initiative: MS-SSIM Correction**
*   **Status:** ✅ Complete (Migrated)
*   **Goal:** To correct and enhance the MS-SSIM metric implementation for ptychographic reconstruction evaluation.
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/archive/2025-07-ms-ssim-correction/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/archive/2025-07-ms-ssim-correction/implementation.md</doc-ref>

---