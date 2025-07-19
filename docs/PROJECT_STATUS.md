# Project Status & Initiative Tracker

**Last Updated:** 2025-07-19

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
    *   **R&D Plan:** <doc-ref type="plan">docs/studies/plan_model_generalization.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">docs/studies/implementation_model_generalization.md</doc-ref>

### **Initiative: Image Registration System**
*   **Status:** ✅ Complete
*   **Goal:** To implement automatic image registration for fair model comparisons by detecting and correcting translational misalignments.
*   **Key Deliverables:**
    *   `ptycho/image/registration.py` with sub-pixel phase correlation
    *   Integration into `scripts/compare_models.py` with `--skip-registration` flag
    *   Unified NPZ file format for reconstruction data
*   **Planning Documents:**
    *   **Context Document:** <doc-ref type="plan">docs/refactor/context_priming_registration.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">docs/refactor/plan_registration.md</doc-ref>

### **Initiative: Evaluation Enhancements**
*   **Status:** ✅ Complete
*   **Goal:** To enhance the model evaluation pipeline by adding the SSIM metric and implementing a fairer, more robust pre-preprocessing method for phase comparison.
*   **Key Deliverables:**
    *   SSIM and MS-SSIM metrics integrated into `ptycho/evaluation.py`
    *   Configurable phase alignment methods (plane-fitting and mean subtraction)
    *   Debug visualization capabilities with `--save-debug-images` flag
    *   Enhanced FRC with configurable smoothing
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">docs/refactor/eval_enhancements/plan_eval_enhancements.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">docs/refactor/eval_enhancements/implementation_eval_enhancements.md</doc-ref>

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
    *   **R&D Plan:** `docs/studies/multirun/plan_statistical_generalization.md`
    *   **Implementation Plan:** `docs/studies/multirun/implementation_statistical_generalization.md`

---

## 🚀 **Current Active Initiative**

### **Initiative: Smart Subsampling for Overlap-Based Training**
*   **Status:** 🚀 Active (Phase 1)
*   **Goal:** To replace the spatially biased sequential subsampling with a "group-then-sample" strategy that ensures both physical coherence and spatial representativeness for overlap-based training (`gridsize > 1`).
*   **Key Deliverables:**
    *   Enhanced data loading pipeline in `ptycho/raw_data.py` with group-first sampling strategy
    *   Automated caching mechanism for expensive neighbor-finding operations
    *   Unified `--n-images` command-line argument with intelligent interpretation based on `gridsize`
    *   Updated documentation explaining the new robust sampling behavior
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">docs/initiatives/smart-subsampling/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">docs/initiatives/smart-subsampling/implementation.md</doc-ref>
    *   **Phase 1 Checklist:** <doc-ref type="checklist">docs/initiatives/smart-subsampling/phase_1_checklist.md</doc-ref>
*   **Current Phase:** Phase 1: Core Data Structure Refactoring

---

## 📋 **Recently Completed Initiatives**

### **Initiative: Spatially-Biased Randomized Sampling Study**
*   **Status:** ✅ Complete
*   **Goal:** To enable generalization studies on random samples from specific spatial regions of datasets, rather than just the first N data points.
*   **Key Deliverables:**
    *   `scripts/tools/shuffle_dataset_tool.py` for randomizing dataset order ✅
    *   Updated documentation in `scripts/tools/README.md` and `scripts/studies/QUICK_REFERENCE.md`
    *   Complete generalization study on top half of fly64 dataset
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">docs/sampling/plan_sampling_study.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">docs/sampling/implementation_sampling_study.md</doc-ref>
    *   **Final Phase Checklist:** <doc-ref type="checklist">docs/sampling/final_phase_validation_checklist.md</doc-ref>

---