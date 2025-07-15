# Project Status & Initiative Tracker

**Last Updated:** 2025-01-14

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
    *   **R&D Plan:** `docs/studies/plan_model_generalization.md`
    *   **Implementation Plan:** `docs/studies/implementation_model_generalization.md`

### **Initiative: Image Registration System**
*   **Status:** ✅ Complete
*   **Goal:** To implement automatic image registration for fair model comparisons by detecting and correcting translational misalignments.
*   **Key Deliverables:**
    *   `ptycho/image/registration.py` with sub-pixel phase correlation
    *   Integration into `scripts/compare_models.py` with `--skip-registration` flag
    *   Unified NPZ file format for reconstruction data
*   **Planning Documents:**
    *   **Context Document:** `docs/refactor/context_priming_registration.md`
    *   **Implementation Plan:** `docs/refactor/plan_registration.md`

### **Initiative: Evaluation Enhancements**
*   **Status:** ✅ Complete
*   **Goal:** To enhance the model evaluation pipeline by adding the SSIM metric and implementing a fairer, more robust pre-preprocessing method for phase comparison.
*   **Key Deliverables:**
    *   SSIM and MS-SSIM metrics integrated into `ptycho/evaluation.py`
    *   Configurable phase alignment methods (plane-fitting and mean subtraction)
    *   Debug visualization capabilities with `--save-debug-images` flag
    *   Enhanced FRC with configurable smoothing
*   **Planning Documents:**
    *   **R&D Plan:** `docs/refactor/eval_enhancements/plan_eval_enhancements.md`
    *   **Implementation Plan:** `docs/refactor/eval_enhancements/implementation_eval_enhancements.md`

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

None - Awaiting new R&D plan.

---