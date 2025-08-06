# Project Status & Initiative Tracker

**Last Updated:** 2025-08-06

This document provides a high-level overview of the major development initiatives for the PtychoPINN project. It tracks completed work and outlines the current active initiative.

---

## 📍 **Current Active Initiative**

**Name:** Probe Parameterization Study - Refactoring Phase  
**Path:** `plans/archive/2025-08-probe-parameterization/`  
**Branch:** TBD (to be created from current branch)  
**Started:** 2025-08-06 (Reinstated)  
**Current Phase:** Phase 1 - Refactor for Reusability and Modularity  
**Progress:** ░░░░░░░░░░░░░░░░ 0% (Planning complete, implementation ready to begin)  
**Next Milestone:** Create robust, reusable tools with proper process isolation  
**R&D Plan:** <doc-ref type="plan">plans/archive/2025-08-probe-parameterization/plan.md</doc-ref>  
**Implementation Plan:** <doc-ref type="plan">plans/archive/2025-08-probe-parameterization/implementation.md</doc-ref>  
**Current Checklist:** <doc-ref type="checklist">plans/archive/2025-08-probe-parameterization/phase_1_checklist.md</doc-ref>

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
    *   **R&D Plan:** <doc-ref type="plan">plans/examples/2025-07-registration-refactor/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/examples/2025-07-registration-refactor/implementation.md</doc-ref>
    *   **Context Document:** <doc-ref type="plan">plans/examples/2025-07-registration-refactor/context_priming_registration.md</doc-ref>

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

---

## 📋 **Recently Completed Initiatives**

### **Initiative: High-Performance Patch Extraction Refactoring**
*   **Status:** ✅ **Complete** - Completed 2025-08-06
*   **Goal:** To optimize patch extraction using batched operations for 4-5x performance improvement
*   **Key Deliverables:**
    *   Batched patch extraction implementation with 4-5x speedup ✅
    *   Feature flag control via `use_batched_patch_extraction` ✅
    *   Comprehensive equivalence testing to 1e-6 tolerance ✅
    *   Memory profiling and performance reporting ✅
    *   Updated documentation and migration guides ✅
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/active/high-performance-patch-extraction/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/active/high-performance-patch-extraction/implementation.md</doc-ref>

### **Initiative: Simulation Workflow Unification**
*   **Status:** ✅ **Complete** - Completed 2025-08-03
*   **Goal:** To fix gridsize > 1 crashes by refactoring the simulation pipeline to use explicit, modular orchestration
*   **Key Deliverables:**
    *   Refactored `simulate_and_save.py` with modular workflow ✅
    *   Comprehensive test suite for gridsize validation ✅
    *   Deprecation warnings on legacy `RawData.from_simulation` ✅
    *   Updated documentation with migration guides ✅
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/examples/2025-08-simulation-workflow-unification/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/examples/2025-08-simulation-workflow-unification/implementation.md</doc-ref>
    *   **Summary:** <doc-ref type="summary">plans/examples/2025-08-simulation-workflow-unification/implementation_summary.md</doc-ref>

### **Initiative: Remove TensorFlow Addons Dependency**
*   **Status:** ✅ **Complete** - Completed 2025-07-27
*   **Goal:** To remove the deprecated TensorFlow Addons dependency by implementing native TensorFlow replacements for `tfa.image.translate` and `tfa.image.gaussian_filter2d`.
*   **Key Deliverables:**
    *   Native `translate_core()` function using `tf.raw_ops.ImageProjectiveTransformV3` ✅
    *   Native `gaussian_filter2d()` implementation with exact TFA compatibility ✅
    *   Comprehensive documentation of interpolation differences ✅
    *   Edge-aware test suite for validation ✅
    *   Complete removal of tensorflow-addons dependency ✅
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/archive/2025-07-remove-tf-addons/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/archive/2025-07-remove-tf-addons/implementation.md</doc-ref>
    *   **Session Documentation:** <doc-ref type="session">plans/archive/2025-07-remove-tf-addons/session_documentation.md</doc-ref>
    *   **Summary:** <doc-ref type="summary">plans/archive/2025-07-remove-tf-addons/summary.md</doc-ref>

### **Initiative: Tike Comparison Integration**
*   **Status:** ✅ **Complete** (All Phases 1-4) - Completed 2025-07-25
*   **Goal:** To integrate the Tike iterative reconstruction algorithm as a third arm in comparison studies, providing a traditional algorithm baseline against which to benchmark the ML models.
*   **Key Deliverables:**
    *   **Phase 1-3:** Standalone Tike reconstruction script (`scripts/reconstruction/run_tike_reconstruction.py`) ✅
    *   **Phase 1-3:** Enhanced comparison framework supporting three-way comparisons (`scripts/compare_models.py`) ✅
    *   **Phase 1-3:** Performance metrics including computation time tracking ✅
    *   **Phase 1-3:** Rich metadata for reproducibility ✅
    *   **Phase 1-3:** Complete documentation including README, command reference, and model comparison guide ✅
    *   **Phase 1-3:** Coordinate convention handling for cross-algorithm compatibility ✅
    *   **Phase 1-3:** Backward compatibility verification ✅
    *   **Phase 1-3:** Full validation workflow with 100/1000 iteration testing ✅
    *   **Phase 4:** Generalization study integration with `--add-tike-arm` flag ✅
    *   **Phase 4:** Parameterized Tike iterations with `--tike-iterations` argument ✅
    *   **Phase 4:** Pre-shuffled dataset compatibility and automated subsampling ✅
    *   **Phase 4:** Complete three-way comparison workflow validation ✅
*   **Final Achievement:** Single-command three-way generalization studies via `./run_complete_generalization_study.sh --add-tike-arm`
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/archive/tike-comparison-integration/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/archive/tike-comparison-integration/implementation.md</doc-ref>
    *   **Phase 4 Implementation Log:** <doc-ref type="log">PHASE_4_IMPLEMENTATION_LOG.md</doc-ref>

### **Initiative: Probe Generalization Study**
*   **Status:** ✅ Complete (Moved from active)
*   **Goal:** To understand the impact of different probe functions (idealized vs. experimental) on PtychoPINN model performance across different overlap constraints (gridsize=1 vs. gridsize=2).
*   **Key Deliverables:**
    *   Verification of synthetic 'lines' dataset workflow for both gridsizes
    *   2x2 experimental matrix comparing idealized/experimental probes with gridsize 1/2
    *   Quantitative comparison report with PSNR, SSIM, and FRC50 metrics
    *   Visualization plots for all four experimental conditions
*   **Planning Documents:**
    *   **R&D Plan:** <doc-ref type="plan">plans/active/probe-generalization-study/plan.md</doc-ref>
    *   **Implementation Plan:** <doc-ref type="plan">plans/active/probe-generalization-study/implementation.md</doc-ref>

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