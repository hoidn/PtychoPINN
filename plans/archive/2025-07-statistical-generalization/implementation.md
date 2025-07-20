<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Statistical Generalization Study Enhancement

**Core Technologies:** Bash, Python, Pandas, Matplotlib

## 📄 DOCUMENT HIERARCHY

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:
- `plan_statistical_generalization.md` (The high-level R&D Plan)
- `implementation_statistical_generalization.md` (This file - The Phased Implementation Plan)
- `phase_1_checklist_stats.md` (Detailed checklist for Phase 1)
- `phase_2_checklist_stats.md` (Detailed checklist for Phase 2)
- `phase_3_checklist_stats.md` (Detailed checklist for Phase 3)

## 🎯 PHASE-BASED IMPLEMENTATION

**Overall Goal:** To create a fully automated workflow for studying model generalization with statistical rigor by running multiple trials and visualizing median performance with percentile ranges.

## 📋 IMPLEMENTATION PHASES

### Phase 1: Multi-Trial Execution Framework

**Goal:** To update the `run_complete_generalization_study.sh` script to support configurable, sequential multi-trial runs and a new nested output directory structure.

**Deliverable:** A modified `run_complete_generalization_study.sh` that correctly accepts and utilizes a `--num-trials` argument, creating a `trial_N` subdirectory for each run, with all execution happening sequentially.

**Implementation Checklist:** 
- The detailed, step-by-step implementation for this phase is tracked in: [ ] `phase_1_checklist_stats.md`

**Key Tasks Summary:**
1. Add `--num-trials` argument parsing to `run_complete_generalization_study.sh`.
2. Implement a new inner loop that iterates from 1 to `num_trials`.
3. Modify the output path for `run_comparison.sh` to include the trial number (e.g., `train_512/trial_1`).
4. Remove parallel execution logic to ensure sequential execution.
5. Ensure logging correctly identifies the trial number for each run.

**Success Test:** All tasks in `phase_1_checklist_stats.md` are marked as done. Running `run_complete_generalization_study.sh --train-sizes "512" --num-trials 2 --dry-run` shows commands with output directories `.../train_512/trial_1` and `.../train_512/trial_2`.

**Duration:** 1 day

---

### Phase 2: Statistical Aggregation and Plotting

**Goal:** To enhance the `aggregate_and_plot_results.py` script to handle the new directory structure, compute statistics, and generate plots with percentile bands.

**Deliverable:** A modified `aggregate_and_plot_results.py` that can parse the multi-trial results, calculate median and percentiles, and produce a plot showing median performance with a shaded percentile region.

**Implementation Checklist:** 
- The detailed, step-by-step implementation for this phase is tracked in: [ ] `phase_2_checklist_stats.md`

**Key Tasks Summary:**
1. Update the file discovery logic to recursively find `comparison_metrics.csv` in `trial_*` subdirectories.
2. Use `pandas.groupby()` and `.quantile([0.25, 0.5, 0.75])` to compute percentile statistics.
3. Modify the plotting function to use `plt.fill_between` to visualize the interquartile range (25th-75th percentile).
4. Update the exported `results.csv` to include columns for median, p25, and p75.

**Success Test:** All tasks in `phase_2_checklist_stats.md` are marked as done. Running the script on a mock multi-trial dataset produces a plot with shaded percentile bands and a CSV with `_median`, `_p25`, and `_p75` columns.

**Duration:** 1 day

---

### Phase 3: Documentation and Final Validation

**Goal:** To update all relevant documentation, perform a full end-to-end test, and validate the success criteria.

**Deliverable:** A fully validated and documented statistical generalization study workflow.

**Implementation Checklist:** 
- The detailed, step-by-step implementation for this phase is tracked in: [ ] `phase_3_checklist_stats.md`

**Key Tasks Summary:**
1. Update `docs/studies/QUICK_REFERENCE.md` and `README.md` with the new `--num-trials` flag.
2. Run a small-scale end-to-end study (`--train-sizes "512" --num-trials 2`).
3. Verify the final plot shows median lines with percentile bands and aggregated CSV contains median/percentile columns.
4. Update the `PROJECT_STATUS.md` file.

**Success Test:** All tasks in `phase_3_checklist_stats.md` are marked as done. The end-to-end micro-study completes successfully and produces the expected artifacts. All R&D plan success criteria are met.

**Duration:** 1 day

---

## 📝 PHASE TRACKING

- [✅] **Phase 1:** Multi-Trial Execution Framework (see `phase_1_checklist_stats.md`)
- [✅] **Phase 2:** Statistical Aggregation and Plotting (see `phase_2_checklist_stats.md`)
- [✅] **Phase 3:** Documentation and Final Validation (see `phase_3_checklist_stats.md`)

**Current Phase:** All phases complete - Initiative ready for completion

**Next Milestone:** A modified `aggregate_and_plot_results.py` that can parse the multi-trial results, calculate median and percentiles, and produce a plot showing median performance with a shaded percentile region.