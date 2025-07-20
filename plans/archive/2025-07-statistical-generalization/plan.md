# Research & Development Plan: Statistical Generalization Study

## 🎯 OBJECTIVE & HYPOTHESIS

**Project/Initiative Name:** Statistical Generalization Study Enhancement

**Problem Statement:** The current model generalization study produces single-point estimates for each performance metric, which are susceptible to noise from random weight initializations and training dynamics. This makes it difficult to assess the true performance and variance of the models.

**Proposed Solution / Hypothesis:** By running multiple training trials for each model at each training set size, we can calculate robust statistics (median and percentiles) for each metric. We hypothesize that using median as the central tendency measure will be more robust to outliers than mean, and that percentiles (25th and 75th) will provide a better representation of the performance distribution. This will provide a more reliable measure of model performance, reveal the variance in training outcomes, and allow for more confident conclusions when comparing models.

**Scope & Deliverables:**
- An updated `run_complete_generalization_study.sh` script that accepts a `--num-trials` argument to orchestrate multiple training runs per configuration.
- A new, nested output directory structure to store the results of each trial.
- An updated `aggregate_and_plot_results.py` script capable of parsing the new directory structure, calculating median and percentiles (25th, 75th) for each metric, and exporting these statistics.
- A new version of the generalization plot that visualizes the median performance with a shaded region representing the 25th to 75th percentile range (interquartile range).
- Updated documentation (`QUICK_REFERENCE.md`, `README.md`) reflecting the new capability.

## 🔬 EXPERIMENTAL DESIGN & CAPABILITIES

**Core Capabilities (Must-have for this cycle):**
1. **Multi-Trial Execution:** The `run_complete_generalization_study.sh` script must be enhanced to loop `m` times for each training size, where `m` is a new `--num-trials` argument (defaulting to 5). It must create a new subdirectory for each trial (e.g., `train_512/trial_1/`, `train_512/trial_2/`, etc.). The execution will be sequential: all trials for a given training size will complete before moving to the next training size.
2. **Statistical Aggregation:** The `aggregate_and_plot_results.py` script must be updated to recursively find all `comparison_metrics.csv` files within the new `trial_*` subdirectories for each training size. It must then use `pandas.groupby()` to calculate the median and percentiles (25th, 50th, 75th) of each metric for every (model, training size) combination.
3. **Enhanced Visualization:** The plotting function within `aggregate_and_plot_results.py` must be modified to use the calculated median and percentiles. It will plot the median as a solid line and use `matplotlib.pyplot.fill_between` to draw a shaded region representing the 25th to 75th percentile range (interquartile range).

**Future Work (Out of scope for now):**
- Implementation of more advanced statistical significance tests (e.g., t-tests, ANOVA) between models at each data point.
- Parallel execution of trials (not prioritized due to GPU memory constraints and added complexity).

## 🛠️ TECHNICAL IMPLEMENTATION DETAILS

**Architectural Note:** The implementation will use sequential execution for all trials and training sizes. This simplifies the script logic, avoids complex process management, and aligns with the practical constraint that GPU training typically requires exclusive access to GPU memory. All trials for a given training size will complete before moving to the next training size.

**Key Modules to Modify:**
- `scripts/studies/run_complete_generalization_study.sh`: To add the main trial loop and create the new directory structure.
- `scripts/studies/aggregate_and_plot_results.py`: To update file discovery logic, implement statistical aggregation (`.groupby().quantile([0.25, 0.5, 0.75])`), and enhance the plotting function.
- `docs/studies/QUICK_REFERENCE.md`: To document the new `--num-trials` flag and its impact.

**Key Dependencies / APIs:**
- **Internal:** The existing `run_comparison.sh` script will be called within the new loop.
- **External:** `pandas` for data aggregation, `matplotlib` for plotting with `fill_between`.

**Data Requirements:**
- **Input Data:** Same as before (a large, prepared training and test dataset).
- **Expected Output Format:**
  - **Directory Structure:** `output_dir/train_SIZE/trial_N/`, where `N` runs from 1 to `num_trials`.
  - **Aggregated CSV (`results.csv`):** Will now contain new columns, e.g., `psnr_phase_median`, `psnr_phase_p25`, `psnr_phase_p75`.
  - **Plot (`.png`):** Will show lines with shaded percentile bands (25th-75th).

## ✅ VALIDATION & VERIFICATION PLAN

**Unit Tests / Checks:**
1. **Test Script Logic:** Run `run_complete_generalization_study.sh --num-trials 2 --dry-run` and verify that the printed commands show the correct nested output directories (`trial_1`, `trial_2`).
2. **Test Aggregation Logic:** Create a mock directory structure with fake `comparison_metrics.csv` files containing known values (e.g., psnr of 10, 15, 20, 25, 30). Run `aggregate_and_plot_results.py` and assert that the output CSV contains a `psnr_median` of 20, `psnr_p25` of 15, and `psnr_p75` of 25.

**Integration / Regression Tests:**
1. **Run a "Micro-Study":** Execute the full workflow with minimal settings: `run_complete_generalization_study.sh --train-sizes "512" --num-trials 2 --skip-data-prep`.
   - Verify that the script completes without errors.
   - Verify that the final plot is generated and contains a shaded percentile region (25th-75th) around the data point for each model.

**Success Criteria (How we know we're done):**
1. The `run_complete_generalization_study.sh` script successfully accepts a `--num-trials` argument and creates the corresponding number of trial subdirectories for each training size.
2. The `aggregate_and_plot_results.py` script successfully parses the new nested directory structure.
3. The final `results.csv` file contains columns for the median and percentiles of each metric (e.g., `psnr_amp_median`, `psnr_amp_p25`, `psnr_amp_p75`).
4. The final plot (`.png`) correctly displays the median metric values as lines and the interquartile range (25th-75th percentile) as a shaded region around the lines.
5. The `QUICK_REFERENCE.md` is updated with the new `--num-trials` flag and an example.