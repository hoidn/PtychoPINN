### **Research & Development Plan**

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Model Generalization Study: Performance vs. Training Set Size

**Problem Statement:** The current `run_comparison.sh` workflow is rigid; it uses the entire provided dataset for training and testing, making it impossible to study how model performance scales with the amount of training data.

**Proposed Solution / Hypothesis:** We will enhance the workflow to control training and testing set sizes. We hypothesize that the physics-informed PtychoPINN model will achieve high performance with significantly fewer training images than the purely data-driven supervised baseline, demonstrating better data efficiency and generalization.

**Scope & Deliverables:**
1.  An updated `run_comparison.sh` script that accepts `--n-train-images` and `--n-test-images` arguments.
2.  A new top-level orchestration script, `scripts/studies/run_generalization_study.sh`, that automates the process of running the comparison multiple times with varying training set sizes.
3.  A new Python script, `scripts/studies/aggregate_and_plot_results.py`, to collect metrics from all runs and generate a final comparison plot (`Metric vs. Training Set Size`).
4.  Updated documentation for the new scripts.

---

## 🔬 **EXPERIMENTAL DESIGN & CAPABILITIES**

**Core Capabilities (Must-have for this cycle):**
1.  **Capability 1:** Update `run_comparison.sh` to allow command-line control over the number of training and testing images used during a single run.
2.  **Capability 2:** Create a master shell script (`run_generalization_study.sh`) that defines a series of training set sizes (e.g., 512, 1024, 2048, 4096) and repeatedly calls the modified `run_comparison.sh` for each size, directing outputs into organized subdirectories.
3.  **Capability 3:** Develop a Python script (`aggregate_and_plot_results.py`) to parse the `comparison_metrics.csv` file from each output subdirectory, aggregate the results, and generate a final comparison plot. This script must be configurable to:
    *   Select the metric for the Y-axis (`psnr`, `frc50`, `mae`, `mse`).
    *   Select the data component (`phase` or `amp`), with `phase` as the default.
    *   Use a logarithmic (base 2) scale for the X-axis (Training Set Size).

**Future Work (Out of scope for now):**
*   Extending the study to run across multiple different datasets automatically.
*   Adding command-line controls for hyperparameters (e.g., learning rate, epochs) to the study.

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Modify:**
*   `scripts/run_comparison.sh`: To add parsing for new named arguments (`--n-train-images`, `--n-test-images`) and pass them to the Python scripts.
*   `scripts/training/train.py`: To correctly use the `n_images` parameter from its config (already supported, just needs verification).
*   `scripts/run_baseline.py`: To correctly use the `n_images` parameter from its config (already supported, just needs verification).

**Key Modules to Create:**
*   `scripts/studies/run_generalization_study.sh`: The new top-level orchestrator script.
*   `scripts/studies/aggregate_and_plot_results.py`: The new results aggregation and plotting tool.

**Key Dependencies / APIs:**
*   **Internal:** `ptycho/workflows/components.py` (specifically `load_data`, which already supports the `n_images` argument).
*   **External:** `pandas` and `matplotlib` for the aggregation script. `bash` for the orchestration script.

**Data Requirements:**
*   **Input Data:** This experiment requires a large, pre-split training dataset and a separate, fixed-size testing dataset. The `scripts/tools/split_dataset_tool.py` is the ideal way to generate these inputs.
    *   _e.g., `datasets/fly001_prepared/fly001_final_downsampled_data_train.npz` (large, for sampling)_
    *   _e.g., `datasets/fly001_prepared/fly001_final_downsampled_data_test.npz` (fixed, for evaluation)_
*   **Expected Output Format:** A final PNG image showing two curves (one for PtychoPINN, one for Baseline) on a plot of `Metric vs. Training Set Size`, and a summary `results.csv` file containing the aggregated data.

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Unit Tests / Component Tests:**
*   [ ] **Test Case 1:** Run the modified `run_comparison.sh` with `--n-train-images 512`. Manually inspect the log output to confirm that the training scripts for both models report using only 512 images.
*   [ ] **Test Case 2:** Create a mock directory structure with 2-3 `comparison_metrics.csv` files and run `aggregate_and_plot_results.py` to verify it correctly parses the data and generates a plot. Test the `--metric` and `--part` flags.

**Integration / Regression Tests:**
*   [ ] Run the new `run_generalization_study.sh` with a small list of training sizes (e.g., `--train-sizes "512 1024"`) to ensure the full end-to-end workflow completes without errors and produces the final plot and summary CSV.

**Success Criteria (How we know we're done):**
*   The `run_generalization_study.sh` script successfully completes a multi-run experiment.
*   The final output plot is generated, clearly showing two performance curves against the number of training images, with configurable axes.
*   The data points on the plot are consistent and plausible (e.g., performance generally improves or saturates as training data increases).
*   The new scripts are documented with clear usage instructions.
