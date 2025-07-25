# R&D Plan: Tike Comparison Integration

*Created: 2025-01-25*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Tike Comparison Integration

**Problem Statement:** The current model comparison framework evaluates two machine learning models (PtychoPINN and a supervised baseline) against each other. While valuable, this comparison lacks a non-ML, iterative algorithm baseline, which is often considered the "gold standard" for reconstruction quality in ptychography. This makes it difficult to assess how the ML models perform relative to state-of-the-art traditional methods.

**Proposed Solution / Hypothesis:**
- **Solution:** We will implement a multi-stage, automated workflow that leverages a pre-shuffled test dataset to integrate Tike as a third comparison arm. This involves:
  - Enhancing the standalone Tike reconstruction script to accept an `--n-images` parameter, allowing it to operate on a subset of a given dataset.
  - Modifying the main study script (run_complete_generalization_study.sh) to orchestrate the Tike reconstruction for each trial, using the first N images from the pre-shuffled test set, where N matches the ML models' training size.
  - Updating the comparison script (compare_models.py) to perform a fair evaluation against the correctly subsampled ground truth.
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

**Prerequisites:**
- This workflow assumes the primary test dataset has been pre-shuffled (e.g., using `scripts/tools/shuffle_dataset_tool.py`) to ensure that taking the first N images constitutes a random sample.

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

4. **Step 4 (Full Integration Test):** Run the enhanced `run_complete_generalization_study.sh` with the `--add-tike-arm` flag on a pre-shuffled test dataset to verify the fully automated three-way study workflow.

**Success Criteria:**
- The `run_complete_generalization_study.sh` script, when run with the `--add-tike-arm` flag on a pre-shuffled test dataset, successfully executes the entire three-way study without manual intervention.
- The final aggregated results correctly compare the performance of all three models, with each evaluated against its appropriate test set.
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