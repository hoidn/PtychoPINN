# Review Request: Phase 1 - Standalone Tike Reconstruction Script

**Initiative:** Tike Comparison Integration
**Generated:** 2025-07-25 14:42:37

This document contains all necessary information to review the work completed for Phase 1.

## Instructions for Reviewer

1.  Analyze the planning documents and the code changes (`git diff`) below.
2.  Create a new file named `review_phase_1.md` in this same directory (`plans/active/tike-comparison-integration/`).
3.  In your review file, you **MUST** provide a clear verdict on a single line: `VERDICT: ACCEPT` or `VERDICT: REJECT`.
4.  If rejecting, you **MUST** provide a list of specific, actionable fixes under a "Required Fixes" heading.

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
**Last Phase Commit Hash:** d9d813131120ae61acffdc725bbca47f3db1587e
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
- [ ] **Phase 1:** Standalone Tike Reconstruction Script - 0% complete
- [ ] **Phase 2:** Integration into the Comparison Workflow - 0% complete
- [ ] **Final Phase:** Validation & Documentation - 0% complete

**Current Phase:** Phase 1: Standalone Tike Reconstruction Script
**Overall Progress:** ░░░░░░░░░░░░░░░░ 0%

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

### Phase Checklist (`phase_1_checklist.md`)

# Phase 1: Standalone Tike Reconstruction Script Checklist

**Initiative:** Tike Comparison Integration  
**Created:** 2025-01-25  
**Phase Goal:** To create a robust, reusable, and well-documented command-line script that performs a Tike reconstruction on a given dataset and saves the output in a standardized .npz format with rich metadata.  
**Deliverable:** A new script `scripts/reconstruction/run_tike_reconstruction.py` and a sample `tike_reconstruction.npz` artifact generated from a standard test dataset.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :------------------------------------------------- |
| **Section 0: Preparation & Setup** |
| 0.A | **Create Directory Structure**                     | `[ ]` | **Why:** To establish the correct location for the new script. <br> **How:** Create the directory `scripts/reconstruction/`. <br> **Verify:** The directory `scripts/reconstruction/` exists. |
| 0.B | **Create Script File**                             | `[ ]` | **Why:** To create the file for the new Tike reconstruction script. <br> **How:** Create an empty file `scripts/reconstruction/run_tike_reconstruction.py`. Add a basic shebang `#!/usr/bin/env python3` and a module-level docstring explaining its purpose. <br> **Permissions:** Make the script executable: `chmod +x scripts/reconstruction/run_tike_reconstruction.py`. |
| 0.C | **Review Tike API & Existing Scripts**             | `[ ]` | **Why:** To understand the Tike API and reuse existing patterns. <br> **How:** Review the `tike.ptycho.reconstruct` function signature. Examine the existing `scripts/tikerecon.py` for a baseline implementation of Tike's parameter setup. <br> **Key Insight:** Note the setup for `PtychoParameters`, `RpieOptions`, `ObjectOptions`, etc. |
| **Section 1: Script Implementation** |
| 1.A | **Implement Command-Line Interface**               | `[ ]` | **Why:** To make the script configurable and user-friendly. <br> **How:** Use `argparse`. Add the following arguments: <br> - `input_npz` (required, positional) <br> - `output_dir` (required, positional) <br> - `--iterations` (optional, type int, default 1000) <br> - `--num-gpu` (optional, type int, default 1) <br> - `--quiet`, `--verbose` for logging control. <br> **Reference:** Use `ptycho/cli_args.py` for logging arguments. |
| 1.B | **Integrate Centralized Logging**                 | `[ ]` | **Why:** To align with project standards for logging and traceability. <br> **How:** Import `setup_logging` from `ptycho.log_config`. Call `setup_logging(Path(args.output_dir), **get_logging_config(args))` at the start of the main function. <br> **API:** Use `logging.getLogger(__name__)` to get the logger instance. |
| 1.C | **Implement Data Loading**                         | `[ ]` | **Why:** To load the necessary arrays from the input dataset. <br> **How:** Create a function `load_tike_data(npz_path)`. It should load the `.npz` file and extract `diffraction`, `probeGuess`, `xcoords`, and `ycoords`. Handle potential key name variations (e.g., `diff3d` vs `diffraction`). <br> **API:** Use `numpy.load()`. Add validation to ensure all required keys are present. |
| 1.D | **Implement Tike Parameter Setup**                | `[ ]` | **Why:** To configure the Tike reconstruction algorithm. <br> **How:** Create a function `configure_tike_parameters(...)`. This function will set up `RpieOptions`, `ObjectOptions`, `ProbeOptions`, and `PtychoParameters` based on the loaded data and CLI arguments. <br> **Key Params:** Use `use_adaptive_moment=True`, `noise_model="poisson"`, and `force_centered_intensity=True` for robust reconstruction. |
| 1.E | **Implement Core Reconstruction Logic**           | `[ ]` | **Why:** To execute the Tike algorithm and capture performance metrics. <br> **How:** In the main function, call `tike.ptycho.reconstruct`. Wrap the call with `time.time()` to measure the `computation_time_seconds`. Capture the final error/convergence metric from the result object. |
| **Section 2: Data Contract & Output Generation** |
| 2.A | **Implement Output Data Contract**                | `[ ]` | **Why:** To produce a standardized, reusable artifact for Phase 2. <br> **How:** Create a function `save_tike_results(...)`. It should save a single `.npz` file named `tike_reconstruction.npz` inside the `output_dir`. <br> **API:** Use `numpy.savez_compressed()`. |
| 2.B | **Save Reconstructed Arrays**                      | `[ ]` | **Why:** To store the primary outputs of the reconstruction. <br> **How:** Inside `save_tike_results`, save the final object and probe from the Tike result object. <br> **Keys:** Use the exact key names `reconstructed_object` and `reconstructed_probe`. Ensure they are 2D complex arrays. |
| 2.C | **Assemble and Save Metadata**                    | `[ ]` | **Why:** To ensure reproducibility and provide context for the reconstruction. <br> **How:** Create a Python dictionary containing all the required metadata (algorithm, version, iterations, timing, parameters). Save this dictionary as a single-element object array under the key `metadata`. <br> **API:** `np.savez_compressed(..., metadata=np.array([metadata_dict]))`. |
| 2.D | **Implement Visualization Output**                | `[ ]` | **Why:** To provide immediate visual feedback on the reconstruction quality. <br> **How:** Create a function `save_visualization(...)`. It should generate a 2x2 plot showing the amplitude and phase of the final reconstructed object and probe. Save it as `reconstruction_visualization.png` in the `output_dir`. |
| **Section 3: Validation & Testing** |
| 3.A | **Test with Standard Dataset**                    | `[ ]` | **Why:** To verify the end-to-end functionality of the script. <br> **How:** Run the script on a known-good test dataset, such as `datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz`. Use a small number of iterations (e.g., `--iterations 10`) for a quick test. <br> **Verify:** The script should complete without errors. |
| 3.B | **Validate Output Artifact**                      | `[ ]` | **Why:** To confirm the script produces a file that meets the data contract. <br> **How:** Write a small, temporary validation script or use an interactive session to load the generated `tike_reconstruction.npz`. Check for the presence and correct shapes/dtypes of `reconstructed_object`, `reconstructed_probe`, and `metadata`. <br> **Example Check:** `data = np.load('...'); meta = data['metadata'][0]; assert meta['iterations'] == 10`. |
| 3.C | **Validate Visualization**                        | `[ ]` | **Why:** To ensure the visual feedback is correct. <br> **How:** Manually inspect the generated `reconstruction_visualization.png`. Verify that the plots are plausible and the titles/labels are correct. |
| **Section 4: Documentation & Finalization** |
| 4.A | **Add Comprehensive Docstrings**                  | `[ ]` | **Why:** To make the script maintainable and understandable. <br> **How:** Add a module-level docstring explaining the script's purpose, usage, and I/O. Add clear docstrings to all functions. |
| 4.B | **Create README for Directory**                   | `[ ]` | **Why:** To document the purpose of the new `scripts/reconstruction/` directory. <br> **How:** Create `scripts/reconstruction/README.md`. Explain that this directory holds scripts for generating reconstructions using various algorithms (starting with Tike). Include a usage example for the new script. |
| 4.C | **Commit Phase 1 Changes**                        | `[ ]` | **Why:** To create a clean checkpoint for the completion of Phase 1. <br> **How:** Stage all new and modified files. Commit with a descriptive message: `git commit -m "Phase 1: Implement standalone Tike reconstruction script\n\n- Create run_tike_reconstruction.py with CLI and logging\n- Implement data contract for output NPZ with metadata\n- Add visualization for immediate feedback"`. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: `python scripts/reconstruction/run_tike_reconstruction.py --input-npz <test_data.npz> --output-dir <tike_output> --iterations 10` runs successfully.
3. The output directory `<tike_output>/` contains `tike_reconstruction.npz` and `reconstruction_visualization.png`.
4. The `tike_reconstruction.npz` file is validated and conforms perfectly to the data contract, including the metadata dictionary.
5. The new script is well-documented and follows project coding standards.

---

## 📋 **Data Contract Specification**

For reference, the output `tike_reconstruction.npz` file must contain:

### Required Arrays:
- **`reconstructed_object`**: Complex 2D array `(M, M)` - The final reconstructed object
- **`reconstructed_probe`**: Complex 2D array `(N, N)` - The final reconstructed probe  

### Required Metadata:
- **`metadata`**: Single-element object array containing a dictionary with:
  - `algorithm`: `"tike"`
  - `tike_version`: Version string from `tike.__version__`
  - `iterations`: Number of iterations used
  - `computation_time_seconds`: Float timing measurement
  - `parameters`: Dictionary of all reconstruction parameters used
  - `input_file`: Path to the input NPZ file
  - `timestamp`: ISO format timestamp of reconstruction

---
## 2. Code Changes for This Phase

**Baseline Commit:** `d9d813131120ae61acffdc725bbca47f3db1587e`
**Current Branch:** `feature/tike-comparison-integration`
**Changes since last phase:**

```diff
diff --git a/scripts/reconstruction/README.md b/scripts/reconstruction/README.md
new file mode 100644
index 0000000..a9cdec1
--- /dev/null
+++ b/scripts/reconstruction/README.md
@@ -0,0 +1,88 @@
+# Reconstruction Scripts Directory
+
+This directory contains scripts for generating ptychographic reconstructions using various traditional algorithms, providing baselines for comparison with machine learning models.
+
+## Available Scripts
+
+### `run_tike_reconstruction.py`
+
+A standalone script that performs ptychographic reconstruction using the Tike library's iterative algorithms.
+
+**Purpose:** Generate traditional algorithm reconstructions that can be integrated into model comparison studies, providing a third arm alongside PtychoPINN and baseline models.
+
+**Usage:**
+```bash
+python run_tike_reconstruction.py <input_npz> <output_dir> [options]
+```
+
+**Arguments:**
+- `input_npz`: NPZ file containing `diffraction`, `probeGuess`, `xcoords`, `ycoords`
+- `output_dir`: Directory where results will be saved
+- `--iterations`: Number of reconstruction iterations (default: 1000)
+- `--num-gpu`: Number of GPUs to use (default: 1)
+- `--quiet`: Suppress console output (file logging only)
+- `--verbose`: Enable DEBUG output to console
+
+**Output Files:**
+- `tike_reconstruction.npz`: Standardized NPZ with `reconstructed_object`, `reconstructed_probe`, and `metadata`
+- `reconstruction_visualization.png`: 2x2 plot showing amplitude and phase of results
+- `logs/debug.log`: Complete execution log
+
+**Examples:**
+```bash
+# Basic reconstruction
+python run_tike_reconstruction.py datasets/fly64/test.npz ./tike_output
+
+# Quick test with fewer iterations
+python run_tike_reconstruction.py datasets/fly64/test.npz ./tike_output --iterations 50
+
+# Quiet mode for automation
+python run_tike_reconstruction.py datasets/fly64/test.npz ./tike_output --quiet
+```
+
+**Integration with Comparison Studies:**
+The output `tike_reconstruction.npz` file is designed to integrate seamlessly with the model comparison workflow in Phase 2 of the Tike Comparison Integration initiative.
+
+## Data Contract
+
+All reconstruction scripts in this directory produce NPZ files with the following standardized format:
+
+### Required Arrays
+- **`reconstructed_object`**: Complex 2D array containing the final reconstructed object
+- **`reconstructed_probe`**: Complex 2D array containing the final reconstructed probe
+
+### Required Metadata
+- **`metadata`**: Single-element object array containing a dictionary with:
+  - `algorithm`: String identifying the reconstruction algorithm
+  - `version`: Version string of the algorithm library
+  - `iterations`: Number of iterations performed
+  - `computation_time_seconds`: Float timing measurement
+  - `parameters`: Dictionary of algorithm-specific parameters
+  - `input_file`: Path to the input NPZ file
+  - `timestamp`: ISO format timestamp of reconstruction
+
+This standardized format ensures compatibility with existing model comparison and evaluation workflows.
+
+## Development Notes
+
+**Adding New Algorithms:**
+1. Create a new script following the naming pattern `run_<algorithm>_reconstruction.py`
+2. Follow the same CLI pattern with `argparse` and logging integration
+3. Implement the standardized data contract for output NPZ files
+4. Include comprehensive docstrings and error handling
+5. Add usage examples to this README
+
+**Dependencies:**
+- Scripts should integrate with the project's centralized logging system (`ptycho.log_config`)
+- Use the standard CLI argument helpers (`ptycho.cli_args`)
+- Follow project conventions for error handling and user feedback
+
+## Future Extensions
+
+This directory is designed to accommodate additional reconstruction algorithms:
+- EPIE (Extended Ptychographic Iterative Engine)
+- RAAR (Relaxed Averaged Alternating Reflections)  
+- Other iterative phase retrieval methods
+- GPU-accelerated implementations
+
+Each new algorithm should follow the established patterns for consistency and integration with the broader comparison framework.
\ No newline at end of file
diff --git a/scripts/reconstruction/run_tike_reconstruction.py b/scripts/reconstruction/run_tike_reconstruction.py
new file mode 100755
index 0000000..138bb3b
--- /dev/null
+++ b/scripts/reconstruction/run_tike_reconstruction.py
@@ -0,0 +1,383 @@
+#!/usr/bin/env python3
+"""
+Standalone Tike Reconstruction Script
+
+This script performs ptychographic reconstruction using the Tike iterative algorithm
+and saves the results in a standardized NPZ format with rich metadata for use in
+model comparison studies.
+
+Usage:
+    python run_tike_reconstruction.py <input_npz> <output_dir> [options]
+
+The script produces:
+- tike_reconstruction.npz: Standardized output with reconstructed object/probe and metadata
+- reconstruction_visualization.png: 2x2 plot showing amplitude/phase of results
+"""
+
+import argparse
+import logging
+import time
+import sys
+import os
+from pathlib import Path
+from datetime import datetime
+import numpy as np
+import matplotlib.pyplot as plt
+import tike.ptycho
+import tike.precision
+import tike
+
+# Add ptycho to path for imports
+sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
+from ptycho.log_config import setup_logging
+from ptycho.cli_args import add_logging_arguments, get_logging_config
+
+
+def parse_arguments():
+    """Parse command-line arguments."""
+    parser = argparse.ArgumentParser(
+        description="Standalone Tike ptychographic reconstruction",
+        formatter_class=argparse.RawDescriptionHelpFormatter,
+        epilog="""
+Examples:
+    python run_tike_reconstruction.py input.npz ./tike_output
+    python run_tike_reconstruction.py input.npz ./tike_output --iterations 500 --quiet
+        """
+    )
+    
+    # Required positional arguments
+    parser.add_argument(
+        'input_npz',
+        help='Input NPZ file containing diffraction, probeGuess, xcoords, ycoords'
+    )
+    parser.add_argument(
+        'output_dir',
+        help='Output directory for reconstruction results'
+    )
+    
+    # Optional arguments
+    parser.add_argument(
+        '--iterations',
+        type=int,
+        default=1000,
+        help='Number of reconstruction iterations (default: 1000)'
+    )
+    parser.add_argument(
+        '--num-gpu',
+        type=int,
+        default=1,
+        help='Number of GPUs to use (default: 1)'
+    )
+    
+    # Add standard logging arguments
+    add_logging_arguments(parser)
+    
+    return parser.parse_args()
+
+
+def load_tike_data(npz_path):
+    """
+    Load necessary arrays from input NPZ file for Tike reconstruction.
+    
+    Args:
+        npz_path: Path to input NPZ file
+        
+    Returns:
+        dict: Dictionary containing diffraction, probe, xcoords, ycoords arrays
+        
+    Raises:
+        FileNotFoundError: If NPZ file doesn't exist
+        KeyError: If required keys are missing from NPZ file
+    """
+    logger = logging.getLogger(__name__)
+    
+    if not os.path.exists(npz_path):
+        raise FileNotFoundError(f"Input NPZ file not found: {npz_path}")
+    
+    logger.info(f"Loading data from {npz_path}")
+    
+    with np.load(npz_path) as data:
+        # Handle potential key name variations
+        diffraction_keys = ['diffraction', 'diff3d']
+        diffraction = None
+        for key in diffraction_keys:
+            if key in data:
+                diffraction = data[key]
+                logger.debug(f"Found diffraction data under key: {key}")
+                break
+        
+        if diffraction is None:
+            raise KeyError(f"No diffraction data found. Expected one of: {diffraction_keys}")
+        
+        # Required arrays
+        required_keys = ['probeGuess', 'xcoords', 'ycoords']
+        missing_keys = [key for key in required_keys if key not in data]
+        if missing_keys:
+            raise KeyError(f"Missing required keys: {missing_keys}")
+        
+        result = {
+            'diffraction': diffraction,
+            'probeGuess': data['probeGuess'],
+            'xcoords': data['xcoords'],
+            'ycoords': data['ycoords']
+        }
+        
+        logger.info(f"Loaded data shapes:")
+        for key, arr in result.items():
+            logger.info(f"  {key}: {arr.shape} ({arr.dtype})")
+    
+    return result
+
+
+def configure_tike_parameters(data_dict, iterations, num_gpu):
+    """
+    Configure Tike reconstruction parameters based on loaded data.
+    
+    Args:
+        data_dict: Dictionary from load_tike_data()
+        iterations: Number of iterations to perform
+        num_gpu: Number of GPUs to use
+        
+    Returns:
+        tuple: (data, tike.ptycho.PtychoParameters) configured for reconstruction
+    """
+    logger = logging.getLogger(__name__)
+    logger.info("Configuring Tike parameters...")
+    
+    # Extract and convert data types
+    diffraction = data_dict['diffraction'].astype(tike.precision.floating)
+    probe = data_dict['probeGuess'].astype(tike.precision.cfloating)
+    
+    # Prepare scan coordinates (stack y, x)
+    scan = np.stack([
+        data_dict['ycoords'].astype(tike.precision.floating),
+        data_dict['xcoords'].astype(tike.precision.floating)
+    ], axis=1)
+    
+    # Add batch and other dimensions to probe as expected by Tike
+    probe = probe[np.newaxis, np.newaxis, np.newaxis, :, :]
+    
+    # Create padded object using Tike's automatic padding
+    psi_2d, scan = tike.ptycho.object.get_padded_object(
+        scan=scan,
+        probe=probe,
+    )
+    psi = psi_2d[np.newaxis, :, :]
+    
+    logger.info(f"Created padded object with shape: {psi.shape}")
+    logger.info(f"Updated scan positions shape: {scan.shape}")
+    
+    # Configure algorithm options
+    algorithm_options = tike.ptycho.RpieOptions(
+        num_iter=iterations,
+        num_batch=10,  # Default batch size
+    )
+    
+    # Configure object options with adaptive moment
+    object_options = tike.ptycho.ObjectOptions(
+        use_adaptive_moment=True
+    )
+    
+    # Configure probe options with robust settings
+    probe_options = tike.ptycho.ProbeOptions(
+        use_adaptive_moment=True,
+        probe_support=0.05,
+        force_centered_intensity=True,
+    )
+    
+    # Position options - keep positions fixed for stability
+    position_options = None
+    
+    # Configure exitwave options with Poisson noise model
+    exitwave_options = tike.ptycho.ExitWaveOptions(
+        measured_pixels=np.ones_like(diffraction[0], dtype=bool),
+        noise_model="poisson",
+    )
+    
+    # Assemble parameters
+    parameters = tike.ptycho.PtychoParameters(
+        psi=psi,
+        probe=probe,
+        scan=scan,
+        algorithm_options=algorithm_options,
+        object_options=object_options,
+        probe_options=probe_options,
+        position_options=position_options,
+        exitwave_options=exitwave_options,
+    )
+    
+    logger.info("Tike parameters configured successfully")
+    
+    return diffraction, parameters
+
+
+def save_tike_results(result, output_dir, metadata_dict):
+    """
+    Save Tike reconstruction results in standardized NPZ format.
+    
+    Args:
+        result: Tike reconstruction result object
+        output_dir: Output directory path
+        metadata_dict: Dictionary containing reconstruction metadata
+    """
+    logger = logging.getLogger(__name__)
+    
+    # Extract reconstructed arrays (remove batch dimensions)
+    reconstructed_object = result.psi[0]  # Remove batch dimension
+    reconstructed_probe = result.probe[0, 0, 0, :, :]  # Remove all extra dimensions
+    
+    # Prepare output file path
+    output_file = os.path.join(output_dir, 'tike_reconstruction.npz')
+    
+    # Save in standardized format
+    np.savez_compressed(
+        output_file,
+        reconstructed_object=reconstructed_object,
+        reconstructed_probe=reconstructed_probe,
+        metadata=np.array([metadata_dict])  # Single-element object array
+    )
+    
+    logger.info(f"Saved reconstruction results to {output_file}")
+    
+    # Log array information for verification
+    logger.debug(f"Saved arrays:")
+    logger.debug(f"  reconstructed_object: {reconstructed_object.shape} ({reconstructed_object.dtype})")
+    logger.debug(f"  reconstructed_probe: {reconstructed_probe.shape} ({reconstructed_probe.dtype})")
+    
+    return output_file
+
+
+def save_visualization(result, output_dir):
+    """
+    Generate and save visualization of reconstruction results.
+    
+    Args:
+        result: Tike reconstruction result object
+        output_dir: Output directory path
+        
+    Returns:
+        str: Path to saved visualization file
+    """
+    logger = logging.getLogger(__name__)
+    
+    # Extract arrays for visualization
+    reconstructed_object = result.psi[0]
+    reconstructed_probe = result.probe[0, 0, 0, :, :]
+    
+    # Create 2x2 plot
+    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
+    
+    # Object amplitude
+    ax = axes[0, 0]
+    im = ax.imshow(np.abs(reconstructed_object), cmap='gray')
+    ax.set_title('Reconstructed Object Amplitude')
+    fig.colorbar(im, ax=ax, shrink=0.8)
+    
+    # Object phase
+    ax = axes[0, 1]
+    im = ax.imshow(np.angle(reconstructed_object), cmap='twilight')
+    ax.set_title('Reconstructed Object Phase')
+    fig.colorbar(im, ax=ax, shrink=0.8)
+    
+    # Probe amplitude
+    ax = axes[1, 0]
+    im = ax.imshow(np.abs(reconstructed_probe), cmap='gray')
+    ax.set_title('Reconstructed Probe Amplitude')
+    fig.colorbar(im, ax=ax, shrink=0.8)
+    
+    # Probe phase
+    ax = axes[1, 1]
+    im = ax.imshow(np.angle(reconstructed_probe), cmap='twilight')
+    ax.set_title('Reconstructed Probe Phase')
+    fig.colorbar(im, ax=ax, shrink=0.8)
+    
+    plt.tight_layout()
+    
+    # Save visualization
+    vis_file = os.path.join(output_dir, 'reconstruction_visualization.png')
+    plt.savefig(vis_file, dpi=150, bbox_inches='tight')
+    plt.close()  # Close to prevent blocking
+    
+    logger.info(f"Saved visualization to {vis_file}")
+    
+    return vis_file
+
+
+def main():
+    """Main entry point for the Tike reconstruction script."""
+    # Parse arguments
+    args = parse_arguments()
+    
+    # Create output directory
+    output_dir = Path(args.output_dir)
+    output_dir.mkdir(parents=True, exist_ok=True)
+    
+    # Set up centralized logging
+    logging_config = get_logging_config(args) if hasattr(args, 'quiet') else {}
+    setup_logging(output_dir, **logging_config)
+    
+    logger = logging.getLogger(__name__)
+    logger.info("Starting Tike reconstruction...")
+    logger.info(f"Input: {args.input_npz}")
+    logger.info(f"Output directory: {output_dir}")
+    logger.info(f"Iterations: {args.iterations}")
+    logger.info(f"GPUs: {args.num_gpu}")
+    
+    try:
+        # Load data
+        data_dict = load_tike_data(args.input_npz)
+        
+        # Configure parameters
+        diffraction, parameters = configure_tike_parameters(
+            data_dict, args.iterations, args.num_gpu
+        )
+        
+        # Run reconstruction with timing
+        logger.info("Starting Tike reconstruction...")
+        start_time = time.time()
+        
+        result = tike.ptycho.reconstruct(
+            data=diffraction,
+            parameters=parameters,
+            num_gpu=args.num_gpu,
+        )
+        
+        end_time = time.time()
+        computation_time = end_time - start_time
+        
+        logger.info(f"Reconstruction completed in {computation_time:.2f} seconds")
+        
+        # Prepare metadata
+        metadata = {
+            'algorithm': 'tike',
+            'tike_version': tike.__version__,
+            'iterations': args.iterations,
+            'computation_time_seconds': computation_time,
+            'parameters': {
+                'num_gpu': args.num_gpu,
+                'batch_size': 10,
+                'noise_model': 'poisson',
+                'use_adaptive_moment': True,
+                'force_centered_intensity': True,
+            },
+            'input_file': str(Path(args.input_npz).resolve()),
+            'timestamp': datetime.now().isoformat(),
+        }
+        
+        # Save results
+        npz_file = save_tike_results(result, output_dir, metadata)
+        vis_file = save_visualization(result, output_dir)
+        
+        logger.info("Reconstruction completed successfully!")
+        logger.info(f"Output files:")
+        logger.info(f"  NPZ data: {npz_file}")
+        logger.info(f"  Visualization: {vis_file}")
+        
+    except Exception as e:
+        logger.error(f"Reconstruction failed: {e}")
+        logger.debug("Full traceback:", exc_info=True)
+        sys.exit(1)
+
+
+if __name__ == "__main__":
+    main()
\ No newline at end of file
```