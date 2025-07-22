# Review Request: Phase 2 - Experimental Probe Integration

**Initiative:** Probe Generalization Study
**Generated:** 2025-07-22 01:18:00

This document contains all necessary information to review the work completed for Phase 2.

## Instructions for Reviewer

1.  Analyze the planning documents and the code changes (`git diff`) below.
2.  Create a new file named `review_phase_2.md` in this same directory (`plans/active/probe-generalization-study/`).
3.  In your review file, you **MUST** provide a clear verdict on a single line: `VERDICT: ACCEPT` or `VERDICT: REJECT`.
4.  If rejecting, you **MUST** provide a list of specific, actionable fixes under a "Required Fixes" heading.

---
## 1. Planning Documents

### R&D Plan (`plan.md`)

# R&D Plan: Probe Generalization Study

*Created: 2025-07-22*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Probe Generalization Study

**Problem Statement:** The impact of using different probe functions (idealized vs. experimental) on the performance of PtychoPINN models, particularly with different overlap constraints (gridsize=1 vs. gridsize=2), is not well understood. Additionally, the viability of the synthetic 'lines' dataset workflow post-refactor needs verification.

**Proposed Solution / Hypothesis:**
- By verifying the synthetic 'lines' simulation path, we can confirm its readiness for controlled experiments.
- By implementing a clear workflow to use an external probe from a file, we can systematically test the model's sensitivity to the probe function.
- We hypothesize that models trained with a realistic experimental probe will show better generalization when tested on experimental data, and that gridsize=2 models will be more robust to variations in the probe function due to the overlap constraint.

**Scope & Deliverables:**
1. Verification that synthetic 'lines' datasets can be generated and trained for both gridsize=1 and gridsize=2.
2. A clear, documented workflow for using a probeGuess from an external .npz file in the simulation pipeline.
3. A final `2x2_comparison_report.md` file summarizing the quantitative results (PSNR, SSIM, FRC50) of the four experimental conditions.
4. A final `2x2_comparison_plot.png` visualizing the reconstructed amplitude and phase for all four conditions.

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Modify/Use:**
- `scripts/simulation/run_with_synthetic_lines.py`: To verify the existing workflow.
- `scripts/simulation/simulate_and_save.py`: This script already supports the required feature. The key is to provide it with an `--input-file` that contains the desired probeGuess and a synthetic objectGuess.
- `scripts/run_comparison.sh`: To execute the training and comparison for each of the four experimental arms.

**The Four Experimental Arms:**
1. **Idealized Probe / Gridsize 1**: Train PtychoPINN (gridsize=1) on a synthetic 'lines' object simulated with the standard default probe.
2. **Idealized Probe / Gridsize 2**: Train PtychoPINN (gridsize=2) on the same 'lines' object with the standard default probe.
3. **Experimental Probe / Gridsize 1**: Train PtychoPINN (gridsize=1) on the same 'lines' object, but simulated using the experimental probe from `datasets/fly64/fly001_64_train_converted.npz`.
4. **Experimental Probe / Gridsize 2**: Train PtychoPINN (gridsize=2) on the same 'lines' object with the experimental probe.

**Key Technical Requirements:**
- Extract experimental probe from existing dataset: `datasets/fly64/fly001_64_train_converted.npz`
- Create automation script to orchestrate all 4 experimental conditions
- Ensure consistent synthetic 'lines' object across all conditions for fair comparison
- Leverage existing comparison framework for standardized metrics and visualization

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Unit / Integration Tests:**
- A new test in `tests/test_simulation.py` that verifies the `run_with_synthetic_lines.py` script successfully generates a valid dataset for both gridsize=1 and gridsize=2.
- Verification that experimental probe extraction and integration works correctly.
- The final 2x2 experiment serves as the ultimate integration test, verifying that the entire pipeline (simulation → training → comparison) works correctly for all four conditions.

**Success Criteria:**
1. The `run_with_synthetic_lines.py` workflow runs without error for both grid sizes.
2. The simulation successfully uses the externally provided experimental probe.
3. All four training runs complete successfully and produce valid models.
4. The final `2x2_comparison_report.md` and `2x2_comparison_plot.png` are generated successfully and contain plausible results for all four experimental arms.
5. Quantitative metrics (PSNR, SSIM, FRC50) show meaningful differences between conditions, validating the experimental design.

**Performance Benchmarks:**
- Each training run should complete within reasonable time bounds (similar to existing generalization studies)
- Memory usage should remain within system constraints
- All generated datasets should conform to data contract specifications

---

## 📁 **File Organization**

**Initiative Path:** `plans/active/probe-generalization-study/`

**Expected Outputs:**
- `plans/active/probe-generalization-study/implementation.md` - Detailed implementation phases
- `experiments/probe-generalization-study/` - Experimental results directory
- `experiments/probe-generalization-study/2x2_comparison_report.md` - Final analysis report
- `experiments/probe-generalization-study/2x2_comparison_plot.png` - Final visualization
- `tests/test_probe_generalization.py` - Unit tests for probe workflows

**Next Step:** Run `/implementation` to generate the phased implementation plan.

### Implementation Plan (`implementation.md`)

<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Probe Generalization Study
**Initiative Path:** `plans/active/probe-generalization-study/`

---
## Git Workflow Information
**Feature Branch:** feature/probe-generalization-study
**Baseline Branch:** devel
**Baseline Commit Hash:** 973d684ad172fe987067b48693f6804ad74facfa
**Last Phase Commit Hash:** e68baaf01658c21921eb95d7af4af71664e232d4
---

**Created:** 2025-07-22
**Core Technologies:** Python, NumPy, TensorFlow, scikit-image, ptychographic simulation

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

- **`plan.md`** - The high-level R&D Plan
  - **`implementation.md`** - This file - The Phased Implementation Plan
    - `phase_1_checklist.md` - Detailed checklist for Phase 1
    - `phase_2_checklist.md` - Detailed checklist for Phase 2  
    - `phase_3_checklist.md` - Detailed checklist for Phase 3
    - `phase_final_checklist.md` - Checklist for the Final Phase

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** To systematically evaluate the impact of probe function variations (idealized vs. experimental) on PtychoPINN model performance across different overlap constraints through a controlled 2x2 experimental study.

**Total Estimated Duration:** 3 days + compute time

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Housekeeping & Workflow Verification**

**Goal:** To perform targeted code cleanup and verify that the existing synthetic 'lines' workflow functions correctly for both gridsize=1 and gridsize=2, adding a new unit test for this capability.

**Deliverable:** A cleaner codebase with a new unit test in `tests/test_simulation.py` that confirms the successful generation of 'lines' datasets for both grid sizes.

**Estimated Duration:** 1 day

**Key Tasks:**
- **Housekeeping:** Execute Phase 1 of the previously defined "Codebase Housekeeping" plan (centralize tests, archive example plans, remove legacy scripts).
- **Verification:** Run the `scripts/simulation/run_with_synthetic_lines.py` script for gridsize=1 and gridsize=2 to confirm it generates valid, trainable datasets.
- **Testing:** Add a new unit test to `tests/test_simulation.py` that automates this verification.

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** The new unit test passes, and `python -m unittest discover -s tests` runs successfully.

---

### **Phase 2: Experimental Probe Integration**

**Goal:** To create a reusable workflow for simulating data using the experimental probe from the fly64 dataset.

**Deliverable:** A new `.npz` file, `simulation_input_experimental_probe.npz`, containing a synthetic 'lines' object and the experimental probeGuess from `datasets/fly64/fly001_64_train_converted.npz`.

**Estimated Duration:** 0.5 days

**Key Tasks:**
- Write a small helper script to load the probeGuess from the fly64 dataset.
- Generate a standard synthetic 'lines' objectGuess.
- Save both arrays into a new .npz file that can be fed into `simulate_and_save.py`.
- Run a small test simulation using this new input file to verify it works.

**Dependencies:** Requires Phase 1 completion.

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:** The command `python scripts/simulation/simulate_and_save.py --input-file simulation_input_experimental_probe.npz` runs successfully.

---

### **Phase 3: Automated 2x2 Study Execution**

**Goal:** To automate and execute the full 2x2 probe generalization study, training all four model configurations.

**Deliverable:** A new orchestration script, `scripts/studies/run_probe_generalization_study.sh`, and the completed training outputs for all four experimental arms.

**Estimated Duration:** 1 day (plus compute time)

**Key Tasks:**
- Create the `run_probe_generalization_study.sh` script.
- The script will execute four separate runs of `run_comparison.sh`, correctly configuring the gridsize and the input data (simulated with either the default or experimental probe).
- Each run will have a distinct output directory (e.g., `probe_study/ideal_gs1`, `probe_study/exp_gs2`).

**Dependencies:** Requires Phase 2 completion.

**Implementation Checklist:** `phase_3_checklist.md`

**Success Test:** The script completes all four training and comparison runs without error, and each output directory contains a valid `comparison_metrics.csv` file.

---

### **Final Phase: Results Aggregation & Documentation**

**Goal:** To analyze the results from the four experiments, generate the final comparison report and plot, and document the findings.

**Deliverable:** The final `2x2_comparison_report.md` and `2x2_comparison_plot.png` artifacts, with the initiative archived.

**Estimated Duration:** 0.5 days

**Key Tasks:**
- Write a Python script to parse the four `comparison_metrics.csv` files.
- Generate the summary table and the 2x2 visualization plot.
- Write a brief analysis of the results in the markdown report.
- Update `docs/PROJECT_STATUS.md` to move this initiative to the "Completed" section.

**Dependencies:** All previous phases complete.

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** All R&D plan success criteria are met, and the final artifacts are generated correctly.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [x] **Phase 1:** Housekeeping & Workflow Verification - 100% complete
- [ ] **Phase 2:** Experimental Probe Integration - 0% complete
- [ ] **Phase 3:** Automated 2x2 Study Execution - 0% complete
- [ ] **Final Phase:** Results Aggregation & Documentation - 0% complete

**Current Phase:** Phase 2: Experimental Probe Integration
**Overall Progress:** ████░░░░░░░░░░░░ 25%

---

## 🚀 **GETTING STARTED**

1.  **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist.
2.  **Begin Implementation:** Follow the checklist tasks in order.
3.  **Track Progress:** Update task states in the checklist as you work.
4.  **Request Review:** Run `/complete-phase` when all Phase 1 tasks are done to generate a review request.

---

## ⚠️ **RISK MITIGATION**

**Potential Blockers:**
- **Risk:** The synthetic 'lines' workflow may fail on one of the grid sizes due to post-refactor changes.
  - **Mitigation:** Phase 1 verification will catch this early, allowing fixes before the main study.
- **Risk:** The experimental probe from fly64 dataset may be incompatible with the simulation pipeline.
  - **Mitigation:** Phase 2 includes validation step with test simulation before proceeding to full study.
- **Risk:** Training runs may fail due to memory constraints or other system issues.
  - **Mitigation:** The orchestration script will include error handling and resume capability.

**Rollback Plan:**
- **Git:** Each phase will be a separate, reviewed commit on the feature branch, allowing for easy reverts.
- **Incremental Validation:** Each phase produces a testable deliverable, preventing error propagation.
- **Existing Infrastructure:** Study leverages proven `run_comparison.sh` framework, minimizing risk of fundamental failures.

### Phase Checklist (`phase_2_checklist.md`)

# Phase 2: Experimental Probe Integration Checklist

**Initiative:** Probe Generalization Study
**Created:** 2025-07-22
**Phase Goal:** To create a reusable workflow for simulating data using the experimental probe from the fly64 dataset.
**Deliverable:** A new `.npz` file, `simulation_input_experimental_probe.npz`, containing a synthetic 'lines' object and the experimental probeGuess from `datasets/fly64/fly001_64_train_converted.npz`.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :------------------------------------------------- |
| **Section 0: Preparation & Analysis**
| 0.A | **Analyze Experimental Probe Structure**          | `[ ]` | **Why:** To understand the format and characteristics of the experimental probe. <br> **How:** Load `datasets/fly64/fly001_64_train_converted.npz` and examine the `probeGuess` array. Check shape, dtype, and complex structure. <br> **Code:** `import numpy as np; data = np.load('datasets/fly64/fly001_64_train_converted.npz'); print(f"ProbeGuess: shape={data['probeGuess'].shape}, dtype={data['probeGuess'].dtype}")` <br> **Expected:** Shape (64, 64), complex64 dtype. |
| 0.B | **Study Synthetic Lines Generation Process**       | `[ ]` | **Why:** To understand how to generate a compatible synthetic 'lines' object. <br> **How:** Review `scripts/simulation/run_with_synthetic_lines.py` lines 45-87. Focus on `generate_and_save_synthetic_input()` function. <br> **Key Points:** Uses `sim_object_image(size=full_object_size)` with `p.set('data_source', 'lines')`, object size is 3.5x probe size. <br> **API:** `ptycho.diffsim.sim_object_image()` and `ptycho.params.set()`. |
| 0.C | **Verify Data Contracts Compliance**              | `[ ]` | **Why:** To ensure the output file will be compatible with `simulate_and_save.py`. <br> **How:** Read `docs/data_contracts.md` section on input file requirements. Confirm required keys: `objectGuess` (complex, large), `probeGuess` (complex, N×N). <br> **Reference:** Both arrays must be complex64, object must be significantly larger than probe for scanning space. |
| **Section 1: Helper Script Implementation**
| 1.A | **Create Script Directory Structure**             | `[ ]` | **Why:** To organize the helper script in a logical location. <br> **How:** Create `scripts/tools/create_experimental_probe_input.py` as a standalone tool. <br> **Pattern:** Follow naming convention from other tools like `visualize_dataset.py`, `split_dataset_tool.py`. <br> **Location:** `scripts/tools/` is the established location for data preparation utilities. |
| 1.B | **Implement Experimental Probe Loading**          | `[ ]` | **Why:** To extract the experimental probe from the fly64 dataset. <br> **How:** Create function `load_experimental_probe(fly64_path: str) -> np.ndarray`. Load the NPZ file, extract `probeGuess`, validate it's complex64. <br> **API:** `np.load(fly64_path)['probeGuess']` <br> **Validation:** Check shape is (64, 64), dtype is complex64, no NaN/inf values. <br> **Error Handling:** Raise clear errors if file missing or probe malformed. |
| 1.C | **Implement Synthetic Lines Object Generation**   | `[ ]` | **Why:** To generate a compatible synthetic object using the same process as the working pipeline. <br> **How:** Create function `generate_synthetic_lines_object(probe_size: int) -> np.ndarray`. Use same logic as `run_with_synthetic_lines.py`. <br> **Steps:** `p.set('data_source', 'lines')`, `p.set('size', object_size)`, call `sim_object_image(size=object_size)`. <br> **Size Rule:** Object size = int(probe_size * 3.5) to match existing workflow. |
| 1.D | **Implement NPZ Output Generation**                | `[ ]` | **Why:** To create the required output file format compatible with `simulate_and_save.py`. <br> **How:** Create function `save_input_file(object_guess: np.ndarray, probe_guess: np.ndarray, output_path: str)`. <br> **Format:** `np.savez(output_path, objectGuess=object_guess.astype(np.complex64), probeGuess=probe_guess.astype(np.complex64))` <br> **Validation:** Ensure both arrays are complex64 before saving. |
| 1.E | **Implement Command Line Interface**              | `[ ]` | **Why:** To make the script usable from command line with proper argument handling. <br> **How:** Use `argparse` with arguments: `--fly64-file` (required), `--output-file` (required), `--probe-size` (optional, default 64). <br> **Example:** `python scripts/tools/create_experimental_probe_input.py --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file simulation_input_experimental_probe.npz` <br> **Validation:** Check input file exists, output directory is writable. |
| **Section 2: Script Integration & Testing**
| 2.A | **Add Error Handling & Logging**                  | `[ ]` | **Why:** To provide clear feedback and handle edge cases gracefully. <br> **How:** Add try/except blocks around file operations and computation steps. Use print statements for progress logging. <br> **Error Cases:** Missing input file, corrupted NPZ data, write permissions, invalid probe dimensions. <br> **Logging:** Print probe shape, object shape, output file path for user verification. |
| 2.B | **Test Script with Fly64 Dataset**                | `[ ]` | **Why:** To verify the script works correctly with real experimental data. <br> **How:** Run `python scripts/tools/create_experimental_probe_input.py --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file simulation_input_experimental_probe.npz`. <br> **Verify:** Output file is created, contains required keys, arrays have correct shapes and types. <br> **Debug:** If errors occur, check file paths, data access permissions, and array manipulations. |
| 2.C | **Validate Output File Structure**                | `[ ]` | **Why:** To ensure the generated file conforms to data contracts and will work with simulation pipeline. <br> **How:** Load the output file and check: keys present (`objectGuess`, `probeGuess`), correct shapes, correct dtypes (complex64), reasonable value ranges. <br> **Code:** `data = np.load('simulation_input_experimental_probe.npz'); print(f"Keys: {list(data.keys())}"); print(f"Object: {data['objectGuess'].shape}, {data['objectGuess'].dtype}"); print(f"Probe: {data['probeGuess'].shape}, {data['probeGuess'].dtype}")` |
| **Section 3: Simulation Pipeline Verification**
| 3.A | **Test Integration with simulate_and_save.py**    | `[ ]` | **Why:** This is the key success criterion - the output must work with the existing simulation pipeline. <br> **How:** Run `python scripts/simulation/simulate_and_save.py --input-file simulation_input_experimental_probe.npz --output-file test_experimental_sim.npz --n-images 100`. <br> **Timeout:** Allow 2-3 minutes for small test. <br> **Expected:** Script completes without errors and generates test_experimental_sim.npz. |
| 3.B | **Validate Simulation Output Structure**          | `[ ]` | **Why:** To confirm the simulation generated a valid, trainable dataset. <br> **How:** Load `test_experimental_sim.npz` and verify it contains all required keys: `objectGuess`, `probeGuess`, `diffraction`, `xcoords`, `ycoords`. <br> **Shapes:** diffraction should be (100, 64, 64), coordinates should be (100,) each. <br> **Reference:** Use `docs/data_contracts.md` as validation checklist. |
| 3.C | **Compare Experimental vs Synthetic Probes**      | `[ ]` | **Why:** To understand the visual and structural differences between probe types for analysis. <br> **How:** Load both the experimental probe and a default synthetic probe. Generate side-by-side amplitude/phase plots. <br> **Code:** Use matplotlib to create 2×2 subplot showing amplitude and phase for both probes. <br> **Save:** Save comparison as `probe_comparison.png` for documentation. |
| **Section 4: Documentation & Cleanup**
| 4.A | **Add Script Documentation**                      | `[ ]` | **Why:** To ensure the tool is usable by others and follows project standards. <br> **How:** Add comprehensive docstring to main script explaining purpose, usage, arguments, and examples. <br> **Include:** Function docstrings following Google style, usage examples in module docstring, error handling documentation. <br> **Reference:** Follow documentation patterns from other tools in `scripts/tools/`. |
| 4.B | **Create Script Usage Documentation**             | `[ ]` | **Why:** To provide users with clear instructions for the new workflow capability. <br> **How:** Consider adding a section to `scripts/tools/README.md` or `scripts/tools/CLAUDE.md` about the new experimental probe input generation capability. <br> **Content:** Brief description, example usage, integration with simulation pipeline. <br> **Optional:** Only if it significantly enhances workflow documentation. |
| 4.C | **Clean Up Test Outputs**                         | `[ ]` | **Why:** To remove temporary files created during testing and verification. <br> **How:** Remove `test_experimental_sim.npz` and any other temporary simulation outputs. <br> **Preserve:** Keep `simulation_input_experimental_probe.npz` as it's the deliverable, and `probe_comparison.png` as documentation. <br> **Document:** Note file locations and purposes in implementation notes. |
| **Section 5: Final Validation & Commitment**
| 5.A | **Run Complete Integration Test**                 | `[ ]` | **Why:** To demonstrate the complete end-to-end workflow before phase completion. <br> **How:** Execute the full pipeline: create input file → run simulation → verify output. Commands: <br> 1. `python scripts/tools/create_experimental_probe_input.py --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file simulation_input_experimental_probe.npz` <br> 2. `python scripts/simulation/simulate_and_save.py --input-file simulation_input_experimental_probe.npz --output-file final_test_experimental_sim.npz --n-images 50` <br> **Success:** Both commands complete successfully, final output file validates correctly. |
| 5.B | **Document Implementation Decisions**             | `[ ]` | **Why:** To record key design choices for future reference and debugging. <br> **How:** Update the "Implementation Notes" section at the bottom of this checklist with: probe loading method, object generation parameters, file format decisions, integration approach. <br> **Include:** Any issues encountered, performance observations, compatibility considerations. |
| 5.C | **Commit Phase 2 Changes**                        | `[ ]` | **Why:** To create a checkpoint for Phase 2 completion and enable Phase 3 work. <br> **How:** Stage changes: `git add .`, commit with descriptive message: `git commit -m "Phase 2: Implement experimental probe integration workflow\n\n- Create create_experimental_probe_input.py tool for extracting fly64 probe\n- Generate synthetic lines object compatible with experimental probe\n- Validate end-to-end simulation pipeline with experimental probe\n- Enable Phase 3 2x2 study execution"`. <br> **Verify:** Confirm git status shows clean working directory after commit. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: `python scripts/simulation/simulate_and_save.py --input-file simulation_input_experimental_probe.npz` runs successfully and produces valid output.
3. The deliverable `simulation_input_experimental_probe.npz` file exists and contains the experimental probe from fly64 dataset plus a synthetic 'lines' object.
4. The new helper script `scripts/tools/create_experimental_probe_input.py` is implemented and documented.
5. Integration testing confirms the experimental probe workflow is compatible with the existing simulation pipeline.

## 📊 Implementation Notes

### Key Design Decisions:
- **Tool Location:** Placed in `scripts/tools/` to follow established project organization for data preparation utilities.
- **Object Generation:** Uses identical logic to `run_with_synthetic_lines.py` to ensure consistency with tested workflow.
- **File Format:** Follows exact NPZ format expected by `simulate_and_save.py` with `objectGuess` and `probeGuess` keys.
- **Probe Size:** Assumes 64×64 probe size to match fly64 dataset, but allows override via command line.

### Expected Challenges:
- **Data Type Consistency:** Ensuring complex64 dtype throughout the pipeline to prevent TensorFlow errors.
- **Array Shape Validation:** Confirming object size is appropriate for probe size to allow adequate scanning range.
- **File Path Handling:** Robust handling of relative/absolute paths and file existence checking.

### Integration Points:
- **Input:** Experimental probe from `datasets/fly64/fly001_64_train_converted.npz`
- **Processing:** Synthetic object generation via `ptycho.diffsim.sim_object_image()`
- **Output:** Compatible input file for `scripts/simulation/simulate_and_save.py`
- **Validation:** End-to-end test ensures Phase 3 can proceed with confidence

### Performance Expectations:
- **Script Execution:** Should complete in under 10 seconds for object generation and file I/O
- **Test Simulation:** 50-100 image simulation should complete in 1-2 minutes
- **Memory Usage:** Minimal - only holds object and probe arrays temporarily in memory

---
## 2. Code Changes for This Phase

**Baseline Commit:** `e68baaf01658c21921eb95d7af4af71664e232d4`
**Current Branch:** `feature/probe-generalization-study`
**Changes since last phase:**

```diff
diff --git a/probe_comparison.png b/probe_comparison.png
new file mode 100644
index 0000000..77c23ac
Binary files /dev/null and b/probe_comparison.png differ
diff --git a/scripts/tools/create_experimental_probe_input.py b/scripts/tools/create_experimental_probe_input.py
new file mode 100755
index 0000000..e07b52e
--- /dev/null
+++ b/scripts/tools/create_experimental_probe_input.py
@@ -0,0 +1,226 @@
+#!/usr/bin/env python3
+"""
+Experimental Probe Input Generator
+
+This tool creates simulation input files by combining experimental probes from the fly64 dataset
+with synthetic 'lines' objects. The output is compatible with the simulate_and_save.py pipeline.
+
+Key Features:
+- Extracts experimental probe from fly64 dataset
+- Generates synthetic 'lines' object using the same process as run_with_synthetic_lines.py
+- Creates NPZ file compatible with simulate_and_save.py data contracts
+- Provides command-line interface for easy integration
+
+Usage:
+    python scripts/tools/create_experimental_probe_input.py \
+        --fly64-file datasets/fly64/fly001_64_train_converted.npz \
+        --output-file simulation_input_experimental_probe.npz
+
+Example Integration:
+    # 1. Create input file with experimental probe
+    python scripts/tools/create_experimental_probe_input.py \
+        --fly64-file datasets/fly64/fly001_64_train_converted.npz \
+        --output-file experimental_input.npz
+    
+    # 2. Run simulation with experimental probe
+    python scripts/simulation/simulate_and_save.py \
+        --input-file experimental_input.npz \
+        --output-file experimental_sim_data.npz \
+        --n-images 2000
+
+Created for: Probe Generalization Study (Phase 2)
+Author: Claude Code Assistant
+"""
+
+import argparse
+import sys
+from pathlib import Path
+import numpy as np
+
+# Import necessary components from the ptycho library
+from ptycho import params as p
+from ptycho.diffsim import sim_object_image
+
+
+def load_experimental_probe(fly64_path: str) -> np.ndarray:
+    """
+    Load the experimental probe from the fly64 dataset.
+    
+    Args:
+        fly64_path: Path to the fly64 NPZ file
+        
+    Returns:
+        The experimental probe as a complex64 array
+        
+    Raises:
+        FileNotFoundError: If the fly64 file doesn't exist
+        KeyError: If the probeGuess key is missing
+        ValueError: If the probe data is malformed
+    """
+    try:
+        print(f"Loading experimental probe from: {fly64_path}")
+        data = np.load(fly64_path)
+        
+        if 'probeGuess' not in data:
+            raise KeyError("probeGuess key not found in fly64 dataset")
+        
+        probe = data['probeGuess']
+        
+        # Validate probe structure
+        if probe.dtype != np.complex64:
+            print(f"Warning: Converting probe dtype from {probe.dtype} to complex64")
+            probe = probe.astype(np.complex64)
+        
+        if len(probe.shape) != 2:
+            raise ValueError(f"Expected 2D probe, got shape {probe.shape}")
+        
+        if probe.shape[0] != probe.shape[1]:
+            raise ValueError(f"Expected square probe, got shape {probe.shape}")
+        
+        # Check for NaN/inf values
+        if not np.isfinite(probe).all():
+            raise ValueError("Probe contains NaN or infinite values")
+        
+        print(f"Successfully loaded probe: shape={probe.shape}, dtype={probe.dtype}")
+        return probe
+        
+    except FileNotFoundError:
+        raise FileNotFoundError(f"Fly64 dataset not found: {fly64_path}")
+    except Exception as e:
+        raise ValueError(f"Error loading experimental probe: {e}")
+
+
+def generate_synthetic_lines_object(probe_size: int) -> np.ndarray:
+    """
+    Generate a synthetic 'lines' object compatible with the experimental probe.
+    
+    Uses the same logic as run_with_synthetic_lines.py to ensure consistency.
+    
+    Args:
+        probe_size: Size of the probe (object will be ~3.5x larger)
+        
+    Returns:
+        The synthetic object as a complex64 array
+    """
+    print(f"Generating synthetic 'lines' object for probe size {probe_size}...")
+    
+    # Calculate object size using same scale factor as run_with_synthetic_lines.py
+    object_scale_factor = 3.5
+    full_object_size = int(probe_size * object_scale_factor)
+    
+    # Set parameters for synthetic object generation
+    p.set('data_source', 'lines')
+    p.set('size', full_object_size)
+    
+    # Generate the synthetic object
+    synthetic_object = sim_object_image(size=full_object_size)
+    synthetic_object = synthetic_object.squeeze().astype(np.complex64)
+    
+    print(f"Generated synthetic object: shape={synthetic_object.shape}, dtype={synthetic_object.dtype}")
+    return synthetic_object
+
+
+def save_input_file(object_guess: np.ndarray, probe_guess: np.ndarray, output_path: str) -> None:
+    """
+    Save the object and probe to an NPZ file compatible with simulate_and_save.py.
+    
+    Args:
+        object_guess: The synthetic object array
+        probe_guess: The experimental probe array
+        output_path: Path where the NPZ file will be saved
+    """
+    print(f"Saving input file to: {output_path}")
+    
+    # Ensure arrays are complex64 for compatibility
+    object_guess = object_guess.astype(np.complex64)
+    probe_guess = probe_guess.astype(np.complex64)
+    
+    # Save using the expected key names for simulate_and_save.py
+    np.savez(
+        output_path,
+        objectGuess=object_guess,
+        probeGuess=probe_guess
+    )
+    
+    print(f"Successfully saved input file with:")
+    print(f"  objectGuess: {object_guess.shape} {object_guess.dtype}")
+    print(f"  probeGuess: {probe_guess.shape} {probe_guess.dtype}")
+
+
+def main():
+    """Main function with command-line interface."""
+    parser = argparse.ArgumentParser(
+        description="Create simulation input file with experimental probe and synthetic object",
+        formatter_class=argparse.RawDescriptionHelpFormatter,
+        epilog="""
+Examples:
+  # Basic usage with fly64 dataset
+  %(prog)s --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file experimental_input.npz
+  
+  # Custom probe size
+  %(prog)s --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file custom_input.npz --probe-size 128
+
+Integration with simulation:
+  # After creating input file, use with simulate_and_save.py
+  python scripts/simulation/simulate_and_save.py --input-file experimental_input.npz --output-file sim_data.npz --n-images 2000
+        """)
+    
+    parser.add_argument(
+        '--fly64-file',
+        required=True,
+        help='Path to fly64 NPZ dataset containing experimental probe'
+    )
+    
+    parser.add_argument(
+        '--output-file',
+        required=True,
+        help='Path for output NPZ file (will be created/overwritten)'
+    )
+    
+    parser.add_argument(
+        '--probe-size',
+        type=int,
+        default=64,
+        help='Expected probe size in pixels (default: 64)'
+    )
+    
+    args = parser.parse_args()
+    
+    try:
+        # Validate input file exists
+        if not Path(args.fly64_file).exists():
+            print(f"Error: Input file does not exist: {args.fly64_file}")
+            return 1
+        
+        # Validate output directory is writable
+        output_path = Path(args.output_file)
+        output_path.parent.mkdir(parents=True, exist_ok=True)
+        
+        # Load experimental probe
+        experimental_probe = load_experimental_probe(args.fly64_file)
+        
+        # Validate probe size matches expectation
+        if experimental_probe.shape[0] != args.probe_size:
+            print(f"Warning: Probe size {experimental_probe.shape[0]} doesn't match expected {args.probe_size}")
+            probe_size = experimental_probe.shape[0]
+        else:
+            probe_size = args.probe_size
+        
+        # Generate synthetic object
+        synthetic_object = generate_synthetic_lines_object(probe_size)
+        
+        # Save combined input file
+        save_input_file(synthetic_object, experimental_probe, args.output_file)
+        
+        print(f"\n✅ Success! Created experimental probe input file: {args.output_file}")
+        print(f"Ready for simulation with: python scripts/simulation/simulate_and_save.py --input-file {args.output_file}")
+        
+        return 0
+        
+    except Exception as e:
+        print(f"❌ Error: {e}")
+        return 1
+
+
+if __name__ == "__main__":
+    sys.exit(main())
\ No newline at end of file
diff --git a/simulation_input_experimental_probe.npz b/simulation_input_experimental_probe.npz
new file mode 100644
index 0000000..c3db414
Binary files /dev/null and b/simulation_input_experimental_probe.npz differ
```