# Phase 4: Full Integration into Generalization Study Script (Simplified Approach) Checklist

**Initiative:** Tike Comparison Integration  
**Created:** 2025-07-25  
**Phase Goal:** To enhance run_complete_generalization_study.sh to automate the three-way comparison, leveraging a pre-shuffled test dataset and existing subsampling logic.  
**Deliverable:** An updated run_complete_generalization_study.sh script that can execute the entire three-way study with a single command, assuming a pre-shuffled test set.

## ✅ Task List

**Instructions:**
- Work through tasks in order. Dependencies are noted in the guidance column.
- The "How/Why & API Guidance" column contains all necessary details for implementation.
- Update the State column as you progress: [ ] (Open) -> [P] (In Progress) -> [D] (Done).

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| **Section 0: Preparation & Prerequisite Verification** |
| 0.A | Verify Pre-shuffled Test Data | [ ] | **Why:** This entire simplified approach depends on having a randomized test dataset. <br> **How:** Manually run scripts/tools/shuffle_dataset_tool.py on the standard test set to create the required input for this workflow. <br> **Command:** `python scripts/tools/shuffle_dataset_tool.py <test_data.npz> <shuffled_test_data.npz> --seed 42`. <br> **Documentation:** Add a note in the study script's help text that the test data should be pre-shuffled for this workflow. |
| **Section 1: Enhancing run_tike_reconstruction.py** |
| 1.A | Add --n-images Argument | [ ] | **Why:** To allow the Tike script to perform reconstructions on a subset of the data. <br> **How:** In scripts/reconstruction/run_tike_reconstruction.py, add an optional --n-images argument to its argparse setup. <br> **Logic:** In the script's data loading function, if --n-images is provided, slice all per-scan arrays (like diffraction, xcoords, ycoords) to use only the first N entries. |
| **Section 2: Enhancing run_complete_generalization_study.sh** |
| 2.A | Add New CLI Arguments | [ ] | **Why:** To enable and configure the Tike comparison arm. <br> **How:** Add the same arguments as before: --add-tike-arm and --tike-iterations. |
| 2.B | Modify Main Trial Loop | [ ] | **Why:** To add the Tike reconstruction step for each trial using the simplified subsampling. <br> **How:** Inside the main trial loop, add a conditional block for --add-tike-arm. This block will call run_tike_reconstruction.py and pass the current train_size to its new --n-images flag. <br> **Example Call:** `python .../run_tike_reconstruction.py "$TEST_DATA" "$tike_output_dir" --iterations "$TIKE_ITERATIONS" --n-images "$train_size"`. |
| 2.C | Modify Comparison Loop | [ ] | **Why:** To pass the correct Tike reconstruction path to the comparison script. <br> **How:** This remains the same as the previous plan. The call to compare_models.py will be enhanced to include the --tike_recon_path pointing to the artifact generated in the step above. |
| **Section 3: Enhancing compare_models.py for Fair Evaluation** |
| 3.A | Modify Test Data Handling | [ ] | **Why:** To ensure each model is evaluated against the correct ground truth data. <br> **How:** The script will now need to handle three sets of data for evaluation: <br> 1. ML Models: Use the full test set (--test_data) with --n-test-images applied. <br> 2. Tike Model: Use the Tike reconstruction artifact (--tike_recon_path). <br> 3. Tike's Ground Truth: Load the full test set, but then take an in-memory slice of the per-scan arrays (xcoords, ycoords) to match the train_size Tike was run on. This is crucial for align_for_evaluation. |
| 3.B | Adapt align_for_evaluation Call for Tike | [ ] | **Why:** To align Tike's reconstruction against the correct, sparsely-sampled ground truth region. <br> **How:** When evaluating Tike, call align_for_evaluation using: <br> - reconstruction_image: The loaded Tike reconstruction. <br> - ground_truth_image: The full objectGuess from the main test set. <br> - scan_coords_yx: The subsampled scan coordinates corresponding to the Tike run. |
| **Section 4: Validation and Documentation** |
| 4.A | Run a Small-Scale 3-Way Study | [ ] | **Why:** To validate the complete, simplified, automated workflow. <br> **How:** First, create the pre-shuffled test set. Then, execute the modified run_complete_generalization_study.sh with a small configuration: --train-sizes "512" --num-trials 2 --add-tike-arm. <br> **Verify:** The script should complete successfully and produce the correct final artifacts. |
| 4.B | Validate Artifacts | [ ] | **Why:** To confirm the outputs are correct. <br> **How:** Check the output directory. Verify that tike_run subdirectories exist. Inspect the final comparison_metrics.csv to ensure it contains 'tike' entries. Check the final plot to ensure it shows three models. |
| 4.C | Update All Relevant Documentation | [ ] | **Why:** To document the new capability and its prerequisite. <br> **How:** Update scripts/studies/README.md, docs/MODEL_COMPARISON_GUIDE.md, etc. Crucially, add a note explaining that for a three-way study, the main test dataset is assumed to be pre-shuffled to ensure random sampling. |
| 4.D | Commit Phase 4 Changes | [ ] | **Why:** To finalize the integration. <br> **How:** Stage all changes. Commit with a message: git commit -m "Phase 4: Integrate Tike arm into study script via simplified subsampling\n\n- Enhance Tike script to accept --n-images\n- Update study script to orchestrate three-way comparison\n- Assumes pre-shuffled test data for random sampling" |

## 🎯 **Current Status**
- **Current Task:** Task 0.A - Verify Pre-shuffled Test Data
- **Progress:** 0% (0/12 tasks completed)
- **Next Step:** Begin Section 0 prerequisite verification