# Phase 3: Automated 2x2 Study Execution Checklist

**Initiative:** Probe Generalization Study
**Created:** 2025-07-22
**Phase Goal:** To automate and execute the full 2x2 probe generalization study, training all four model configurations.
**Deliverable:** A new orchestration script, `scripts/studies/run_probe_generalization_study.sh`, and the completed training outputs for all four experimental arms.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :------------------------------------------------- |
| **Section 0: Preparation & Design**
| 0.A | **Analyze Experimental Setup Requirements**       | `[ ]` | **Why:** To understand the exact configuration needed for each of the four experimental arms. <br> **How:** Review the implementation plan Phase 3 requirements and identify the four experimental configurations: <br> 1. Idealized Probe/Gridsize 1: Standard probe + lines object + gridsize=1 <br> 2. Idealized Probe/Gridsize 2: Standard probe + lines object + gridsize=2 <br> 3. Experimental Probe/Gridsize 1: Fly64 probe + lines object + gridsize=1 <br> 4. Experimental Probe/Gridsize 2: Fly64 probe + lines object + gridsize=2 <br> **Key Insight:** Each arm requires different input data and gridsize configuration. |
| 0.B | **Study run_comparison.sh API & Parameters**      | `[ ]` | **Why:** To understand how to properly invoke run_comparison.sh with the correct parameters for each experimental arm. <br> **How:** Review `scripts/run_comparison.sh` lines 26-47 for usage. Key parameters: `<train_data.npz> <test_data.npz> <output_dir> [--n-train-images N] [--n-test-images N]`. <br> **API:** `./scripts/run_comparison.sh train.npz test.npz output_dir --n-train-images 2000` <br> **Note:** The script will use `configs/comparison_config.yaml` which has `gridsize: 1` by default - we need separate configs for gridsize=2. |
| 0.C | **Design Configuration Strategy for Grid Sizes**  | `[ ]` | **Why:** To handle different gridsize configurations properly since `run_comparison.sh` uses `configs/comparison_config.yaml`. <br> **How:** Plan to create gridsize-specific config files: `configs/comparison_config_gs1.yaml` (gridsize: 1) and `configs/comparison_config_gs2.yaml` (gridsize: 2). <br> **Alternative:** Modify script to pass gridsize as parameter, but config approach is cleaner and follows existing patterns. <br> **Reference:** See `inference_gridsize2_config.yaml` for gridsize=2 configuration example. |
| 0.D | **Verify Phase 2 Deliverable Availability**       | `[ ]` | **Why:** To ensure the experimental probe input file from Phase 2 is available for use. <br> **How:** Verify `simulation_input_experimental_probe.npz` exists in project root from Phase 2. Check it contains `objectGuess` (lines object) and `probeGuess` (fly64 experimental probe). <br> **Command:** `ls -la simulation_input_experimental_probe.npz && python -c "import numpy as np; data=np.load('simulation_input_experimental_probe.npz'); print(f'Keys: {list(data.keys())}'); print(f'Shapes: obj={data[\"objectGuess\"].shape}, probe={data[\"probeGuess\"].shape}')"` <br> **Expected:** File exists with correct NPZ structure from Phase 2. |
| **Section 1: Configuration File Setup**
| 1.A | **Create Gridsize-Specific Configuration Files**  | `[ ]` | **Why:** To ensure proper gridsize configuration for each experimental arm without modifying the base comparison_config.yaml. <br> **How:** Copy `configs/comparison_config.yaml` to create `configs/comparison_config_gs1.yaml` and `configs/comparison_config_gs2.yaml`. Modify the `gridsize` parameter in each: gs1 has `gridsize: 1`, gs2 has `gridsize: 2`. <br> **Commands:** <br> `cp configs/comparison_config.yaml configs/comparison_config_gs1.yaml` <br> `cp configs/comparison_config.yaml configs/comparison_config_gs2.yaml` <br> **Edit:** Change `gridsize: 1` to `gridsize: 2` in gs2 config. |
| 1.B | **Validate Configuration Files**                  | `[ ]` | **Why:** To ensure the configuration files are syntactically correct and contain expected values. <br> **How:** Load each config file with Python yaml parser to verify syntax and check key parameters. <br> **Commands:** <br> `python -c "import yaml; print(yaml.safe_load(open('configs/comparison_config_gs1.yaml'))['gridsize'])"` <br> `python -c "import yaml; print(yaml.safe_load(open('configs/comparison_config_gs2.yaml'))['gridsize'])"` <br> **Expected:** gs1 config returns 1, gs2 config returns 2, no syntax errors. |
| 1.C | **Test Configuration Compatibility**              | `[ ]` | **Why:** To verify the new config files work with the existing training pipeline before using in the full study. <br> **How:** Run a quick test training command with each config to validate compatibility. Use minimal parameters for quick validation. <br> **Commands:** <br> `python scripts/training/train.py --config configs/comparison_config_gs1.yaml --train_data_file datasets/fly64/fly001_64_train_converted.npz --output_dir config_test_gs1 --n_images 50 --nepochs 1` <br> **Note:** This is a smoke test - should start training without errors, then can be interrupted. |
| **Section 2: Data Preparation & Simulation**
| 2.A | **Generate Synthetic Data for Idealized Probe Arms**| `[ ]` | **Why:** To create the training/test datasets for experimental arms 1 & 2 (idealized probe with gridsize 1 & 2). <br> **How:** Use `scripts/simulation/run_with_synthetic_lines.py` to generate datasets. Create separate datasets for consistency and isolation. <br> **Commands:** <br> `python scripts/simulation/run_with_synthetic_lines.py --output-dir probe_study_data/ideal_probe --n-images 3000` <br> **Output:** Creates `probe_study_data/ideal_probe/simulated_data.npz` with synthetic lines object and default probe. <br> **Size:** Generate 3000 images to allow 2000 train + 1000 test split. |
| 2.B | **Generate Synthetic Data for Experimental Probe Arms**| `[ ]` | **Why:** To create training/test datasets for experimental arms 3 & 4 (experimental probe with gridsize 1 & 2). <br> **How:** Use `scripts/simulation/simulate_and_save.py` with the Phase 2 experimental probe input file. <br> **Commands:** <br> `python scripts/simulation/simulate_and_save.py --input-file simulation_input_experimental_probe.npz --output-file probe_study_data/exp_probe/simulated_data.npz --n-images 3000` <br> **Directory:** Create `probe_study_data/exp_probe/` directory structure. <br> **Validation:** Verify output contains same shape diffraction patterns as idealized probe data for fair comparison. |
| 2.C | **Create Train/Test Dataset Splits**             | `[ ]` | **Why:** To create separate train/test files from the simulated data for proper experimental methodology. <br> **How:** Use `scripts/tools/split_dataset_tool.py` to split both ideal and experimental probe datasets. Create 2000 train / 1000 test splits. <br> **Commands:** <br> `python scripts/tools/split_dataset_tool.py probe_study_data/ideal_probe/simulated_data.npz --train_size 2000 --test_size 1000 --train_output probe_study_data/ideal_train.npz --test_output probe_study_data/ideal_test.npz` <br> `python scripts/tools/split_dataset_tool.py probe_study_data/exp_probe/simulated_data.npz --train_size 2000 --test_size 1000 --train_output probe_study_data/exp_train.npz --test_output probe_study_data/exp_test.npz` |
| 2.D | **Validate Dataset Consistency**                  | `[ ]` | **Why:** To ensure all four datasets have consistent properties for fair comparison (same diffraction pattern size, same number of images). <br> **How:** Load all four NPZ files and compare shapes, data types, and key statistics. <br> **Script:** Create small validation script to compare: <br> ```python <br> files = ['probe_study_data/ideal_train.npz', 'probe_study_data/ideal_test.npz', 'probe_study_data/exp_train.npz', 'probe_study_data/exp_test.npz'] <br> for f in files: data = np.load(f); print(f"{f}: diff={data['diffraction'].shape}, dtype={data['diffraction'].dtype}") <br> ``` <br> **Expected:** All diffraction arrays should have shape (N, 64, 64) and same dtype. |
| **Section 3: Orchestration Script Implementation**
| 3.A | **Create Script Directory Structure**             | `[ ]` | **Why:** To organize the orchestration script in the established location with proper structure. <br> **How:** Create `scripts/studies/run_probe_generalization_study.sh` following the naming pattern of existing study scripts. <br> **Reference:** Follow structure from `scripts/studies/run_generalization_study.sh` for consistency. <br> **Permissions:** Ensure script is executable with `chmod +x scripts/studies/run_probe_generalization_study.sh`. |
| 3.B | **Implement Script Header & Documentation**       | `[ ]` | **Why:** To provide clear usage instructions and establish script metadata following project conventions. <br> **How:** Add comprehensive header with purpose, usage, examples, and parameter descriptions. Follow the established format from other study scripts. <br> **Include:** Script purpose, usage syntax, parameter descriptions, examples, author info, and modification date. <br> **Style:** Match the documentation style of `run_generalization_study.sh` and `run_complete_generalization_study.sh`. |
| 3.C | **Implement Argument Parsing & Validation**       | `[ ]` | **Why:** To handle command-line arguments properly and provide user-friendly error messages. <br> **How:** Parse output directory argument (required) and optional parameters. Validate that required data files exist before proceeding. <br> **Arguments:** <br> - `<output_dir>`: Base output directory (required) <br> - `--skip-if-exists`: Skip completed runs (optional) <br> - `--verbose`: Enable detailed logging (optional) <br> **Validation:** Check that probe_study_data/ exists with all four NPZ files. |
| 3.D | **Implement Four-Arm Experiment Logic**           | `[ ]` | **Why:** To orchestrate the execution of all four experimental arms with proper configuration and output organization. <br> **How:** Implement four separate calls to `run_comparison.sh`, each with correct parameters: <br> **Arm 1:** `./scripts/run_comparison.sh probe_study_data/ideal_train.npz probe_study_data/ideal_test.npz $OUTPUT_DIR/ideal_gs1 --config configs/comparison_config_gs1.yaml --n-train-images 2000` <br> **Arm 2:** Similar but with gs2 config and ideal_gs2 output dir <br> **Arms 3 & 4:** Use exp_train.npz/exp_test.npz with appropriate configs <br> **Error Handling:** Each run should be wrapped with error checking and logging. |
| 3.E | **Add Progress Monitoring & Logging**             | `[ ]` | **Why:** To provide clear feedback during the long-running experimental process and enable debugging if issues occur. <br> **How:** Add timestamp logging, progress indicators, and intermediate status checks. Log start/end times for each experimental arm. <br> **Features:** <br> - Print experiment arm being executed <br> - Show estimated time remaining <br> - Log success/failure of each arm <br> - Create summary log file <br> **Format:** Use consistent logging format with timestamps and clear status indicators. |
| **Section 4: Error Handling & Recovery**
| 4.A | **Implement Experiment State Tracking**           | `[ ]` | **Why:** To enable resuming interrupted experiments and provide clear status feedback. <br> **How:** Check for existence of `comparison_metrics.csv` in each arm's output directory to determine completion status. Allow `--skip-if-exists` to skip completed arms. <br> **Logic:** <br> ```bash <br> if [ -f "$ARM_OUTPUT_DIR/comparison_metrics.csv" ] && [ "$SKIP_IF_EXISTS" = "true" ]; then <br>   echo "Arm already completed, skipping..." <br> fi <br> ``` <br> **Benefit:** Allows recovery from partial failures without rerunning completed arms. |
| 4.B | **Add Comprehensive Error Handling**              | `[ ]` | **Why:** To gracefully handle failures and provide actionable error messages for debugging. <br> **How:** Wrap each `run_comparison.sh` call with error checking. Log failure details and continue with remaining arms if possible. <br> **Error Cases:** <br> - Data file not found <br> - Configuration file invalid <br> - Training failure <br> - Disk space issues <br> - Permission problems <br> **Response:** Log error, save context information, attempt to continue with next arm. |
| 4.C | **Implement Resource Validation**                 | `[ ]` | **Why:** To validate system resources before starting the long-running experiment to prevent late-stage failures. <br> **How:** Check available disk space, memory, and data file accessibility before starting any training runs. <br> **Checks:** <br> - Verify all input NPZ files exist and are readable <br> - Check available disk space (estimate ~2GB per arm) <br> - Verify configuration files are valid YAML <br> - Test write permissions in output directory <br> **Thresholds:** Require at least 10GB free space, warn if less than 20GB. |
| **Section 5: Testing & Integration**
| 5.A | **Test Script with Dry-Run Mode**                | `[ ]` | **Why:** To validate the script logic without executing expensive training runs. <br> **How:** Add `--dry-run` parameter that prints all commands without executing them. Test argument parsing, file validation, and command construction. <br> **Implementation:** Add conditional execution: <br> ```bash <br> if [ "$DRY_RUN" != "true" ]; then <br>   eval $COMPARISON_COMMAND <br> else <br>   echo "DRY RUN: $COMPARISON_COMMAND" <br> fi <br> ``` <br> **Validation:** Verify all four commands are constructed correctly with proper file paths and configurations. |
| 5.B | **Execute Quick Integration Test**                | `[ ]` | **Why:** To validate end-to-end functionality with minimal compute resources before full experiment. <br> **How:** Run the script with `--n-train-images 50` and `nepochs 1` modifications to the config files for rapid testing. <br> **Command:** `./scripts/studies/run_probe_generalization_study.sh quick_test_output` <br> **Expected:** All four arms start successfully, create output directories, begin training process. Can be interrupted once training starts. <br> **Validation:** Check that output directories are created with expected structure. |
| 5.C | **Validate Output Directory Structure**           | `[ ]` | **Why:** To ensure the script creates the expected output organization that matches the planned deliverable structure. <br> **How:** After integration test, verify the directory structure matches the planned layout: <br> ```<br> output_dir/<br>   ideal_gs1/<br>     pinn_run/<br>     baseline_run/<br>     comparison_metrics.csv<br>   ideal_gs2/<br>   exp_gs1/<br>   exp_gs2/<br> ```<br> **Check:** Verify each subdirectory contains the expected run_comparison.sh outputs. |
| **Section 6: Production Execution**
| 6.A | **Execute Full Probe Generalization Study**      | `[ ]` | **Why:** This is the main deliverable - execute all four experimental arms with full training parameters. <br> **How:** Run the complete script with full dataset and training parameters: <br> `./scripts/studies/run_probe_generalization_study.sh probe_generalization_results` <br> **Duration:** Expect 2-4 hours total compute time depending on hardware. <br> **Monitoring:** Monitor progress logs, check for errors, verify intermediate outputs are being created correctly. <br> **Resources:** Ensure adequate disk space and CPU/GPU availability. |
| 6.B | **Monitor & Validate Experimental Progress**      | `[ ]` | **Why:** To ensure all four arms complete successfully and produce valid results. <br> **How:** Periodically check the progress of each arm, validate intermediate outputs, and monitor system resources. <br> **Checkpoints:** <br> - Each arm starts successfully <br> - Training progresses normally (loss decreasing) <br> - Comparison metrics are generated <br> - No resource exhaustion occurs <br> **Intervention:** If any arm fails, investigate and rerun if necessary using `--skip-if-exists` to preserve completed arms. |
| 6.C | **Validate Final Experimental Outputs**           | `[ ]` | **Why:** To confirm all four experimental arms produced valid, complete results meeting the success criteria. <br> **How:** Check each output directory for required files: <br> ```bash <br> for arm in ideal_gs1 ideal_gs2 exp_gs1 exp_gs2; do <br>   if [ ! -f "probe_generalization_results/$arm/comparison_metrics.csv" ]; then <br>     echo "ERROR: Missing metrics for $arm" <br>   fi <br> done <br> ``` <br> **Validation:** Verify each `comparison_metrics.csv` contains expected columns and reasonable metric values. |
| **Section 7: Documentation & Verification**
| 7.A | **Document Experimental Configuration**           | `[ ]` | **Why:** To record the exact parameters and configuration used for reproducibility and analysis. <br> **How:** Create `probe_generalization_results/EXPERIMENT_LOG.md` documenting: <br> - Script execution time and duration <br> - Configuration files used <br> - Dataset sizes and parameters <br> - Any issues or anomalies encountered <br> - System specifications (if relevant) <br> **Content:** Include timestamps, file paths, parameter values, and any deviations from planned methodology. |
| 7.B | **Validate Success Criteria Compliance**          | `[ ]` | **Why:** To confirm the phase deliverable meets all specified success criteria before completion. <br> **How:** Systematically verify each success criterion: <br> 1. `run_probe_generalization_study.sh` script exists and is functional <br> 2. All four training runs completed without error <br> 3. Each output directory contains valid `comparison_metrics.csv` <br> 4. Experimental arms used correct probe types and gridsizes <br> **Documentation:** Record verification results in experiment log. |
| 7.C | **Create Results Summary Preview**                | `[ ]` | **Why:** To provide a preliminary overview of results for validation and to prepare for Phase 4 analysis. <br> **How:** Create basic summary of metrics from all four `comparison_metrics.csv` files. Show key metrics (MAE, PSNR, SSIM) for each experimental arm. <br> **Script:** Simple Python script to load and summarize: <br> ```python <br> import pandas as pd <br> arms = ['ideal_gs1', 'ideal_gs2', exp_gs1', 'exp_gs2'] <br> for arm in arms: <br>   df = pd.read_csv(f'probe_generalization_results/{arm}/comparison_metrics.csv') <br>   print(f"{arm}: PSNR_mean={df['pinn_psnr'].mean():.2f}") <br> ``` |
| **Section 8: Final Integration & Commitment**
| 8.A | **Archive Intermediate Data Files**              | `[ ]` | **Why:** To clean up temporary files while preserving essential data for future analysis. <br> **How:** Move simulation intermediate files to organized archive directory. Keep final datasets for potential future use but remove temporary generation files. <br> **Actions:** <br> - Move `probe_study_data/` to `archived_study_data/` <br> - Keep the four train/test NPZ files in main directory <br> - Remove temporary simulation files <br> **Preserve:** All files needed for result reproduction and Phase 4 analysis. |
| 8.B | **Update Documentation References**               | `[ ]` | **Why:** To ensure the new script is discoverable and properly documented in project documentation. <br> **How:** Add reference to `scripts/studies/run_probe_generalization_study.sh` in relevant documentation files: <br> - Add entry to `scripts/studies/README.md` <br> - Update `CLAUDE.md` studies section if appropriate <br> - Consider adding to `docs/COMMANDS_REFERENCE.md` <br> **Content:** Brief description of script purpose and basic usage example. |
| 8.C | **Commit Phase 3 Implementation**                | `[ ]` | **Why:** To create a checkpoint for Phase 3 completion and enable Final Phase work with all results available. <br> **How:** Stage all changes and commit with descriptive message documenting the complete 2x2 experimental study: <br> `git add .` <br> `git commit -m "Phase 3: Complete 2x2 probe generalization study execution\n\n- Implement run_probe_generalization_study.sh orchestration script\n- Execute all four experimental arms (ideal/exp probe × gs1/gs2)\n- Generate complete training and comparison results\n- Validate all success criteria met\n- Enable Final Phase results analysis and documentation"`<br> **Verify:** Clean git status after commit. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The orchestration script `scripts/studies/run_probe_generalization_study.sh` is implemented and functional.
3. All four experimental arms have been executed successfully:
   - Idealized Probe / Gridsize 1 (`probe_generalization_results/ideal_gs1/`)
   - Idealized Probe / Gridsize 2 (`probe_generalization_results/ideal_gs2/`)
   - Experimental Probe / Gridsize 1 (`probe_generalization_results/exp_gs1/`)
   - Experimental Probe / Gridsize 2 (`probe_generalization_results/exp_gs2/`)
4. Each output directory contains a valid `comparison_metrics.csv` file with quantitative results.
5. The phase success test passes: Each experimental arm completed without error and produced comparison metrics.
6. All intermediate data files are properly organized and archived.

## 📊 Implementation Notes

### Four Experimental Arms Configuration:
1. **Idealized Probe / Gridsize 1:**
   - Data: `probe_study_data/ideal_train.npz` / `probe_study_data/ideal_test.npz`
   - Config: `configs/comparison_config_gs1.yaml` (gridsize: 1)
   - Output: `probe_generalization_results/ideal_gs1/`

2. **Idealized Probe / Gridsize 2:**
   - Data: `probe_study_data/ideal_train.npz` / `probe_study_data/ideal_test.npz`  
   - Config: `configs/comparison_config_gs2.yaml` (gridsize: 2)
   - Output: `probe_generalization_results/ideal_gs2/`

3. **Experimental Probe / Gridsize 1:**
   - Data: `probe_study_data/exp_train.npz` / `probe_study_data/exp_test.npz`
   - Config: `configs/comparison_config_gs1.yaml` (gridsize: 1)
   - Output: `probe_generalization_results/exp_gs1/`

4. **Experimental Probe / Gridsize 2:**
   - Data: `probe_study_data/exp_train.npz` / `probe_study_data/exp_test.npz`
   - Config: `configs/comparison_config_gs2.yaml` (gridsize: 2)  
   - Output: `probe_generalization_results/exp_gs2/`

### Key Script Features:
- **Error Recovery:** `--skip-if-exists` flag allows resuming interrupted experiments
- **Progress Monitoring:** Detailed logging with timestamps and status indicators  
- **Resource Validation:** Pre-flight checks for disk space and file accessibility
- **Flexible Configuration:** Separate config files for different gridsize requirements
- **Comprehensive Documentation:** Built-in help and usage examples

### Expected Performance:
- **Total Duration:** 2-4 hours depending on hardware (50 epochs × 4 arms)
- **Disk Usage:** ~8-10GB for all experimental outputs
- **Memory Requirements:** Standard TensorFlow training requirements per arm
- **Parallelization:** Could be enhanced for parallel execution in future versions

### Integration Points:
- **Input:** Phase 2 experimental probe data and synthetic lines generation
- **Processing:** Four independent `run_comparison.sh` executions with proper configuration
- **Output:** Structured results ready for Final Phase aggregation and analysis
- **Validation:** Success measured by presence of valid `comparison_metrics.csv` in each arm

### Risk Mitigation:
- **Partial Failure Recovery:** Skip-if-exists mechanism prevents lost work
- **Resource Monitoring:** Pre-flight validation prevents late-stage resource failures  
- **Configuration Isolation:** Separate config files prevent parameter conflicts
- **Comprehensive Logging:** Detailed logs enable effective debugging of any issues