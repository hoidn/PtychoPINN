# Phase 2: Implement the Two-Stage Workflow Checklist

**Initiative:** Probe Parameterization Study (Corrected)  
**Created:** 2025-08-01  

**Phase Goal:** To create the new, robust two-stage workflow by implementing dedicated scripts for data preparation and experiment execution, thereby permanently fixing the gridsize configuration bug.

**Deliverable:** Two new scripts: `scripts/studies/prepare_2x2_study.py` for data generation and `scripts/studies/run_2x2_study.sh` for experiment execution.

## ✅ **Task List**

**Instructions:**
- Work through tasks in order. Dependencies are noted in the guidance column.
- The "How/Why & API Guidance" column contains all necessary details for implementation.
- Update the State column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| **Section 1: Data Preparation Script Implementation** | | | |
| 1.A | Create prepare_2x2_study.py structure | [D] | **Why:** To implement the centralized data preparation stage that prevents configuration bugs.<br>**How:** Create `scripts/studies/prepare_2x2_study.py` with argparse setup. Arguments: `--output-dir` (required), `--object-source` (default: synthetic lines), `--quick-test` flag (fewer images/epochs), `--gridsize-list` (default: "1,2"). Add logging setup from `ptycho.log_config`.<br>**File:** `scripts/studies/prepare_2x2_study.py` |
| 1.B | Implement probe extraction and creation | [D] | **Why:** To generate the two probe variants (idealized and hybrid) needed for the study.<br>**How:** Generate idealized probe using `ptycho.probe.get_default_probe()`. Extract experimental probe from source dataset using `load_probe_from_source` from simulation_utils. Use `scripts/tools/create_hybrid_probe.py` as subprocess to create hybrid probe combining idealized amplitude with experimental phase. Save both as `idealized_probe.npy` and `hybrid_probe.npy` in output directory. Log probe characteristics (shape, dtype, amplitude range).<br>**Dependencies:** Phase 1 tools must be complete |
| 1.C | Implement synthetic object generation | [D] | **Why:** To create a controlled object for the 2x2 study using the existing synthetic lines generator.<br>**How:** Use `scripts/simulation/run_with_synthetic_lines.py` to generate base synthetic object. Alternatively, call the synthetic lines generation function directly from `ptycho.datagen`. Save as `synthetic_input.npz` in output directory. Ensure object is significantly larger than probe (e.g., 224x224 object for 64x64 probe).<br>**File:** Uses existing synthetic data generation |
| 1.D | Implement 4-condition simulation loop | [D] | **Why:** To generate all training and test datasets for the 2x2 experimental matrix.<br>**How:** Create nested loops: for each gridsize (1,2) and each probe type (idealized, hybrid), create subdirectory (e.g., `gs1_idealized/`, `gs2_hybrid/`). For each condition, call `simulate_and_save.py` twice: once with seed=42 for training data, once with seed=43 for test data. Use appropriate `--n-images` based on `--quick-test` flag.<br>**Dependencies:** Requires enhanced simulate_and_save.py from Phase 1 |
| **Section 2: Execution Script Implementation** | | | |
| 2.A | Create run_2x2_study.sh structure | [D] | **Why:** To implement the isolated execution stage that prevents gridsize configuration leakage.<br>**How:** Create `scripts/studies/run_2x2_study.sh` as bash script. Arguments: `--study-dir` (required), `--quick-test` flag, `--parallel` flag. Add robust error handling, logging to study directory, and progress tracking. Use `set -euo pipefail` for strict error handling.<br>**File:** `scripts/studies/run_2x2_study.sh` |
| 2.B | Implement condition detection and validation | [D] | **Why:** To automatically detect prepared experimental conditions and validate completeness.<br>**How:** Scan study directory for subdirectories matching pattern `gs[12]_(idealized|hybrid)/`. For each found condition, verify presence of `train_data.npz` and `test_data.npz`. Log detected conditions and any missing files. Exit with error if any required files are missing.<br>**Verification:** Script should detect exactly 4 conditions for a complete study |
| 2.C | Implement isolated training execution | [D] | **Why:** To train models in separate processes to prevent configuration contamination.<br>**How:** For each condition, extract gridsize from directory name. Launch `ptycho_train` in subprocess with correct `--gridsize`, `--train_data_file`, `--test_data_file`, and `--output_dir` arguments. Capture stdout/stderr to condition-specific log files. Use `wait` to ensure completion before proceeding to next condition.<br>**Critical:** Each training run must be in a completely separate subprocess |
| 2.D | Implement isolated evaluation execution | [D] | **Why:** To run comparisons in isolated processes with correct gridsize detection.<br>**How:** For each trained condition, launch `scripts/compare_models.py` in subprocess. Pass the trained model directory, test data file, and appropriate output directory. Ensure the evaluation uses the correct gridsize (detect from model directory or pass explicitly). Log evaluation results and capture any errors.<br>**Dependencies:** May require enhanced compare_models.py with gridsize detection |
| **Section 3: Enhanced Model Comparison Support** | | | |
| 3.A | Add gridsize detection to compare_models.py | [D] | **Why:** To make the evaluation step robust against gridsize misconfiguration.<br>**How:** Add `detect_gridsize_from_model_dir()` function to `scripts/compare_models.py`. Check for params.dill or config files in model directory to extract gridsize. Add `--gridsize` command-line argument as override. Use detected/specified gridsize for proper data loading and evaluation.<br>**File:** `scripts/compare_models.py` |
| 3.B | Enhance compare_models.py error handling | [D] | **Why:** To provide clear error messages when evaluation fails due to configuration issues.<br>**How:** Add try-catch blocks around model loading and evaluation steps. Provide specific error messages for common issues (gridsize mismatch, missing files, incompatible data shapes). Log detailed error information to help with debugging failed study runs.<br>**File:** `scripts/compare_models.py` |
| **Section 4: Integration Testing & Validation** | | | |
| 4.A | Test prepare_2x2_study.py in quick mode | [D] | **Why:** To verify the data preparation script works correctly before full study execution.<br>**How:** Run `python scripts/studies/prepare_2x2_study.py --output-dir test_prep_quick --quick-test` and verify: 1) All 4 condition directories created, 2) All train/test NPZ files present, 3) Probe files created correctly, 4) No errors in preparation log. Check file sizes are reasonable for quick test.<br>**Command:** `python scripts/studies/prepare_2x2_study.py --output-dir test_prep_quick --quick-test` |
| 4.B | Test run_2x2_study.sh in quick mode | [D] | **Why:** To verify the execution script correctly trains and evaluates all conditions.<br>**How:** Run `bash scripts/studies/run_2x2_study.sh --study-dir test_prep_quick --quick-test` and verify: 1) All 4 models train successfully, 2) All evaluations complete, 3) Logs show correct gridsize for each condition, 4) No configuration errors occur. Check for expected output files.<br>**Command:** `bash scripts/studies/run_2x2_study.sh --study-dir test_prep_quick --quick-test` |
| 4.C | Validate gridsize correctness in logs | [D] | **Why:** To confirm the critical bug is fixed - each condition uses the correct gridsize.<br>**How:** Examine training logs in each condition subdirectory. Verify: 1) `gs1_*` conditions show gridsize=1 in logs, 2) `gs2_*` conditions show gridsize=2 in logs, 3) No gridsize value is inherited from previous runs, 4) Model architecture matches expected gridsize configuration.<br>**Critical:** This is the primary success criterion for fixing the configuration bug |

---

## 🎯 **Success Criteria**

This phase is complete when:

1. **All tasks in the table above are marked [D] (Done).**
2. **The new two-script workflow runs successfully in `--quick-test` mode.**
3. **Log validation confirms correct gridsize usage:** A check of the logs confirms that each training run was initiated with the correct gridsize parameter.
4. **No configuration leakage:** Each experimental condition runs in complete isolation without inheriting configuration from previous runs.
5. **All output files generated:** Both preparation and execution scripts produce expected output files without errors.