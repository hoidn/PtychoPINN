# Phase 3: 2x2 Study Orchestration and Execution Checklist (Corrected)

**Initiative:** Probe Parameterization Study  
**Created:** 2025-08-01  
**Phase Goal:** To implement a robust, two-stage workflow for executing the full 2x2 probe generalization study. This phase creates the scripts to prepare all data upfront and then execute the training and evaluation in isolated processes, permanently fixing the gridsize configuration bug.  
**Deliverable:** Two new master scripts, `scripts/studies/prepare_2x2_study.py` and `scripts/studies/run_2x2_study.sh`, and the completed training and evaluation outputs for all four experimental arms.

| ID | Task Description | State | How/Why & API Guidance |
|----|-----------------|-------|------------------------|
| **Section 1: Data Preparation Script (prepare_2x2_study.py)** | | | |
| 1.A | Create prepare_2x2_study.py script | [ ] | **Why:** To centralize all data generation into a single, verifiable script, separating it from execution. <br> **How:** Create `scripts/studies/prepare_2x2_study.py`. It will take `--output-dir`, `--dataset` (source for experimental phase), and `--quick-test` as arguments. It will use the tools from Phase 1. <br> **File:** `scripts/studies/prepare_2x2_study.py` |
| 1.B | Implement Probe Generation Logic | [ ] | **Why:** To create the two probe variants (idealized and hybrid) for the study. <br> **How:** Inside the script, generate the idealized probe using `ptycho.probe.get_default_probe()`. Then, call `scripts/tools/create_hybrid_probe.py` as a subprocess to create the hybrid probe (idealized amplitude + experimental phase). Save both to the output directory. |
| 1.C | Implement 4-Condition Simulation Loop | [ ] | **Why:** To generate all 8 required datasets (4 train, 4 test) with scientific rigor. <br> **How:** Loop through gridsizes (1, 2) and probe types (idealized, hybrid). For each of the 4 conditions, call `scripts/simulation/simulate_and_save.py` twice: once with `--seed 42` for `train_data.npz` and once with `--seed 43` for `test_data.npz`. This creates independent train/test sets. |
| **Section 2: Experiment Execution Script (run_2x2_study.sh)** | | | |
| 2.A | Create run_2x2_study.sh script | [ ] | **Why:** To execute experiments in isolated processes, fixing the gridsize configuration bug. <br> **How:** Create `scripts/studies/run_2x2_study.sh`. It will take `--study-dir` as input and automatically detect the four condition subdirectories prepared in the previous step. Use `set -euo pipefail` for robustness. <br> **File:** `scripts/studies/run_2x2_study.sh` |
| 2.B | Implement Isolated Training Loop | [ ] | **Why:** This is the core fix for the configuration bug. <br> **How:** The script will loop through the four condition directories. For each, it will extract the gridsize from the directory name (e.g., "gs1" -> 1). It will then launch `ptycho_train` in a new subprocess, passing the correct `--gridsize` and data paths. |
| 2.C | Implement Isolated Evaluation Loop | [ ] | **Why:** To evaluate each model against its corresponding, independently simulated test set. <br> **How:** After each training run completes, the script will launch `scripts/compare_models.py` in another new subprocess, passing the paths to the newly trained model and the correct `test_data.npz` for that condition. |
| **Section 3: Final Validation and Analysis** | | | |
| 3.A | Run Full Workflow in Quick Mode | [ ] | **Why:** To perform an end-to-end validation of the entire corrected pipeline. <br> **How:** First, run `python scripts/studies/prepare_2x2_study.py --output-dir test_study --quick-test`. Then, run `bash scripts/studies/run_2x2_study.sh --study-dir test_study --quick-test`. <br> **Verification:** Check logs to confirm each run used the correct gridsize. |
| 3.B | Aggregate and Plot Results | [ ] | **Why:** To generate the final scientific output of the study. <br> **How:** Run `scripts/studies/aggregate_and_plot_results.py` on the completed study directory to generate the summary CSV and comparison plots. |
| 3.C | Cleanup and Documentation | [ ] | **Why:** To finalize the initiative and remove obsolete code. <br> **How:** Delete the old, flawed study scripts and their tests. Update `docs/PROJECT_STATUS.md` and other relevant documentation to reflect the new, successful workflow. |

## 🎯 Success Criteria

This phase is complete when:

- All tasks in the table above are marked **[D]** (Done).
- The new two-script workflow (`prepare_2x2_study.py` and `run_2x2_study.sh`) runs successfully in `--quick-test` mode.
- A check of the training logs confirms that each experimental arm was executed with the correct gridsize parameter, proving the configuration bug is fixed.
- The final `aggregate_and_plot_results.py` script successfully generates a summary report from the study's output.
- All obsolete scripts from the previous attempt have been removed.