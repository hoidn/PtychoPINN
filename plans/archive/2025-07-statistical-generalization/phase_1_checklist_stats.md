# Agent Implementation Checklist: Phase 1 - Multi-Trial Execution Framework

**Overall Goal for this Phase:** To update the `run_complete_generalization_study.sh` script to support configurable, sequential multi-trial runs with a nested output directory structure.

**Instructions for Agent:**
1. Copy this checklist into your working memory.
2. Update the `State` for each item as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).
3. Follow the `How/Why & API Guidance` column carefully for implementation details.

---

| ID | Task Description | State | How/Why & API Guidance |
| :-- | :-- | :-- | :-- |
| **Section 0: Preparation & Context Priming** |
| 0.A | **Review Key Documents & APIs** | `[ ]` | **Why:** To load the necessary context and technical specifications before coding. <br> **Docs:** `docs/studies/multirun/plan_statistical_generalization.md`, `scripts/studies/run_complete_generalization_study.sh`. <br> **APIs:** Bash `getopts` or manual loop for argument parsing, `seq` for loops. |
| 0.B | **Identify Target Files for Modification** | `[ ]` | **Why:** To have a clear list of files that will be touched during this phase. <br> **Files:** `scripts/studies/run_complete_generalization_study.sh` (Modify). |
| **Section 1: Argument Parsing and Validation** |
| 1.A | **Add `--num-trials` argument parsing** | `[ ]` | **Why:** To allow users to configure the number of trials. <br> **How:** In `run_complete_generalization_study.sh`, add a `DEFAULT_NUM_TRIALS=5` variable. In the argument parsing loop, add a case for `--num-trials` that updates this variable. <br> **File:** `scripts/studies/run_complete_generalization_study.sh`. |
| 1.B | **Update script's help function** | `[ ]` | **Why:** To make the new feature discoverable. <br> **How:** Add the `--num-trials` flag and its description to the `show_help()` function in the script. <br> **File:** `scripts/studies/run_complete_generalization_study.sh`. |
| 1.C | **Add validation for `--num-trials`** | `[ ]` | **Why:** To prevent errors from invalid input. <br> **How:** After the argument parsing loop, add a check to ensure `$NUM_TRIALS` is a positive integer. If not, print an error and exit. `if ! [[ "$NUM_TRIALS" =~ ^[1-9][0-9]*$ ]]; then ... fi` <br> **File:** `scripts/studies/run_complete_generalization_study.sh`. |
| **Section 2: Implement Sequential Multi-Trial Loop** |
| 2.A | **Create nested trial loop** | `[ ]` | **Why:** This is the core logic for running multiple trials. <br> **How:** Inside the main `for train_size in $TRAIN_SIZES; do` loop, add a new inner loop: `for trial in $(seq 1 "$NUM_TRIALS"); do ... done`. Move the existing training and comparison logic inside this new inner loop. <br> **File:** `scripts/studies/run_complete_generalization_study.sh`. |
| 2.B | **Modify output directory path** | `[ ]` | **Why:** To ensure each trial's output is saved to a unique, nested directory. <br> **How:** Inside the new trial loop, construct the output path for the training/comparison logic to be `"$OUTPUT_DIR/train_$train_size/trial_$trial"`. <br> **File:** `scripts/studies/run_complete_generalization_study.sh`. |
| 2.C | **Update logging and console output** | `[ ]` | **Why:** To provide clear feedback on which trial is currently running. <br> **How:** Modify `echo` and `log` statements within the loop to include the trial number, e.g., `log "Training models for train_size=$train_size (Trial $trial/$NUM_TRIALS)"`. <br> **File:** `scripts/studies/run_complete_generalization_study.sh`. |
| 2.D | **Remove Parallel Execution Logic** | `[ ]` | **Why:** To simplify the script and align with the sequential nature of GPU training. <br> **How:** Remove the `train_single_size` function, the `pids` array, and all logic related to backgrounding processes (`&`) and `wait`. The script should now run all steps sequentially. <br> **File:** `scripts/studies/run_complete_generalization_study.sh`. |
| **Section 3: Verification and Finalization** |
| 3.A | **Update `--dry-run` output** | `[ ]` | **Why:** To ensure the dry run accurately reflects the new sequential execution plan. <br> **How:** Check that the `run_cmd` function, when in dry-run mode, prints the full nested path including the `trial_N` subdirectory for each sequential step. <br> **File:** `scripts/studies/run_complete_generalization_study.sh`. |
| 3.B | **Update final summary message** | `[ ]` | **Why:** To accurately report the total number of training runs performed. <br> **How:** In the `generate_summary` function, calculate the total runs as `total_runs=$((${#TRAIN_SIZES_ARRAY[@]} * NUM_TRIALS))` and update the summary text. <br> **File:** `scripts/studies/run_complete_generalization_study.sh`. |
| **Section 4: Testing** |
| 4.A | **Perform dry run test** | `[ ]` | **Why:** To quickly verify the new loop and path logic without running a full training cycle. <br> **How:** Run `./scripts/studies/run_complete_generalization_study.sh --train-sizes "512 1024" --num-trials 2 --dry-run`. Check the console output for the correct nested paths (`.../trial_1`, `.../trial_2`). |
| 4.B | **Perform integration test ("Micro-Study")** | `[ ]` | **Why:** To confirm the end-to-end workflow functions correctly. <br> **How:** Run `./scripts/studies/run_complete_generalization_study.sh --train-sizes "512" --num-trials 2 --skip-data-prep`. After completion, verify that the directory structure `output_dir/train_512/trial_1/` and `.../trial_2/` exists and contains results. |