# Phase 3: 2x2 Study Orchestration and Execution Checklist

**Initiative:** Probe Parameterization Study
**Created:** 2025-08-01
**Phase Goal:** To automate and execute the full 2x2 probe generalization study, which serves as the final, comprehensive integration test of all new components.
**Deliverable:** A new master script `scripts/studies/run_2x2_probe_study.sh` and the completed training and evaluation outputs for all four experimental arms.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :--------------------- |
| **Section 0: Prerequisites Verification** |
| 0.A | **Verify Phase 1 & 2 completion**                  | `[ ]` | **Why:** Ensure all dependencies are ready for the study. <br> **How:** Check: 1) `scripts/tools/create_hybrid_probe.py` exists and works, 2) `scripts/simulation/simulate_and_save.py` accepts --probe-file, 3) Run `python -m pytest tests/test_decoupled_simulation.py` passes. <br> **Verify:** All components from previous phases are functional. |
| 0.B | **Review existing study scripts**                  | `[ ]` | **Why:** Learn from established patterns in the project. <br> **How:** Examine `scripts/studies/run_complete_generalization_study.sh` and `scripts/run_comparison.sh` to understand: checkpointing patterns, argument parsing, directory structure, error handling. Note reusable patterns. <br> **Files:** Study scripts in `scripts/studies/` |
| **Section 1: Script Structure and Setup** |
| 1.A | **Create script with header and usage**            | `[ ]` | **Why:** Establish the foundation with proper documentation. <br> **How:** Create `scripts/studies/run_2x2_probe_study.sh`. Add: shebang `#!/bin/bash`, script description, usage function showing all options (--output-dir, --quick-test, --parallel-jobs, --dataset). Set `set -euo pipefail` for error handling. <br> **File:** `scripts/studies/run_2x2_probe_study.sh` |
| 1.B | **Implement argument parsing**                     | `[ ]` | **Why:** Allow flexible configuration of the study. <br> **How:** Parse arguments: `--output-dir` (required), `--quick-test` (flag), `--parallel-jobs N` (default 1), `--dataset` (default datasets/fly/fly64_transposed.npz), `--skip-completed` (flag). Use while/case loop for parsing. Validate required args. |
| 1.C | **Set up study parameters**                        | `[ ]` | **Why:** Define the experimental matrix clearly. <br> **How:** Define arrays: `GRIDSIZES=(1 2)`, `PROBE_TYPES=("default" "hybrid")`. Set quick-test overrides: `N_TRAIN=512; N_TEST=128; EPOCHS=5` vs full: `N_TRAIN=5000; N_TEST=1000; EPOCHS=50`. Create output directory structure plan. |
| 1.D | **Implement checkpoint detection**                 | `[ ]` | **Why:** Allow resuming interrupted studies efficiently. <br> **How:** Create function `is_step_complete()` that checks for marker files: `.simulation_done`, `.training_done`, `.evaluation_done`. Skip completed steps when `--skip-completed` is set. Log skipped steps. |
| **Section 2: Probe Generation Logic** |
| 2.A | **Create default probe extraction**                | `[ ]` | **Why:** Need reference probe for comparison. <br> **How:** Extract default probe from dataset: `np_cmd="import numpy as np; data=np.load('$DATASET'); np.save('$OUTPUT_DIR/default_probe.npy', data['probeGuess'])"`. Execute with `python -c "$np_cmd"`. Verify file created. |
| 2.B | **Generate hybrid probe**                          | `[ ]` | **Why:** Create the experimental probe variant. <br> **How:** Determine amplitude and phase sources (for initial version, use same dataset with different keys or idealized phase). Run: `python scripts/tools/create_hybrid_probe.py "$DATASET" "$DATASET" --output "$OUTPUT_DIR/hybrid_probe.npy"`. Add error checking. |
| 2.C | **Validate both probes**                           | `[ ]` | **Why:** Ensure probes are valid before expensive training. <br> **How:** Create validation function that loads each probe and checks: finite values, complex dtype, reasonable size. Log probe statistics (shape, mean amplitude, phase variance). Fail early if invalid. |
| **Section 3: Simulation Pipeline** |
| 3.A | **Implement simulation function**                  | `[ ]` | **Why:** Generate training data for each configuration. <br> **How:** Create `run_simulation()` function with params: gridsize, probe_type, output_subdir. Build command: `python scripts/simulation/simulate_and_save.py --input-file "$DATASET" --probe-file "$probe_path" --output-file "$output_subdir/simulated_data.npz" --n-images "$N_TRAIN" --gridsize "$gridsize"`. Add logging and error handling. |
| 3.B | **Add simulation checkpointing**                   | `[ ]` | **Why:** Track completion of expensive simulation steps. <br> **How:** After successful simulation, create marker: `touch "$output_subdir/.simulation_done"`. Check marker existence before running simulation. Log whether running or skipping. |
| **Section 4: Training Pipeline** |
| 4.A | **Implement training function**                    | `[ ]` | **Why:** Train models for each experimental condition. <br> **How:** Create `run_training()` function. Build command: `ptycho_train --train-data "$output_subdir/simulated_data.npz" --output-dir "$output_subdir/model" --epochs "$EPOCHS" --batch-size 32`. Handle both PINN and baseline model types. |
| 4.B | **Add training progress tracking**                 | `[ ]` | **Why:** Monitor long-running training jobs. <br> **How:** Redirect training output to log file: `> "$output_subdir/training.log" 2>&1`. Add timestamp logging. For interactive mode, use `tee` to show output while logging. Create `.training_done` marker on success. |
| **Section 5: Evaluation Pipeline** |
| 5.A | **Implement comparison function**                  | `[ ]` | **Why:** Evaluate trained models consistently. <br> **How:** Create `run_evaluation()` function. Use: `python scripts/compare_models.py --pinn-dir "$output_subdir/model" --baseline-dir "$output_subdir/model" --test-data "$test_data_path" --output-dir "$output_subdir/evaluation"`. Note: comparing same model for now, adjust for actual comparison needs. |
| 5.B | **Extract metrics for summary**                    | `[ ]` | **Why:** Prepare data for final aggregation. <br> **How:** After evaluation, copy key metrics: `cp "$output_subdir/evaluation/comparison_metrics.csv" "$output_subdir/metrics_summary.csv"`. Add experiment metadata (gridsize, probe_type) to filename or content for later aggregation. |
| **Section 6: Orchestration Logic** |
| 6.A | **Implement main execution loop**                  | `[ ]` | **Why:** Coordinate all experimental arms. <br> **How:** Create nested loops: `for gridsize in "${GRIDSIZES[@]}"; do for probe_type in "${PROBE_TYPES[@]}"; do` ... Create arm name like `gs${gridsize}_${probe_type}`, create output subdirectory, call pipeline functions in sequence. |
| 6.B | **Add parallel execution support**                 | `[ ]` | **Why:** Utilize multiple GPUs if available. <br> **How:** If `--parallel-jobs > 1`, use GNU parallel or background jobs with job control. Track PIDs, wait for completion. Implement job slot management to limit concurrent jobs. Default to sequential execution. |
| 6.C | **Implement error handling and cleanup**           | `[ ]` | **Why:** Ensure robustness and clean failure modes. <br> **How:** Add trap for cleanup on exit. If any arm fails, log error but continue with others (unless --fail-fast set). Summarize failures at end. Save script state for debugging. |
| **Section 7: Quick Test Mode** |
| 7.A | **Add quick test parameter overrides**             | `[ ]` | **Why:** Enable rapid validation of the pipeline. <br> **How:** When --quick-test set, override: N_TRAIN=256, N_TEST=128, EPOCHS=2. Add "[QUICK TEST]" prefix to output directory. Log that quick test mode is active. Reduce dataset if needed. |
| 7.B | **Create minimal test execution**                  | `[ ]` | **Why:** Verify orchestration without full computation. <br> **How:** In quick test, optionally run only one arm (gs1_default) to verify pipeline. Add --quick-full flag to run all four arms with reduced parameters. |
| **Section 8: Documentation and Finalization** |
| 8.A | **Update scripts/studies/CLAUDE.md**               | `[ ]` | **Why:** Document the new study script for developers. <br> **How:** Add section describing run_2x2_probe_study.sh: purpose (probe parameterization study), usage examples, output structure, interpretation guide. Follow existing documentation patterns in file. |
| 8.B | **Add execution examples to script**               | `[ ]` | **Why:** Help users understand common usage patterns. <br> **How:** In script header comments, add examples: basic usage, quick test mode, parallel execution, resuming interrupted run. Include expected runtime estimates and resource requirements. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: The `run_2x2_probe_study.sh` script completes a full run without errors
   - Quick test execution: `./scripts/studies/run_2x2_probe_study.sh --output-dir test_2x2_study --quick-test`
   - Verify output directory contains four subdirectories (gs1_default, gs1_hybrid, gs2_default, gs2_hybrid)
   - Each subdirectory has: simulated_data.npz, model/, evaluation/, and metrics_summary.csv
3. The script handles interruption gracefully (can be resumed with --skip-completed)
4. Documentation is complete and examples work as shown
5. Full study execution is ready to run (may be deferred to available compute time)