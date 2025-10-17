### **Agent Implementation Checklist: Phase 2: Create Multi-Run Orchestration**

**Overall Goal for this Phase:** To build a master script that automates running comparisons across multiple training set sizes.

**Instructions for Agent:**
1.  Copy this checklist into your working memory.
2.  Update the `State` for each item as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).
3.  Follow the `How/Why & API Guidance` column carefully for implementation details.

---

| ID  | Task Description                                   | State | How/Why & API Guidance
| **Section 0: Preparation & Context Priming**
| 0.A | **Review Key Documents & APIs**                    | `[ ]` | **Why:** To load the necessary context and technical specifications before coding. <br> **Docs:** `docs/studies/plan_model_generalization.md`, `scripts/run_comparison.sh` (modified in Phase 1). <br> **APIs:** `bash getopts`, `mkdir -p`, `dirname`, bash array handling.
| 0.B | **Identify Target Files for Modification/Creation**| `[ ]` | **Why:** To have a clear list of files that will be touched during this phase. <br> **Files:** `scripts/studies/run_generalization_study.sh (Create)`, potentially `scripts/studies/` directory structure.
| **Section 1: Directory Structure & Script Creation**
| 1.A | **Create scripts/studies/ directory if needed**    | `[ ]` | **Why:** To organize study-related scripts in a dedicated location. <br> **How:** Use `mkdir -p scripts/studies` to ensure directory exists. <br> **File:** Directory creation.
| 1.B | **Create run_generalization_study.sh skeleton**    | `[ ]` | **Why:** To establish the main orchestration script structure. <br> **How:** Create executable shell script with proper shebang, usage function, and argument parsing using `getopts`. Include options for `--train-sizes`, `--test-data`, `--train-data`, `--output-dir`. <br> **File:** `scripts/studies/run_generalization_study.sh`.
| **Section 2: Argument Parsing & Configuration**
| 2.A | **Implement command-line argument parsing**        | `[ ]` | **Why:** To accept user configuration for training sizes and data paths. <br> **How:** Use `getopts` to parse named arguments. Support `--train-sizes` (space-separated list), `--test-data`, `--train-data`, `--output-dir`. Convert `--train-sizes` string into bash array. <br> **File:** `scripts/studies/run_generalization_study.sh`.
| 2.B | **Add argument validation and defaults**           | `[ ]` | **Why:** To ensure script fails gracefully with helpful error messages. <br> **How:** Validate required arguments are provided, check file existence for data paths, set reasonable defaults for optional arguments. <br> **File:** `scripts/studies/run_generalization_study.sh`.
| **Section 3: Multi-Run Logic Implementation**
| 3.A | **Implement training size iteration loop**         | `[ ]` | **Why:** To execute comparison runs for each specified training size. <br> **How:** Loop through the training sizes array, create subdirectory for each run (e.g., `train_512`, `train_1024`), call `run_comparison.sh` with appropriate arguments. <br> **File:** `scripts/studies/run_generalization_study.sh`.
| 3.B | **Add output directory organization**              | `[ ]` | **Why:** To keep results from different training sizes separate and organized. <br> **How:** For each training size N, create subdirectory `${output_dir}/train_${N}/` and pass this as `--output-dir` to `run_comparison.sh`. <br> **File:** `scripts/studies/run_generalization_study.sh`.
| 3.C | **Implement progress reporting**                   | `[ ]` | **Why:** To provide user feedback during long-running multi-run experiments. <br> **How:** Add echo statements before each run showing current progress (e.g., "Running comparison 2/4: train_size=1024"). Include timestamps. <br> **File:** `scripts/studies/run_generalization_study.sh`.
| **Section 4: Error Handling & Robustness**
| 4.A | **Add error handling for failed runs**            | `[ ]` | **Why:** To handle cases where individual comparison runs fail without stopping the entire study. <br> **How:** Check exit code of `run_comparison.sh` calls, log failures, continue with remaining runs. Optionally use `set -e` with trap handling. <br> **File:** `scripts/studies/run_generalization_study.sh`.
| 4.B | **Add cleanup and signal handling**               | `[ ]` | **Why:** To gracefully handle script interruption and clean up partial results. <br> **How:** Implement trap for SIGINT/SIGTERM to cleanup incomplete runs and log study interruption. <br> **File:** `scripts/studies/run_generalization_study.sh`.
| **Section 5: Testing & Validation**
| 5.A | **Test with minimal training sizes**              | `[ ]` | **Why:** To verify the script works correctly before running full experiments. <br> **How:** Run with `--train-sizes "512 1024"` using existing test data to verify directory creation, argument passing, and output organization. <br> **File:** Test execution.
| 5.B | **Verify output directory structure**             | `[ ]` | **Why:** To confirm that results are properly organized for downstream processing. <br> **How:** Check that each training size creates its own subdirectory with complete `comparison_metrics.csv` and other expected outputs. <br> **File:** Test verification.
| **Section 6: Documentation & Finalization**
| 6.A | **Add comprehensive usage documentation**          | `[ ]` | **Why:** To provide clear instructions for users running the study. <br> **How:** Add detailed usage function with examples, parameter descriptions, and typical workflow instructions. Include comments in the script. <br> **File:** `scripts/studies/run_generalization_study.sh`.
| 6.B | **Make script executable and test final version** | `[ ]` | **Why:** To ensure the script is ready for use. <br> **How:** Run `chmod +x scripts/studies/run_generalization_study.sh` and perform final end-to-end test with small training sizes. <br> **File:** `scripts/studies/run_generalization_study.sh`.
