# Agent Implementation Checklist: Final Phase - Validation & Documentation

**Overall Goal for this Phase:** To validate the complete convert -> split -> shuffle -> run_study workflow with a real dataset and update all relevant documentation.

## Instructions for Agent

Copy this checklist into your working memory.
Update the State for each item as you progress: [ ] (Open) -> [P] (In Progress) -> [D] (Done).
Follow the How/Why & API Guidance column carefully for implementation details.

## Implementation Checklist

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| **Section 0: Preparation & Context Priming** | | | |
| 0.A | Review Key Documents & APIs | [ ] | **Why:** To load the necessary context and technical specifications before coding. <br> **Docs:** docs/sampling/plan_sampling_study.md, scripts/tools/README.md, scripts/studies/QUICK_REFERENCE.md. <br> **APIs:** scripts/tools/transpose_rename_convert_tool.py, scripts/tools/split_dataset_tool.py, scripts/tools/shuffle_dataset_tool.py, scripts/studies/run_complete_generalization_study.sh. |
| 0.B | Identify Target Files for Modification | [ ] | **Why:** To have a clear list of files that will be touched during this phase. <br> **Files:** scripts/tools/README.md (Modify), scripts/studies/QUICK_REFERENCE.md (Modify), docs/PROJECT_STATUS.md (Modify). |
| **Section 1: End-to-End Workflow Validation** | | | |
| 1.A | Step 1: Convert Raw fly64 Data | [ ] | **Why:** To ensure the dataset conforms to the project's canonical format. <br> **How:** Run `python scripts/tools/transpose_rename_convert_tool.py datasets/fly64/fly001_64_train.npz datasets/fly64/fly001_64_converted.npz`. <br> **File:** datasets/fly64/fly001_64_converted.npz (Create). |
| 1.B | Step 2: Split Converted Data | [ ] | **Why:** To isolate the top half of the scan area for the study. <br> **How:** Run `python scripts/tools/split_dataset_tool.py datasets/fly64/fly001_64_converted.npz datasets/fly64/ --split-fraction 0.5 --split-axis y`. <br> **File:** datasets/fly64/fly001_64_converted_train.npz (Create). |
| 1.C | Step 3: Shuffle the Top-Half Data | [ ] | **Why:** To randomize the training samples within the spatial subset. <br> **How:** Run `python scripts/tools/shuffle_dataset_tool.py datasets/fly64/fly001_64_converted_train.npz datasets/fly64/fly001_64_top_half_shuffled.npz`. <br> **File:** datasets/fly64/fly001_64_top_half_shuffled.npz (Create). |
| 1.D | Step 4: Execute the Generalization Study | [ ] | **Why:** To run the full experiment using the prepared data. <br> **How:** Run `./scripts/studies/run_complete_generalization_study.sh --train-data "datasets/fly64/fly001_64_top_half_shuffled.npz" --test-data "datasets/fly64/fly001_64_converted.npz" --output-dir "fly64_top_half_study" --train-sizes "256 512" --num-trials 2 --skip-data-prep`. |
| 1.E | Verify Study Output | [ ] | **Why:** To confirm the entire workflow completed successfully. <br> **How:** Check that the fly64_top_half_study directory exists and contains subdirectories for each training size, comparison plots, and a results.csv file. |
| **Section 2: Documentation Updates** | | | |
| 2.A | Update tools/README.md | [ ] | **Why:** To document the new shuffle_dataset_tool.py for other developers. <br> **How:** Add a new section to the README explaining the purpose of the shuffling tool, its command-line arguments, and a usage example. |
| 2.B | Update studies/QUICK_REFERENCE.md | [ ] | **Why:** To make the new workflow easily discoverable and reusable. <br> **How:** Add a new entry under "Common Workflows" or create a new section for "Spatially-Biased Studies" that shows the convert -> split -> shuffle -> run_study command chain. |
| **Section 3: Finalization** | | | |
| 3.A | Code Formatting & Linting | [ ] | **Why:** To maintain code quality and project standards. <br> **How:** Run the project's standard formatters and linters on scripts/tools/shuffle_dataset_tool.py. |
| 3.B | Update PROJECT_STATUS.md | [ ] | **Why:** To mark the initiative as complete and ready for the next one. <br> **How:** Move the "Spatially-Biased Randomized Sampling Study" initiative from "Current Active Initiative" to the top of the "Completed Initiatives" section. |
| 3.C | Clean Up Intermediate Files | [ ] | **Why:** To keep the repository clean. <br> **How:** Remove the intermediate .npz files created during the workflow (fly001_64_converted.npz, fly001_64_converted_train.npz, fly001_64_top_half_shuffled.npz). The final study output directory is the main artifact. |

## Notes

- This checklist represents the final validation phase of the Spatially-Biased Randomized Sampling Study initiative
- Each task builds upon the previous ones in the workflow
- The end goal is to demonstrate the complete convert->split->shuffle->run_study pipeline with real data
- All documentation updates ensure the new capabilities are discoverable by future developers