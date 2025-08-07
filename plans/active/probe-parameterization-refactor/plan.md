# R&D Plan: Probe Parameterization Study (Revised)

*Created: 2025-08-01*
*Revised: 2025-08-01*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Generalization Test with Decoupled Probe Simulation

**Problem Statement:** The current simulation workflow implicitly ties the probe function to the dataset it was generated from. There is no straightforward, reusable mechanism to simulate diffraction patterns using an arbitrary combination of an object from one source and a probe from another. This limits our ability to conduct controlled studies on how probe variations affect reconstruction.

**Proposed Solution / Hypothesis:**
- **Solution:** We will create a new, modular simulation workflow that decouples the object and probe sources. This will culminate in a comprehensive 2x2 integration study. The study will compare a synthetic, idealized probe (Probe A) against a hybrid probe (Probe B), which combines the perfect amplitude of the synthetic probe with the realistic, aberrated phase from an experimental dataset. This setup is designed to specifically test the model's robustness to phase aberrations.
- **Hypothesis:** We hypothesize that models trained with a realistic, aberrated probe phase will show different performance characteristics than those trained with an idealized probe, and that the gridsize=2 overlap constraint will make the model more robust to these probe variations.

---

## 📚 **LEARNING FROM PREVIOUS ATTEMPT**

A prior attempt to implement this study revealed critical architectural issues that led to disorganized code and incorrect results. A root cause analysis identified the following, which this revised plan directly addresses:

1. **Monolithic and Fragile Workflow:** The previous attempt used a single, large shell script that mixed data preparation, simulation, and training. This led to unpredictable side effects, particularly the gridsize configuration bug, where subsequent runs in the script would incorrectly inherit the gridsize from the first run.

   **This Plan's Solution:** We will adopt a two-stage architecture, completely separating data preparation from experiment execution. A new Python script will prepare all datasets first, and a separate shell script will then execute the training runs in isolated processes, eliminating the configuration bug.

2. **Lack of Component-Level Validation:** Without dedicated, modular tools, it was difficult to test and validate intermediate steps (like the creation of a hybrid probe or the simulation of a single dataset), leading to a final failure that was hard to debug.

   **This Plan's Solution:** The new data preparation script will be a standalone, testable utility. Each step (probe creation, simulation) will produce a verifiable artifact on disk.

3. **Flawed Train/Test Split Strategy:** The initial approach would have used a spatial split on a single simulated dataset, which is not scientifically robust for this study.

   **This Plan's Solution:** We will implement a more rigorous strategy by simulating two independent datasets (one for training, one for testing) for each of the four experimental conditions, using different random seeds for the scan positions.

---

## 🛠️ **METHODOLOGY / SOLUTION APPROACH (REVISED)**

This initiative will follow a more robust, two-stage approach that prioritizes modularity, testability, and process isolation.

**Stage 1: Centralized Data Preparation:** A new, dedicated Python script (`scripts/studies/prepare_probe_study.py`) will be created to handle all data generation tasks. It will:
- Extract the default probe and create the hybrid probe.
- For each of the four experimental conditions, it will run the simulation pipeline twice to generate independent `train_data.npz` and `test_data.npz` files.

**Stage 2: Isolated Experiment Execution:** A new, clean orchestration script (`scripts/studies/run_probe_study_corrected.sh`) will execute the study. It will:
- Assume all data has been prepared by the script from Stage 1.
- Loop through the four pre-made directories.
- Launch `ptycho_train` and `compare_models.py` for each condition in a separate, isolated process, guaranteeing that the correct configuration is loaded each time.

**Core Module Enhancements:**
- `ptycho/raw_data.py`: The efficient "sample-then-group" subsampling logic will be integrated as a permanent improvement.
- `scripts/simulation/simulate_and_save.py`: The `--probe-file` argument will be added to enable the decoupled workflow.

---

## 🎯 **DELIVERABLES**

1. **New Data Preparation Script:** A new utility script, `scripts/studies/prepare_probe_study.py`.
2. **New Study Orchestrator:** A new master script, `scripts/studies/run_probe_study_corrected.sh`.
3. **Enhanced Simulation Script:** An updated `scripts/simulation/simulate_and_save.py` with a new `--probe-file` argument.
4. **Improved Core Module:** An updated `ptycho/raw_data.py` with more efficient subsampling.
5. **Final Artifact:** A `2x2_study_report.md` file generated by the study script with quantitative metrics (PSNR, SSIM, MS-SSIM).

---

## ✅ **VALIDATION & VERIFICATION PLAN**

### Step 1: Component Testing (Incremental)
- Unit test the efficient subsampling logic in `raw_data.py`.
- Run `prepare_probe_study.py` and verify that it correctly generates all 8 datasets and 2 probe files with the expected properties.
- Test the enhanced `simulate_and_save.py` with the `--probe-file` option on a small scale.

### Step 2: End-to-End Integration Test (The 2x2 Study)

**Experimental Design:** A 2x2 study using a single synthetic 'lines' object.

|                    | Gridsize = 1          | Gridsize = 2          |
|---------------------------|-----------------------|-----------------------|
| **Probe A: Idealized** | Arm 1: Train & evaluate | Arm 2: Train & evaluate |
| **Probe B: Hybrid (Ideal Amp + Exp. Phase)** | Arm 3: Train & evaluate | Arm 4: Train & evaluate |

**Metrics:** PSNR, SSIM, MS-SSIM, FRC50

**Success Criteria:**
- **Workflow Success:** The `run_probe_study_corrected.sh` script must complete all four arms without error, and the logs must show that the correct gridsize was used for each run.
- **Viability:** All four models must train successfully and achieve a reconstruction PSNR of > 20 dB.
- **Measurable Impact:** Models trained with the Hybrid Probe are expected to show < 3 dB PSNR degradation vs the Default Probe.
- **Robustness Hypothesis:** The performance gap (degradation) between Default and Hybrid probes should be smaller for gridsize=2 than for gridsize=1.

---

## 🚀 **RISK MITIGATION**

1. **Risk:** The new data preparation script is complex and could have bugs.
   - **Mitigation:** The script will be designed to be idempotent and will produce verifiable artifacts at each step. We will validate the output datasets with `visualize_dataset.py` before proceeding to the execution stage.

2. **Risk:** The full 2x2 study is computationally expensive and time-consuming.
   - **Mitigation:** Both new scripts will include a `--quick-test` mode (fewer images, fewer epochs) to allow for rapid, end-to-end validation of the entire pipeline before launching the full study.

3. **Risk:** Unforeseen interactions with the gridsize > 1 logic.
   - **Mitigation:** The new efficient subsampling in `raw_data.py` is simpler and less prone to error than the previous "group-then-sample" logic. The isolated execution of each arm will prevent cross-contamination of configurations.

---

## 📁 **File Organization**

**Initiative Path:** `plans/active/probe-parameterization-study/`

**Next Step:** Run `/implementation` to generate the phased implementation plan for this revised R&D plan.