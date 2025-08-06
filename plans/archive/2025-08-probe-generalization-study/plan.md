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