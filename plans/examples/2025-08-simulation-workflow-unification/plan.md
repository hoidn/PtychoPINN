# R&D Plan: Simulation Workflow Unification

*Created: 2025-08-02*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** Simulation Workflow Unification

**Problem Statement:** The current simulation pipeline, specifically the workflow invoked by `scripts/simulation/simulate_and_save.py`, contains a critical architectural flaw. It relies on a legacy, monolithic function (`RawData.from_simulation`) that improperly mixes data preparation and physics simulation. This leads to a tensor shape mismatch bug when using `gridsize > 1`, causing the simulation to crash. Furthermore, this legacy path is inconsistent with the modern, more robust data generation logic used by the main training pipeline, creating architectural debt and maintenance challenges.

**Proposed Solution / Hypothesis:**
- **Solution:** We will refactor the `simulate_and_save.py` workflow to abandon the monolithic `RawData.from_simulation` method. Instead, it will be re-implemented to explicitly orchestrate the distinct steps of coordinate grouping, ground truth patch extraction, and diffraction simulation using the modern, modular helper functions that are already proven in the main training pipeline. This will align the simulation workflow with the project's best practices of explicit, decoupled logic.
- **Hypothesis:** By refactoring the simulation pipeline to use a modular, step-by-step approach, we will not only fix the `gridsize > 1` bug but also improve the pipeline's correctness, maintainability, and consistency with the rest of the codebase. This architectural alignment will prevent future regressions and make the simulation tools more robust and easier to debug.

---

## 🛠️ **METHODOLOGY / SOLUTION APPROACH**

The core of this initiative is to decompose and re-orchestrate the simulation logic. Instead of relying on a single black-box function, the `simulate_and_save.py` script will be modified to manage the data flow explicitly.

### The Refactored Workflow:

1. **Load Inputs:** Load `objectGuess` and `probeGuess` from specified files.

2. **Generate & Group Coordinates:** Use the modular `ptycho.raw_data.group_coords()` function to generate scan positions and group them according to the `gridsize`.

3. **Extract Ground Truth Patches (Y):** Use the modular `ptycho.raw_data.get_image_patches()` to extract object patches in the correct multi-channel "Channel Format".

4. **Format Conversion & Simulation (X):**
   - Explicitly convert the Y patches from "Channel Format" to "Flat Format" using `ptycho.tf_helper._channel_to_flat()`.
   - Call the core physics engine `ptycho.diffsim.illuminate_and_diffract()` with the correctly formatted flat tensor.
   - Convert the resulting flat diffraction tensor back to "Channel Format" using `ptycho.tf_helper._flat_to_channel()`.

5. **Assemble & Save:** Combine all generated arrays into a final `.npz` file that adheres to the project's data contracts.

This approach isolates the change to the high-level orchestration script, leaving the core, stable components (`illuminate_and_diffract`, `group_coords`, etc.) unmodified.

---

## 🎯 **DELIVERABLES**

1. **Refactored Simulation Script:** An updated `scripts/simulation/simulate_and_save.py` that implements the new, modular workflow.

2. **New Integration Test Suite:** A new test file, `tests/simulation/test_simulate_and_save.py`, that provides comprehensive validation for the refactored pipeline.

3. **Updated Documentation:** Revisions to `scripts/simulation/CLAUDE.md` and `README.md` to reflect the unified and corrected architecture.

4. **Deprecation of Legacy Method:** The `RawData.from_simulation` method will be marked with a `DeprecationWarning` to guide future development.

---

## ✅ **VALIDATION & VERIFICATION PLAN**

This initiative's success depends on rigorous validation to ensure correctness and prevent regressions.

### Unit / Integration Tests:

A new integration test suite (`tests/simulation/test_simulate_and_save.py`) will be created to verify the end-to-end behavior of the refactored `simulate_and_save.py` script. The tests will:

1. **Verify gridsize=1 Regression:** Run the script with `gridsize=1` and assert that the output `.npz` file contains tensors with the correct single-channel shapes (e.g., `(B, N, N)`).

2. **Verify gridsize=2 Correctness:** Run the script with `gridsize=2` and assert that the output `.npz` file contains tensors with the correct multi-channel shapes (e.g., `(B, N, N, 4)`).

3. **Verify Probe Override:** Run the script with the `--probe-file` argument and assert that the `probeGuess` in the output file matches the external probe.

4. **Content Sanity Check:** Perform basic checks on the output data to ensure it is physically plausible (e.g., non-zero, correct data types).

5. **Data Contract Compliance:** Verify that all output files strictly adhere to the specifications in `docs/data_contracts.md`:
   - `diffraction` is `float32` amplitude (not intensity)
   - `Y` patches (if generated) are `complex64` and 3D
   - All required keys are present with correct shapes

### Success Criteria:

- The `ValueError` crash when running `simulate_and_save.py` with `gridsize > 1` is resolved.
- The refactored script produces valid, training-ready datasets that conform to the project's data contracts for both `gridsize=1` and `gridsize > 1`.
- All new integration tests pass, confirming both the fix and the absence of regressions in the `gridsize=1` case.
- The simulation pipeline's logic is now explicit, modular, and consistent with the main training pipeline's data handling.
- Performance benchmarks show no significant regression compared to the legacy implementation.

---

## 🚀 **RISK MITIGATION**

**Risk:** The refactoring introduces a subtle bug that leads to silent data corruption (e.g., mismatch between coordinates and patches).
- **Mitigation:** The new integration test suite will include content validation to catch such issues. Visual inspection of the output using `visualize_dataset.py` will be a required manual step during development.

**Risk:** The change breaks an unknown, downstream dependency on the old `RawData.from_simulation` method.
- **Mitigation:** The method will be deprecated with a warning first, not immediately removed. A codebase-wide search for its usage will be performed. The focus of the change is on the `simulate_and_save.py` script, which is the primary known user.

**Risk:** The refactoring is more complex than anticipated and takes longer than planned.
- **Mitigation:** The phased implementation plan will break the work into manageable chunks. The core components (`group_coords`, `get_image_patches`, `illuminate_and_diffract`) are already implemented and tested, reducing the scope to orchestration logic.

**Risk:** Performance regression due to explicit format conversions.
- **Mitigation:** Performance benchmarks will be included in the test suite to ensure the refactored pipeline maintains acceptable performance levels.

---

## 📁 **File Organization**

**Initiative Path:** `plans/examples/2025-08-simulation-workflow-unification/`

**Next Step:** Run `/implementation` to generate the phased implementation plan.