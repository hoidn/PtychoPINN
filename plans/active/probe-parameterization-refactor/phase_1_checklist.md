# Phase 1: Refactor for Reusability and Modularity Checklist

**Initiative:** Probe Parameterization Study (Corrected)  
**Created:** 2025-08-01  

**Phase Goal:** To extract the core logic from the successful experimental scripts into general-purpose, reusable tools and modules, following the modular design of the original plan.

**Deliverable:** A new helper module `ptycho/workflows/simulation_utils.py`, a new standalone tool `scripts/tools/create_hybrid_probe.py`, a revised `simulate_and_save.py`, and a comprehensive suite of unit tests for all new components.

## ✅ **Task List**

**Instructions:**
- Work through tasks in order. Dependencies are noted in the guidance column.
- The "How/Why & API Guidance" column contains all necessary details for implementation.
- Update the State column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| **Section 1: Core Helper Module & Subsampling Improvement** | | | |
| 1.A | Create simulation_utils.py module | [D] | **Why:** To centralize reusable probe loading and validation logic.<br>**How:** Create `ptycho/workflows/simulation_utils.py`. Implement `load_probe_from_source` (handles .npy, .npz, and direct array inputs) and `validate_probe_object_compatibility` (ensures probe is smaller than object). Add comprehensive docstrings and error handling.<br>**File:** `ptycho/workflows/simulation_utils.py` |
| 1.B | Add Unit Tests for simulation_utils.py | [D] | **Why:** To ensure the new helper functions are robust and reliable.<br>**How:** Create `tests/workflows/test_simulation_utils.py`. Add test cases for all supported input types, invalid inputs (wrong shape, wrong dtype, missing keys), and the probe/object size validation.<br>**File:** `tests/workflows/test_simulation_utils.py` |
| 1.C | Revise ptycho/raw_data.py Subsampling | [D] | **Why:** To replace the inefficient "group-then-sample" logic with the superior "sample-then-group" strategy for gridsize > 1.<br>**How:** Modify the `generate_grouped_data` method. For gridsize > 1, implement the logic to first randomly sample nsamples starting points, and then find the K-nearest neighbors for only those sampled points. This is a critical performance optimization.<br>**File:** `ptycho/raw_data.py` |
| **Section 2: Standalone Tools Implementation** | | | |
| 2.A | Enhance simulate_and_save.py | [D] | **Why:** To decouple the probe from the object source, a core goal of the study.<br>**How:** Add a `--probe-file` command-line argument. Integrate the `load_probe_from_source` and `validate_probe_object_compatibility` helpers from `simulation_utils.py` to handle the external probe. Ensure backward compatibility (script must still work without the new flag).<br>**File:** `scripts/simulation/simulate_and_save.py` |
| 2.B | Create create_hybrid_probe.py Tool | [D] | **Why:** To create a modular, reusable tool for generating probes with mixed characteristics.<br>**How:** Create `scripts/tools/create_hybrid_probe.py`. It should take two source files, extract amplitude from the first and phase from the second, and combine them into a new probe. Use the `load_probe_from_source` helper. Add options for visualization and normalization.<br>**CRITICAL:** The script must validate that the amplitude source and phase source probes have the exact same shape. If they do not, it must raise a ValueError and exit. Do not implement any automatic resizing.<br>**File:** `scripts/tools/create_hybrid_probe.py` |
| 2.C | Add Unit Tests for create_hybrid_probe.py | [D] | **Why:** To validate the probe mixing algorithm and file handling.<br>**How:** Create `tests/tools/test_create_hybrid_probe.py`. Add test cases for matching/mismatched dimensions, normalization, and correct preservation of amplitude and phase from their respective sources.<br>**File:** `tests/tools/test_create_hybrid_probe.py` |
| **Section 3: Integration Testing & Validation** | | | |
| 3.A | Create Integration Test for Decoupled Simulation | [D] | **Why:** To verify that the enhanced `simulate_and_save.py` works correctly with an external probe.<br>**How:** Create a new test file, `tests/test_decoupled_simulation.py`. The test should:<br>1. Create a dummy input NPZ and a separate external probe file.<br>2. Run `simulate_and_save.py` with the `--probe-file` flag.<br>3. Load the output and assert that the `probeGuess` in the output matches the external probe, not the one from the input NPZ.<br>**File:** `tests/test_decoupled_simulation.py` |
| 3.B | Run Full Test Suite | [D] | **Why:** To ensure no regressions were introduced in other parts of the codebase.<br>**How:** Run the entire project test suite from the root directory: `python -m unittest discover -s tests -p "test_*.py"`. All existing and new tests must pass.<br>**Command:** `python -m unittest discover -s tests -p "test_*.py"` |
| **Section 4: Visual Validation Workflow** | | | |
| 4.A | Create a Validation Jupyter Notebook | [D] | **Why:** To provide an interactive, visual environment for verifying the outputs of the new tools before proceeding to expensive training.<br>**How:** Create a new notebook: `notebooks/validate_probe_tools.ipynb`. Structure it with clear markdown sections for each validation step.<br>**File:** `notebooks/validate_probe_tools.ipynb` |
| 4.B | Visually Validate Hybrid Probe Creation | [D] | **Why:** To confirm the create_hybrid_probe.py tool correctly combines the synthetic amplitude with the experimental phase.<br>**How:** In the notebook, first generate a synthetic probe using ptycho.probe.get_default_probe(). Then, load the experimental probe from the source dataset (e.g., fly64_transposed.npz). Use the new tool to create the hybrid probe. Plot all three probes (synthetic, experimental, hybrid) side-by-side. Verify that the hybrid's amplitude matches the synthetic probe and its phase matches the experimental probe.<br>**Dependency:** Task 2.B |
| 4.C | Visually Validate Decoupled Simulation | [D] | **Why:** To confirm that `simulate_and_save.py` correctly uses the external probe.<br>**How:** In the notebook, run a small simulation using `--probe-file` with the newly created hybrid probe. Then, load the output `simulated_data.npz` and plot its `probeGuess`. It must visually match the hybrid probe, not the probe from the original input file.<br>**Dependency:** Task 2.A |
| 4.D | Visually Validate gridsize=2 Data | [D] | **Why:** To confirm that the new efficient subsampling in `raw_data.py` produces physically plausible neighbor groups.<br>**How:** In the notebook, load a gridsize=2 simulated dataset. Extract a few sample groups (e.g., the 4 diffraction patterns for a single training sample) and plot them. They should appear visually similar, representing overlapping views of the same object region.<br>**Dependency:** Task 1.C |
| 4.E | Commit Executed Notebook | [D] | **Why:** To create a permanent, visual record of the successful validation.<br>**How:** After all cells in the notebook have been run and the outputs are confirmed to be correct, save and commit the notebook with its outputs. This serves as a visual "receipt" that the tools were working correctly at the end of Phase 1.<br>**File:** `notebooks/validate_probe_tools.ipynb` |

---

## 🎯 **Success Criteria**

This phase is complete when:

1. **All tasks in the table above are marked [D] (Done).**
2. **The phase success test passes:** All new and existing unit/integration tests pass when running the full test suite.
3. **The new tools** (`create_hybrid_probe.py` and the enhanced `simulate_and_save.py`) are functional and can be used to manually prepare the data needed for the 2x2 study.
4. **The performance** of `ptycho/raw_data.py` for gridsize > 1 is significantly improved due to the new subsampling logic.
5. **All new code** is well-documented and follows project conventions.
6. **Visual validation passes:** The executed Jupyter notebook demonstrates that all tools are working correctly with visual confirmation of expected outputs.
7. **De-risking complete:** High confidence that the foundational tools are correct and subsequent expensive training runs will be built on a solid foundation.