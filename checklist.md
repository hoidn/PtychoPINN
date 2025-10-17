### **Updated LLM Agent Checklist (Coordinate-Focused Debugging)**

**Objective:** Isolate the root cause of the "zero variance" bug by testing coordinate-related hypotheses, without modifying the `.cache()` call.
**Test Case:** Use `tike_outputs/fly001/fly001_reconstructed.npz` as the input dataset.

---

#### **Phase 1: Instrument the Codebase for Coordinate Tracing**

This phase adds temporary, detailed logging to the functions involved in coordinate processing.

*   [x] **1.1. Instrument `calculate_relative_coords`:**
    *   [x] Read the file `ptycho/raw_data.py`.
    *   [x] Locate the function `calculate_relative_coords`.
    *   [x] **Action:** Add `print()` statements to display the `shape` and the first 2-3 rows of the `nn_indices` array after it is calculated. This will test **Hypothesis 2d (Grouping Logic Error)**.
    *   [x] **Action:** Add `print()` statements to display the `shape` and the first element (`[0, ...]`) of `coords_nn`, `coords_offsets`, and `coords_relative`. This will help test **Hypothesis 2b (Broadcasting Error)**.
    *   [x] Save the modified `ptycho/raw_data.py` file.

*   [x] **1.2. Instrument `get_image_patches`:**
    *   [x] Read the file `ptycho/raw_data.py`.
    *   [x] Locate the function `get_image_patches`.
    *   [x] **Action:** Inside the `for i in range(B * c)` loop, add a `print()` statement that executes only for the first 5 iterations (`if i < 5:`). This print should display the iteration number `i` and the value of the `offset` variable. This will test **Hypothesis 2a (Integer Truncation)**.
    *   [x] Save the modified `ptycho/raw_data.py` file.

*   [ ] **1.3. Instrument `hh.translate` (If necessary):**
    *   [ ] **Hold Action:** Do not modify this file yet. We will only proceed to instrument this core helper if the diagnostics from steps 1.1 and 1.2 show that varied, floating-point offsets are being passed correctly.

---

#### **Phase 2: Execute and Analyze in a Jupyter Notebook**

This phase uses an interactive notebook to run the instrumented code and analyze the diagnostic output.

*   [ ] **2.1. Create Jupyter Notebook:** Create a new file named `debug_coordinate_flow.ipynb`.

*   [ ] **2.2. Populate Notebook - Cell by Cell:**
    *   [ ] **Cell 1 (Imports & Config):** Write Python code to import necessary modules (`numpy`, `ptycho.nongrid_simulation`, etc.) and set up a basic configuration with `gridsize=1` and `N=128`.
    *   [ ] **Cell 2 (Load Data):** Set `npz_path` to `tike_outputs/fly001/fly001_reconstructed.npz`. Call `load_probe_object(npz_path)`.
    *   [ ] **Cell 3 (Run Simulation):**
        *   [ ] **Action:** Call `simulate_from_npz` with a small `nimages` (e.g., 500) and a fixed `seed=42`. This will trigger all the diagnostic `print` statements that were added in Phase 1.

*   [x] **2.3. Agent Self-Correction and Analysis:**
    *   [x] **Condition:** Carefully examine the complete print output from Cell 3.
    *   [x] **Analysis Step 1: Check `nn_indices` output.**
        *   ✅ **RESULT:** The rows of `nn_indices` are different from each other (0, 1, 2...).
        *   ✅ **CONCLUSION:** Hypothesis 2d (Grouping Logic Error) is **FALSE**. Proceed.
    *   [x] **Analysis Step 2: Check `coords_relative` output.**
        *   ❌ **RESULT:** `coords_relative[0]` shows `[[[0.] [0.]]]` - ALL ZEROS!
        *   🚨 **CONCLUSION:** The bug is a broadcasting error in `get_relative_coords`. **ROOT CAUSE IDENTIFIED.**
    *   [x] **Analysis Step 3: Check the `offset` loop output from `get_image_patches`.**
        *   ✅ **RESULT:** The 5 printed `offset` vectors are **different by large amounts** with floating-point precision.
        *   ✅ **CONCLUSION:** Hypothesis 2a (Integer Truncation) is **FALSE**. No truncation issue in `hh.translate`.

---

#### **Phase 3: Formulate a Fix and Propose Next Steps**

*   [ ] **3.1. Propose a Code Change:** Based on the analysis in 2.3, draft a specific code modification to fix the identified issue.
    *   *If Hypothesis 2a is confirmed:* The proposal would be to investigate `hh.translate` to ensure it handles floating-point shifts correctly, possibly by using a library function known for sub-pixel accuracy like `scipy.ndimage.shift`.
    *   *If another hypothesis is confirmed:* The proposal will target the specific line of code identified as faulty.

*   [ ] **3.2. Outline Verification:** Describe how to re-run the notebook after applying the proposed fix to confirm that the simulation variance is now non-zero.

This updated checklist provides a rigorous, methodical way to test the coordinate-related hypotheses. It forces the agent to trace the data step-by-step and make a logical deduction based on the instrumented output, rather than jumping to conclusions.

