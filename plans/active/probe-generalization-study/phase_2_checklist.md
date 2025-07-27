# Phase 2: Experimental Probe Integration Checklist

**Initiative:** Probe Generalization Study
**Created:** 2025-07-22
**Phase Goal:** To create a reusable workflow for simulating data using the experimental probe from the fly64 dataset.
**Deliverable:** A new `.npz` file, `simulation_input_experimental_probe.npz`, containing a synthetic 'lines' object and the experimental probeGuess from `datasets/fly64/fly001_64_train_converted.npz`.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :------------------------------------------------- |
| **Section 0: Preparation & Analysis**
| 0.A | **Analyze Experimental Probe Structure**          | `[ ]` | **Why:** To understand the format and characteristics of the experimental probe. <br> **How:** Load `datasets/fly64/fly001_64_train_converted.npz` and examine the `probeGuess` array. Check shape, dtype, and complex structure. <br> **Code:** `import numpy as np; data = np.load('datasets/fly64/fly001_64_train_converted.npz'); print(f"ProbeGuess: shape={data['probeGuess'].shape}, dtype={data['probeGuess'].dtype}")` <br> **Expected:** Shape (64, 64), complex64 dtype. |
| 0.B | **Study Synthetic Lines Generation Process**       | `[ ]` | **Why:** To understand how to generate a compatible synthetic 'lines' object. <br> **How:** Review `scripts/simulation/run_with_synthetic_lines.py` lines 45-87. Focus on `generate_and_save_synthetic_input()` function. <br> **Key Points:** Uses `sim_object_image(size=full_object_size)` with `p.set('data_source', 'lines')`, object size is 3.5x probe size. <br> **API:** `ptycho.diffsim.sim_object_image()` and `ptycho.params.set()`. |
| 0.C | **Verify Data Contracts Compliance**              | `[ ]` | **Why:** To ensure the output file will be compatible with `simulate_and_save.py`. <br> **How:** Read `docs/data_contracts.md` section on input file requirements. Confirm required keys: `objectGuess` (complex, large), `probeGuess` (complex, N×N). <br> **Reference:** Both arrays must be complex64, object must be significantly larger than probe for scanning space. |
| **Section 1: Helper Script Implementation**
| 1.A | **Create Script Directory Structure**             | `[ ]` | **Why:** To organize the helper script in a logical location. <br> **How:** Create `scripts/tools/create_experimental_probe_input.py` as a standalone tool. <br> **Pattern:** Follow naming convention from other tools like `visualize_dataset.py`, `split_dataset_tool.py`. <br> **Location:** `scripts/tools/` is the established location for data preparation utilities. |
| 1.B | **Implement Experimental Probe Loading**          | `[ ]` | **Why:** To extract the experimental probe from the fly64 dataset. <br> **How:** Create function `load_experimental_probe(fly64_path: str) -> np.ndarray`. Load the NPZ file, extract `probeGuess`, validate it's complex64. <br> **API:** `np.load(fly64_path)['probeGuess']` <br> **Validation:** Check shape is (64, 64), dtype is complex64, no NaN/inf values. <br> **Error Handling:** Raise clear errors if file missing or probe malformed. |
| 1.C | **Implement Synthetic Lines Object Generation**   | `[ ]` | **Why:** To generate a compatible synthetic object using the same process as the working pipeline. <br> **How:** Create function `generate_synthetic_lines_object(probe_size: int) -> np.ndarray`. Use same logic as `run_with_synthetic_lines.py`. <br> **Steps:** `p.set('data_source', 'lines')`, `p.set('size', object_size)`, call `sim_object_image(size=object_size)`. <br> **Size Rule:** Object size = int(probe_size * 3.5) to match existing workflow. |
| 1.D | **Implement NPZ Output Generation**                | `[ ]` | **Why:** To create the required output file format compatible with `simulate_and_save.py`. <br> **How:** Create function `save_input_file(object_guess: np.ndarray, probe_guess: np.ndarray, output_path: str)`. <br> **Format:** `np.savez(output_path, objectGuess=object_guess.astype(np.complex64), probeGuess=probe_guess.astype(np.complex64))` <br> **Validation:** Ensure both arrays are complex64 before saving. |
| 1.E | **Implement Command Line Interface**              | `[ ]` | **Why:** To make the script usable from command line with proper argument handling. <br> **How:** Use `argparse` with arguments: `--fly64-file` (required), `--output-file` (required), `--probe-size` (optional, default 64). <br> **Example:** `python scripts/tools/create_experimental_probe_input.py --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file simulation_input_experimental_probe.npz` <br> **Validation:** Check input file exists, output directory is writable. |
| **Section 2: Script Integration & Testing**
| 2.A | **Add Error Handling & Logging**                  | `[ ]` | **Why:** To provide clear feedback and handle edge cases gracefully. <br> **How:** Add try/except blocks around file operations and computation steps. Use print statements for progress logging. <br> **Error Cases:** Missing input file, corrupted NPZ data, write permissions, invalid probe dimensions. <br> **Logging:** Print probe shape, object shape, output file path for user verification. |
| 2.B | **Test Script with Fly64 Dataset**                | `[ ]` | **Why:** To verify the script works correctly with real experimental data. <br> **How:** Run `python scripts/tools/create_experimental_probe_input.py --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file simulation_input_experimental_probe.npz`. <br> **Verify:** Output file is created, contains required keys, arrays have correct shapes and types. <br> **Debug:** If errors occur, check file paths, data access permissions, and array manipulations. |
| 2.C | **Validate Output File Structure**                | `[ ]` | **Why:** To ensure the generated file conforms to data contracts and will work with simulation pipeline. <br> **How:** Load the output file and check: keys present (`objectGuess`, `probeGuess`), correct shapes, correct dtypes (complex64), reasonable value ranges. <br> **Code:** `data = np.load('simulation_input_experimental_probe.npz'); print(f"Keys: {list(data.keys())}"); print(f"Object: {data['objectGuess'].shape}, {data['objectGuess'].dtype}"); print(f"Probe: {data['probeGuess'].shape}, {data['probeGuess'].dtype}")` |
| **Section 3: Simulation Pipeline Verification**
| 3.A | **Test Integration with simulate_and_save.py**    | `[ ]` | **Why:** This is the key success criterion - the output must work with the existing simulation pipeline. <br> **How:** Run `python scripts/simulation/simulate_and_save.py --input-file simulation_input_experimental_probe.npz --output-file test_experimental_sim.npz --n-images 100`. <br> **Timeout:** Allow 2-3 minutes for small test. <br> **Expected:** Script completes without errors and generates test_experimental_sim.npz. |
| 3.B | **Validate Simulation Output Structure**          | `[ ]` | **Why:** To confirm the simulation generated a valid, trainable dataset. <br> **How:** Load `test_experimental_sim.npz` and verify it contains all required keys: `objectGuess`, `probeGuess`, `diffraction`, `xcoords`, `ycoords`. <br> **Shapes:** diffraction should be (100, 64, 64), coordinates should be (100,) each. <br> **Reference:** Use `docs/data_contracts.md` as validation checklist. |
| 3.C | **Compare Experimental vs Synthetic Probes**      | `[ ]` | **Why:** To understand the visual and structural differences between probe types for analysis. <br> **How:** Load both the experimental probe and a default synthetic probe. Generate side-by-side amplitude/phase plots. <br> **Code:** Use matplotlib to create 2×2 subplot showing amplitude and phase for both probes. <br> **Save:** Save comparison as `probe_comparison.png` for documentation. |
| **Section 4: Documentation & Cleanup**
| 4.A | **Add Script Documentation**                      | `[ ]` | **Why:** To ensure the tool is usable by others and follows project standards. <br> **How:** Add comprehensive docstring to main script explaining purpose, usage, arguments, and examples. <br> **Include:** Function docstrings following Google style, usage examples in module docstring, error handling documentation. <br> **Reference:** Follow documentation patterns from other tools in `scripts/tools/`. |
| 4.B | **Create Script Usage Documentation**             | `[ ]` | **Why:** To provide users with clear instructions for the new workflow capability. <br> **How:** Consider adding a section to `scripts/tools/README.md` or `scripts/tools/CLAUDE.md` about the new experimental probe input generation capability. <br> **Content:** Brief description, example usage, integration with simulation pipeline. <br> **Optional:** Only if it significantly enhances workflow documentation. |
| 4.C | **Clean Up Test Outputs**                         | `[ ]` | **Why:** To remove temporary files created during testing and verification. <br> **How:** Remove `test_experimental_sim.npz` and any other temporary simulation outputs. <br> **Preserve:** Keep `simulation_input_experimental_probe.npz` as it's the deliverable, and `probe_comparison.png` as documentation. <br> **Document:** Note file locations and purposes in implementation notes. |
| **Section 5: Final Validation & Commitment**
| 5.A | **Run Complete Integration Test**                 | `[ ]` | **Why:** To demonstrate the complete end-to-end workflow before phase completion. <br> **How:** Execute the full pipeline: create input file → run simulation → verify output. Commands: <br> 1. `python scripts/tools/create_experimental_probe_input.py --fly64-file datasets/fly64/fly001_64_train_converted.npz --output-file simulation_input_experimental_probe.npz` <br> 2. `python scripts/simulation/simulate_and_save.py --input-file simulation_input_experimental_probe.npz --output-file final_test_experimental_sim.npz --n-images 50` <br> **Success:** Both commands complete successfully, final output file validates correctly. |
| 5.B | **Document Implementation Decisions**             | `[ ]` | **Why:** To record key design choices for future reference and debugging. <br> **How:** Update the "Implementation Notes" section at the bottom of this checklist with: probe loading method, object generation parameters, file format decisions, integration approach. <br> **Include:** Any issues encountered, performance observations, compatibility considerations. |
| 5.C | **Commit Phase 2 Changes**                        | `[ ]` | **Why:** To create a checkpoint for Phase 2 completion and enable Phase 3 work. <br> **How:** Stage changes: `git add .`, commit with descriptive message: `git commit -m "Phase 2: Implement experimental probe integration workflow\n\n- Create create_experimental_probe_input.py tool for extracting fly64 probe\n- Generate synthetic lines object compatible with experimental probe\n- Validate end-to-end simulation pipeline with experimental probe\n- Enable Phase 3 2x2 study execution"`. <br> **Verify:** Confirm git status shows clean working directory after commit. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: `python scripts/simulation/simulate_and_save.py --input-file simulation_input_experimental_probe.npz` runs successfully and produces valid output.
3. The deliverable `simulation_input_experimental_probe.npz` file exists and contains the experimental probe from fly64 dataset plus a synthetic 'lines' object.
4. The new helper script `scripts/tools/create_experimental_probe_input.py` is implemented and documented.
5. Integration testing confirms the experimental probe workflow is compatible with the existing simulation pipeline.

## 📊 Implementation Notes

### Key Design Decisions:
- **Tool Location:** Placed in `scripts/tools/` to follow established project organization for data preparation utilities.
- **Object Generation:** Uses identical logic to `run_with_synthetic_lines.py` to ensure consistency with tested workflow.
- **File Format:** Follows exact NPZ format expected by `simulate_and_save.py` with `objectGuess` and `probeGuess` keys.
- **Probe Size:** Assumes 64×64 probe size to match fly64 dataset, but allows override via command line.

### Expected Challenges:
- **Data Type Consistency:** Ensuring complex64 dtype throughout the pipeline to prevent TensorFlow errors.
- **Array Shape Validation:** Confirming object size is appropriate for probe size to allow adequate scanning range.
- **File Path Handling:** Robust handling of relative/absolute paths and file existence checking.

### Integration Points:
- **Input:** Experimental probe from `datasets/fly64/fly001_64_train_converted.npz`
- **Processing:** Synthetic object generation via `ptycho.diffsim.sim_object_image()`
- **Output:** Compatible input file for `scripts/simulation/simulate_and_save.py`
- **Validation:** End-to-end test ensures Phase 3 can proceed with confidence

### Performance Expectations:
- **Script Execution:** Should complete in under 10 seconds for object generation and file I/O
- **Test Simulation:** 50-100 image simulation should complete in 1-2 minutes
- **Memory Usage:** Minimal - only holds object and probe arrays temporarily in memory