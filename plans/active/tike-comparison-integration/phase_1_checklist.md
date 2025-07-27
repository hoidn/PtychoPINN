# Phase 1: Standalone Tike Reconstruction Script Checklist

**Initiative:** Tike Comparison Integration  
**Created:** 2025-01-25  
**Phase Goal:** To create a robust, reusable, and well-documented command-line script that performs a Tike reconstruction on a given dataset and saves the output in a standardized .npz format with rich metadata.  
**Deliverable:** A new script `scripts/reconstruction/run_tike_reconstruction.py` and a sample `tike_reconstruction.npz` artifact generated from a standard test dataset.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :------------------------------------------------- |
| **Section 0: Preparation & Setup** |
| 0.A | **Create Directory Structure**                     | `[ ]` | **Why:** To establish the correct location for the new script. <br> **How:** Create the directory `scripts/reconstruction/`. <br> **Verify:** The directory `scripts/reconstruction/` exists. |
| 0.B | **Create Script File**                             | `[ ]` | **Why:** To create the file for the new Tike reconstruction script. <br> **How:** Create an empty file `scripts/reconstruction/run_tike_reconstruction.py`. Add a basic shebang `#!/usr/bin/env python3` and a module-level docstring explaining its purpose. <br> **Permissions:** Make the script executable: `chmod +x scripts/reconstruction/run_tike_reconstruction.py`. |
| 0.C | **Review Tike API & Existing Scripts**             | `[ ]` | **Why:** To understand the Tike API and reuse existing patterns. <br> **How:** Review the `tike.ptycho.reconstruct` function signature. Examine the existing `scripts/tikerecon.py` for a baseline implementation of Tike's parameter setup. <br> **Key Insight:** Note the setup for `PtychoParameters`, `RpieOptions`, `ObjectOptions`, etc. |
| **Section 1: Script Implementation** |
| 1.A | **Implement Command-Line Interface**               | `[ ]` | **Why:** To make the script configurable and user-friendly. <br> **How:** Use `argparse`. Add the following arguments: <br> - `input_npz` (required, positional) <br> - `output_dir` (required, positional) <br> - `--iterations` (optional, type int, default 1000) <br> - `--num-gpu` (optional, type int, default 1) <br> - `--quiet`, `--verbose` for logging control. <br> **Reference:** Use `ptycho/cli_args.py` for logging arguments. |
| 1.B | **Integrate Centralized Logging**                 | `[ ]` | **Why:** To align with project standards for logging and traceability. <br> **How:** Import `setup_logging` from `ptycho.log_config`. Call `setup_logging(Path(args.output_dir), **get_logging_config(args))` at the start of the main function. <br> **API:** Use `logging.getLogger(__name__)` to get the logger instance. |
| 1.C | **Implement Data Loading**                         | `[ ]` | **Why:** To load the necessary arrays from the input dataset. <br> **How:** Create a function `load_tike_data(npz_path)`. It should load the `.npz` file and extract `diffraction`, `probeGuess`, `xcoords`, and `ycoords`. Handle potential key name variations (e.g., `diff3d` vs `diffraction`). <br> **API:** Use `numpy.load()`. Add validation to ensure all required keys are present. |
| 1.D | **Implement Tike Parameter Setup**                | `[ ]` | **Why:** To configure the Tike reconstruction algorithm. <br> **How:** Create a function `configure_tike_parameters(...)`. This function will set up `RpieOptions`, `ObjectOptions`, `ProbeOptions`, and `PtychoParameters` based on the loaded data and CLI arguments. <br> **Key Params:** Use `use_adaptive_moment=True`, `noise_model="poisson"`, and `force_centered_intensity=True` for robust reconstruction. |
| 1.E | **Implement Core Reconstruction Logic**           | `[ ]` | **Why:** To execute the Tike algorithm and capture performance metrics. <br> **How:** In the main function, call `tike.ptycho.reconstruct`. Wrap the call with `time.time()` to measure the `computation_time_seconds`. Capture the final error/convergence metric from the result object. |
| **Section 2: Data Contract & Output Generation** |
| 2.A | **Implement Output Data Contract**                | `[ ]` | **Why:** To produce a standardized, reusable artifact for Phase 2. <br> **How:** Create a function `save_tike_results(...)`. It should save a single `.npz` file named `tike_reconstruction.npz` inside the `output_dir`. <br> **API:** Use `numpy.savez_compressed()`. |
| 2.B | **Save Reconstructed Arrays**                      | `[ ]` | **Why:** To store the primary outputs of the reconstruction. <br> **How:** Inside `save_tike_results`, save the final object and probe from the Tike result object. <br> **Keys:** Use the exact key names `reconstructed_object` and `reconstructed_probe`. Ensure they are 2D complex arrays. |
| 2.C | **Assemble and Save Metadata**                    | `[ ]` | **Why:** To ensure reproducibility and provide context for the reconstruction. <br> **How:** Create a Python dictionary containing all the required metadata (algorithm, version, iterations, timing, parameters). Save this dictionary as a single-element object array under the key `metadata`. <br> **API:** `np.savez_compressed(..., metadata=np.array([metadata_dict]))`. |
| 2.D | **Implement Visualization Output**                | `[ ]` | **Why:** To provide immediate visual feedback on the reconstruction quality. <br> **How:** Create a function `save_visualization(...)`. It should generate a 2x2 plot showing the amplitude and phase of the final reconstructed object and probe. Save it as `reconstruction_visualization.png` in the `output_dir`. |
| **Section 3: Validation & Testing** |
| 3.A | **Test with Standard Dataset**                    | `[ ]` | **Why:** To verify the end-to-end functionality of the script. <br> **How:** Run the script on a known-good test dataset, such as `datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz`. Use a small number of iterations (e.g., `--iterations 10`) for a quick test. <br> **Verify:** The script should complete without errors. |
| 3.B | **Validate Output Artifact**                      | `[ ]` | **Why:** To confirm the script produces a file that meets the data contract. <br> **How:** Write a small, temporary validation script or use an interactive session to load the generated `tike_reconstruction.npz`. Check for the presence and correct shapes/dtypes of `reconstructed_object`, `reconstructed_probe`, and `metadata`. <br> **Example Check:** `data = np.load('...'); meta = data['metadata'][0]; assert meta['iterations'] == 10`. |
| 3.C | **Validate Visualization**                        | `[ ]` | **Why:** To ensure the visual feedback is correct. <br> **How:** Manually inspect the generated `reconstruction_visualization.png`. Verify that the plots are plausible and the titles/labels are correct. |
| **Section 4: Documentation & Finalization** |
| 4.A | **Add Comprehensive Docstrings**                  | `[ ]` | **Why:** To make the script maintainable and understandable. <br> **How:** Add a module-level docstring explaining the script's purpose, usage, and I/O. Add clear docstrings to all functions. |
| 4.B | **Create README for Directory**                   | `[ ]` | **Why:** To document the purpose of the new `scripts/reconstruction/` directory. <br> **How:** Create `scripts/reconstruction/README.md`. Explain that this directory holds scripts for generating reconstructions using various algorithms (starting with Tike). Include a usage example for the new script. |
| 4.C | **Commit Phase 1 Changes**                        | `[ ]` | **Why:** To create a clean checkpoint for the completion of Phase 1. <br> **How:** Stage all new and modified files. Commit with a descriptive message: `git commit -m "Phase 1: Implement standalone Tike reconstruction script\n\n- Create run_tike_reconstruction.py with CLI and logging\n- Implement data contract for output NPZ with metadata\n- Add visualization for immediate feedback"`. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: `python scripts/reconstruction/run_tike_reconstruction.py --input-npz <test_data.npz> --output-dir <tike_output> --iterations 10` runs successfully.
3. The output directory `<tike_output>/` contains `tike_reconstruction.npz` and `reconstruction_visualization.png`.
4. The `tike_reconstruction.npz` file is validated and conforms perfectly to the data contract, including the metadata dictionary.
5. The new script is well-documented and follows project coding standards.

---

## 📋 **Data Contract Specification**

For reference, the output `tike_reconstruction.npz` file must contain:

### Required Arrays:
- **`reconstructed_object`**: Complex 2D array `(M, M)` - The final reconstructed object
- **`reconstructed_probe`**: Complex 2D array `(N, N)` - The final reconstructed probe  

### Required Metadata:
- **`metadata`**: Single-element object array containing a dictionary with:
  - `algorithm`: `"tike"`
  - `tike_version`: Version string from `tike.__version__`
  - `iterations`: Number of iterations used
  - `computation_time_seconds`: Float timing measurement
  - `parameters`: Dictionary of all reconstruction parameters used
  - `input_file`: Path to the input NPZ file
  - `timestamp`: ISO format timestamp of reconstruction