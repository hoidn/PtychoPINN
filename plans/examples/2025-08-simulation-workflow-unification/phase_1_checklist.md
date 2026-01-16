# Phase 1: Core Refactoring - Replace Monolithic Function Checklist

**Initiative:** Simulation Workflow Unification
**Created:** 2025-08-02
**Phase Goal:** To refactor `scripts/simulation/simulate_and_save.py` to use explicit orchestration of modular functions instead of the monolithic `RawData.from_simulation` method.
**Deliverable:** A refactored `simulate_and_save.py` script that explicitly orchestrates coordinate grouping, patch extraction, and diffraction simulation, fixing the gridsize > 1 crash.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :------------------------------------------------- |
| **Section 0: Preparation & Context Priming** |
| 0.A | **Review Key Documents & APIs**                    | `[D]` | **Why:** To understand the architectural issues and correct patterns before coding. <br> **Docs:** `docs/DEVELOPER_GUIDE.md` Section 3.4 (Tensor formats), `docs/data_contracts.md` (NPZ format specs). <br> **APIs:** `ptycho.raw_data.group_coords()`, `ptycho.raw_data.get_image_patches()`, `ptycho.diffsim.illuminate_and_diffract()`, `ptycho.tf_helper._channel_to_flat()`, `ptycho.tf_helper._flat_to_channel()`. |
| 0.B | **Analyze Current Implementation**                  | `[D]` | **Why:** To understand the legacy code structure and identify all parts that need refactoring. <br> **How:** Open `scripts/simulation/simulate_and_save.py` and trace all calls to `RawData.from_simulation`. Document the current workflow and identify which parts map to the new modular functions. <br> **File:** `scripts/simulation/simulate_and_save.py` |
| 0.C | **Set Up Debug Environment**                        | `[D]` | **Why:** To enable comprehensive logging for debugging tensor shape issues. <br> **How:** Import logging module, set up debug-level logging with format that includes function names and line numbers. Create a `--debug` command-line flag if not present. <br> **Verify:** Running with `--debug` shows detailed log messages. |
| **Section 1: Input Loading & Validation** |
| 1.A | **Implement NPZ Input Loading**                     | `[D]` | **Why:** To properly load objectGuess and probeGuess from input files. <br> **How:** Use `np.load()` to load the NPZ file. Extract `objectGuess` and `probeGuess` arrays. Add validation to ensure arrays exist and have correct dtypes (complex64). <br> **Code:** `data = np.load(args.input_file); obj = data['objectGuess']; probe = data['probeGuess']` <br> **Verify:** Log shapes and dtypes of loaded arrays. |
| 1.B | **Add Probe Override Logic**                        | `[D]` | **Why:** To support the `--probe-file` argument for custom probe functions. <br> **How:** If `args.probe_file` is provided, load probe from that file instead. Validate probe shape matches expected dimensions. <br> **Code:** `if args.probe_file: probe_data = np.load(args.probe_file); probe = probe_data['probeGuess']` |
| **Section 2: Coordinate Generation & Grouping** |
| 2.A | **Import and Configure Parameters**                 | `[D]` | **Why:** To set up the legacy params system required by group_coords. <br> **How:** Import `ptycho.params as p`, set required parameters: `p.set('N', probe.shape[0])`, `p.set('gridsize', args.gridsize)`, `p.set('scan', args.scan_type)`. <br> **Note:** This is temporary compatibility with legacy system. |
| 2.B | **Generate Grouped Coordinates**                    | `[D]` | **Why:** To create scan positions and group them according to gridsize. <br> **How:** First generate scan coordinates based on scan type and n_images. Then call `group_coords()` with these coordinates: `xcoords, ycoords = generate_scan_positions(args.n_images, args.scan_type); offset_tuple = ptycho.raw_data.group_coords(xcoords, ycoords)`. This returns `(scan_offsets, group_neighbors)`. <br> **Note:** Check the exact API of group_coords() - it takes coordinates as input, not n_images. <br> **Verify:** Log the shapes - scan_offsets should be `(n_groups, 2)`, group_neighbors should be `(n_groups, gridsize²)`. |
| **Section 3: Patch Extraction** |
| 3.A | **Extract Object Patches (Y)**                      | `[D]` | **Why:** To extract ground truth patches in Channel Format for simulation. <br> **How:** Call `Y_patches = ptycho.raw_data.get_image_patches(obj, scan_offsets, group_neighbors)`. <br> **Expected shape:** `(n_groups, N, N, gridsize²)` for Channel Format. <br> **Verify:** Log shape and ensure it matches expected Channel Format. |
| 3.B | **Validate Patch Content**                          | `[D]` | **Why:** To ensure patches contain valid complex data. <br> **How:** Check that patches have non-zero values, contain both real and imaginary parts. Log min/max values for sanity check. <br> **Code:** `assert np.any(Y_patches != 0); assert np.any(np.imag(Y_patches) != 0)` |
| **Section 4: Format Conversion & Physics Simulation** |
| 4.A | **Convert Channel to Flat Format**                  | `[D]` | **Why:** The physics engine requires Flat Format input. <br> **How:** Import TensorFlow, create session if needed. Call `Y_flat = ptycho.tf_helper._channel_to_flat(Y_patches)`. <br> **Expected shape:** From `(B, N, N, C)` to `(B*C, N, N, 1)`. <br> **Critical:** This step is essential for gridsize > 1 to work correctly. |
| 4.B | **Prepare Probe for Simulation**                    | `[D]` | **Why:** Probe needs to be tiled to match the flat batch size. <br> **How:** Expand probe dimensions and tile: `probe_batch = np.tile(probe[np.newaxis, :, :, np.newaxis], (Y_flat.shape[0], 1, 1, 1))`. <br> **Note:** This explicit tiling is safe but could be optimized later using TensorFlow broadcasting. <br> **Verify:** probe_batch.shape should be `(B*C, N, N, 1)`. |
| 4.C | **Run Physics Simulation**                          | `[D]` | **Why:** To generate diffraction patterns from object patches. <br> **How:** Call `X_flat = ptycho.diffsim.illuminate_and_diffract(Y_flat, probe_batch, nphotons=args.nphotons)`. <br> **Note:** This returns amplitude (not intensity) as required by data contract. <br> **Verify:** X_flat should have same shape as Y_flat, all real values. |
| 4.D | **Convert Flat to Channel Format**                  | `[D]` | **Why:** To return diffraction data to consistent Channel Format. <br> **How:** Call `X_channel = ptycho.tf_helper._flat_to_channel(X_flat, gridsize=args.gridsize)`. <br> **Expected shape:** From `(B*C, N, N, 1)` back to `(B, N, N, C)`. |
| **Section 5: Output Assembly & Saving** |
| 5.A | **Reshape Arrays for NPZ Format**                   | `[D]` | **Why:** Data contract requires 3D arrays for diffraction and Y. <br> **How:** For gridsize=1, squeeze channel dimension: `diffraction = np.squeeze(X_channel, axis=-1)`. For gridsize>1, reshape to 3D by flattening groups: `diffraction = X_channel.reshape(-1, N, N)`. <br> **Final shape:** `(n_images, N, N)` where n_images = n_groups * gridsize². |
| 5.B | **Prepare Coordinate Arrays**                       | `[D]` | **Why:** NPZ format requires separate xcoords and ycoords arrays. <br> **How:** Extract base coordinates from scan_offsets. For gridsize=1: `xcoords = scan_offsets[:, 1]; ycoords = scan_offsets[:, 0]`. For gridsize>1: Need to expand coordinates for each neighbor in the group. Use group_neighbors indices to look up the correct coordinates for each pattern. <br> **Critical:** Each of the B*C diffraction patterns must have its correct unique coordinate pair. <br> **Verify:** Length of coords matches first dimension of diffraction array. |
| 5.C | **Assemble Output Dictionary**                      | `[D]` | **Why:** To create NPZ file conforming to data contract. <br> **How:** Create dict with required keys: `output = {'diffraction': diffraction.astype(np.float32), 'objectGuess': obj, 'probeGuess': probe, 'xcoords': xcoords, 'ycoords': ycoords}`. <br> **Note:** Ensure diffraction is float32 as per contract. |
| 5.D | **Save NPZ File**                                   | `[D]` | **Why:** To persist the simulated dataset. <br> **How:** Call `np.savez_compressed(args.output_file, **output)`. <br> **Verify:** File exists and can be loaded back successfully. |
| **Section 6: Cleanup & Finalization** |
| 6.A | **Add Comprehensive Error Handling**                | `[D]` | **Why:** To make the script robust and provide helpful error messages. <br> **How:** Wrap main logic in try-except blocks. Catch specific errors (FileNotFoundError, KeyError for missing arrays, ValueError for shape mismatches). Provide informative error messages that guide users. |
| 6.B | **Update Script Documentation**                     | `[D]` | **Why:** To document the new workflow and help future maintainers. <br> **How:** Update the script's docstring to explain the new modular workflow. Add inline comments explaining each major step, especially the format conversions. Document why RawData.from_simulation is no longer used. |
| 6.C | **Maintain Backward Compatibility**                 | `[D]` | **Why:** To ensure existing command-line interfaces still work. <br> **How:** Verify all existing command-line arguments are preserved and functional. Test with various argument combinations. Add any new arguments (like --debug) to argparse with appropriate defaults. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: `python scripts/simulation/simulate_and_save.py --input-file datasets/fly/fly001_transposed.npz --output-file test_sim.npz --gridsize 2` completes without errors and produces a valid NPZ file.
3. No regressions are introduced - the script still works correctly for gridsize=1.
4. The output NPZ file conforms to the data contract specifications in `docs/data_contracts.md`.

## 📝 Notes

- The critical fix for gridsize > 1 is in the format conversion steps (4.A and 4.D). These ensure tensors have the correct shapes for the physics simulation.
- Pay special attention to coordinate handling for gridsize > 1 - coordinates need to be properly expanded to match the flattened diffraction array.
- Use debug logging liberally to trace tensor shapes through the pipeline - this will help catch shape mismatches early.
- Remember that `diffraction` should contain amplitude values (not intensity) as per the data contract.