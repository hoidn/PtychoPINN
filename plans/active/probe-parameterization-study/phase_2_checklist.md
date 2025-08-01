# Phase 2: Enhance Simulation Script and Validate Decoupling Checklist

**Initiative:** Probe Parameterization Study
**Created:** 2025-08-01
**Phase Goal:** To integrate the new probe-loading logic into the main simulation script and validate that the object and probe sources can be successfully decoupled for both gridsize=1 and gridsize=2.
**Deliverable:** An enhanced `scripts/simulation/simulate_and_save.py` script with a new `--probe-file` argument, and a new integration test.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :--------------------- |
| **Section 0: Preparation & Verification** |
| 0.A | **Verify Phase 1 completion**                      | `[ ]` | **Why:** Ensure all dependencies from Phase 1 are ready. <br> **How:** Run `python -m pytest tests/workflows/test_simulation_utils.py tests/tools/test_create_hybrid_probe.py`. Verify `ptycho/workflows/simulation_utils.py` exists with both helper functions. <br> **Verify:** All tests pass, helper module is importable. |
| 0.B | **Review simulate_and_save.py structure**          | `[ ]` | **Why:** Understand existing code before modification. <br> **How:** Read `scripts/simulation/simulate_and_save.py`. Identify: argument parsing section, probe loading logic, simulation call. Note how `probeGuess` is currently extracted from input NPZ. <br> **File:** `scripts/simulation/simulate_and_save.py` |
| **Section 1: Enhance Simulation Script** |
| 1.A | **Add --probe-file argument to argparse**          | `[ ]` | **Why:** Enable external probe specification via command line. <br> **How:** Add `parser.add_argument('--probe-file', type=str, help='Path to external probe file (.npy or .npz) to override the probe from input file')`. Place after existing arguments. <br> **Verify:** `python scripts/simulation/simulate_and_save.py --help` shows new argument. |
| 1.B | **Import helper functions from Phase 1**           | `[ ]` | **Why:** Reuse validated probe loading logic. <br> **How:** Add imports at top: `from ptycho.workflows.simulation_utils import load_probe_from_source, validate_probe_object_compatibility`. Ensure imports are after sys.path manipulation if present. |
| 1.C | **Implement probe override logic**                 | `[ ]` | **Why:** Allow external probe to replace the default from input file. <br> **How:** After loading `data` dict, add conditional: `if args.probe_file: probe = load_probe_from_source(args.probe_file); validate_probe_object_compatibility(probe, data['objectGuess']); data['probeGuess'] = probe; logger.info(f"Overriding probe with external file: {args.probe_file}")`. <br> **Location:** After NPZ loading, before simulation call. |
| 1.D | **Add error handling for probe loading**           | `[ ]` | **Why:** Provide clear feedback on probe loading failures. <br> **How:** Wrap probe loading in try-except block. Catch `ValueError` from validation, `FileNotFoundError` for missing files, `KeyError` for missing NPZ keys. Log errors and re-raise with context: `raise ValueError(f"Failed to load probe from {args.probe_file}: {str(e)}")`. |
| 1.E | **Verify backward compatibility**                  | `[ ]` | **Why:** Ensure script still works without --probe-file. <br> **How:** Test script without the new argument: `python scripts/simulation/simulate_and_save.py --input-file datasets/fly/fly64_transposed.npz --output-file test_compat.npz --n-images 100`. <br> **Verify:** Script runs successfully, uses probe from input file. |
| **Section 2: Create Integration Test** |
| 2.A | **Create test file structure**                     | `[ ]` | **Why:** Establish comprehensive integration testing. <br> **How:** Create `tests/test_decoupled_simulation.py`. Import: unittest, tempfile, numpy, subprocess, os. Add imports for simulation_utils helpers. Structure with `class TestDecoupledSimulation(unittest.TestCase)`. |
| 2.B | **Implement test data setup**                      | `[ ]` | **Why:** Create controlled test data for validation. <br> **How:** In `setUp()` method: create small test object (128x128 complex), test probe (32x32 complex), save to temporary NPZ file. Create separate probe files (.npy and .npz format) for testing different input types. Use `tempfile.NamedTemporaryFile` for cleanup. |
| 2.C | **Test probe override with .npy file**             | `[ ]` | **Why:** Verify .npy probe loading works correctly. <br> **How:** Create test method `test_probe_override_npy()`. Run simulate_and_save.py with --probe-file pointing to .npy probe. Load output NPZ, verify `probeGuess` matches the override probe (use `np.allclose`). Check simulation completed successfully. |
| 2.D | **Test probe override with .npz file**             | `[ ]` | **Why:** Verify .npz probe loading works correctly. <br> **How:** Create test method `test_probe_override_npz()`. Similar to 2.C but with .npz file containing 'probeGuess' key. Verify correct key extraction and probe override. |
| 2.E | **Test gridsize=1 data consistency**               | `[ ]` | **Why:** Ensure data pipeline remains valid for standard case. <br> **How:** Create test `test_gridsize1_consistency()`. Run simulation with gridsize=1, verify output NPZ contains all required keys per data contract: diffraction, Y, xcoords, ycoords. Check shapes are consistent. |
| 2.F | **Test gridsize=2 data consistency**               | `[ ]` | **Why:** Validate overlap constraint handling with external probe. <br> **How:** Create test `test_gridsize2_consistency()`. Run with gridsize=2, verify Y array has correct shape (n_images, H, W) without channel dimension. Verify coordinate generation still works correctly. |
| 2.G | **Test error case: probe too large**               | `[ ]` | **Why:** Ensure validation catches invalid configurations. <br> **How:** Create test `test_probe_too_large_error()`. Create probe larger than object, attempt simulation. Verify script exits with error status, check error message contains "too large" validation error. Use `subprocess.run` with `capture_output=True`. |
| 2.H | **Test error case: invalid probe file**            | `[ ]` | **Why:** Verify graceful handling of bad inputs. <br> **How:** Create test `test_invalid_probe_file_error()`. Test with: non-existent file, NPZ without 'probeGuess' key, non-complex data. Verify appropriate error messages for each case. |
| **Section 3: Documentation Updates** |
| 3.A | **Update scripts/simulation/CLAUDE.md**            | `[ ]` | **Why:** Document new capability for future developers. <br> **How:** Add section describing --probe-file option. Include: purpose (decoupled probe studies), usage example, supported formats (.npy, .npz), validation rules (probe must be smaller than object). Follow existing doc style. |
| 3.B | **Update scripts/simulation/README.md**            | `[ ]` | **Why:** User-facing documentation for the enhancement. <br> **How:** Add --probe-file to options table. Add example section: "Using External Probe for Studies". Show command examples for both .npy and .npz usage. Mention use case for probe parameterization studies. |
| **Section 4: Validation & Testing** |
| 4.A | **Run all integration tests**                      | `[ ]` | **Why:** Verify complete implementation works correctly. <br> **How:** Execute `python -m pytest tests/test_decoupled_simulation.py -v`. All tests should pass. If any fail, debug and fix issues before proceeding. |
| 4.B | **Manual end-to-end test with real data**          | `[ ]` | **Why:** Validate with actual dataset beyond unit tests. <br> **How:** Using fly64 dataset: 1) Create hybrid probe using Phase 1 tool, 2) Run `python scripts/simulation/simulate_and_save.py --input-file datasets/fly/fly64_transposed.npz --probe-file hybrid_probe.npy --output-file test_e2e.npz --n-images 1000`. <br> **Verify:** Simulation completes, output file has overridden probe. |
| 4.C | **Verify no regression in existing tests**         | `[ ]` | **Why:** Ensure changes don't break existing functionality. <br> **How:** Run any existing tests for simulation module: `python -m pytest tests/ -k simulation`. Check project test suite if available. All existing tests should still pass. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: The new integration test passes, confirming that a simulation run with an external probe produces a valid, trainable dataset that adheres to all data contracts.
   - Run: `python -m pytest tests/test_decoupled_simulation.py -v`
   - All test methods should pass
3. Manual validation confirms the enhanced script works with real data
4. No regressions are introduced in the existing test suite
5. Documentation is updated to reflect the new capability