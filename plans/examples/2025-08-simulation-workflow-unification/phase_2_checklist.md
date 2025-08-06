# Phase 2: Integration Testing & Validation Checklist

**Initiative:** Simulation Workflow Unification
**Created:** 2025-08-02
**Phase Goal:** To create a comprehensive test suite that validates the refactored simulation pipeline for both gridsize=1 and gridsize > 1 cases.
**Deliverable:** A new test file `tests/simulation/test_simulate_and_save.py` with comprehensive integration tests covering all validation scenarios from the R&D plan.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :------------------------------------------------- |
| **Section 0: Test Infrastructure Setup** |
| 0.A | **Create Test Directory Structure**                 | `[D]` | **Why:** To organize tests following project conventions. <br> **How:** Create `tests/simulation/` directory if it doesn't exist. Add `__init__.py` files to make it a proper Python package. <br> **Command:** `mkdir -p tests/simulation; touch tests/simulation/__init__.py` <br> **Verify:** Directory exists and is importable. |
| 0.B | **Set Up Test File with Imports**                   | `[D]` | **Why:** To establish the test framework and necessary imports. <br> **How:** Create `tests/simulation/test_simulate_and_save.py`. Import unittest, numpy, tempfile, subprocess. Import data contract validation utilities. <br> **Code:** `import unittest; import numpy as np; import tempfile; import subprocess; import os` |
| 0.C | **Create Test Data Fixtures**                       | `[D]` | **Why:** To have consistent test data for all test cases. <br> **How:** Create helper function to generate minimal valid NPZ files with known properties. Include small objectGuess (e.g., 128x128), probeGuess (e.g., 32x32). <br> **Code:** `def create_test_npz(obj_size=128, probe_size=32): obj = np.random.complex64(...); probe = np.random.complex64(...); return {'objectGuess': obj, 'probeGuess': probe}` |
| **Section 1: Gridsize=1 Regression Tests** |
| 1.A | **Implement Basic Gridsize=1 Test**                 | `[D]` | **Why:** To ensure the refactoring doesn't break existing functionality. <br> **How:** Create test that runs simulate_and_save.py with gridsize=1, n_images=100. Verify output NPZ exists and has correct structure. <br> **Test name:** `test_gridsize1_basic_functionality` <br> **Verify:** Output file exists, can be loaded. |
| 1.B | **Verify Gridsize=1 Output Shapes**                 | `[D]` | **Why:** To ensure single-channel data has correct dimensions. <br> **How:** Load output NPZ, check shapes: diffraction should be (100, 32, 32), xcoords/ycoords should be (100,). No channel dimension should exist. <br> **Assertions:** `self.assertEqual(data['diffraction'].shape, (100, 32, 32))` |
| 1.C | **Validate Gridsize=1 Data Types**                  | `[D]` | **Why:** To ensure data contract compliance. <br> **How:** Check dtypes: diffraction should be float32, objectGuess/probeGuess should be complex64, coordinates should be float64. <br> **Assertions:** `self.assertEqual(data['diffraction'].dtype, np.float32)` |
| **Section 2: Gridsize=2 Correctness Tests** |
| 2.A | **Implement Basic Gridsize=2 Test**                 | `[D]` | **Why:** To verify the core bug fix works. <br> **How:** Create test that runs simulate_and_save.py with gridsize=2, n_images=100. This should complete without ValueError. <br> **Test name:** `test_gridsize2_no_crash` <br> **Verify:** Process completes with return code 0. |
| 2.B | **Verify Gridsize=2 Output Shapes**                 | `[D]` | **Why:** To ensure multi-channel data is correctly flattened. <br> **How:** Load output NPZ. For n_groups=25 with gridsize=2, diffraction should be (100, 32, 32) where 100 = 25 * 4. Coordinates should also be (100,). <br> **Note:** The 4 patterns per group should be flattened into the batch dimension. |
| 2.C | **Validate Gridsize=2 Coordinate Expansion**        | `[D]` | **Why:** To ensure each pattern has unique, correct coordinates. <br> **How:** Check that coordinates have correct number of unique values and proper spatial relationships for neighbors. Plot first few groups to verify neighbor patterns. <br> **Critical:** This validates the tricky coordinate expansion logic. |
| **Section 3: Feature-Specific Tests** |
| 3.A | **Test Probe Override Functionality**               | `[D]` | **Why:** To verify --probe-file argument works correctly. <br> **How:** Create test with custom probe NPZ file. Run simulation with --probe-file. Verify output probeGuess matches the custom probe, not the one from input file. <br> **Test name:** `test_probe_override` |
| 3.B | **Test Available Scan Types**                       | `[D]` | **Why:** To ensure supported scan patterns work correctly. <br> **How:** First check what scan types the refactored script actually supports (likely just 'random' initially). Test each available type and verify it produces valid output with appropriate coordinate patterns. <br> **Note:** Simplify to test only what's implemented rather than assuming all scan types are available. |
| **Section 4: Data Contract Compliance** |
| 4.A | **Implement Comprehensive Contract Test**           | `[D]` | **Why:** To ensure output strictly follows data contract. <br> **How:** Create test that validates all required keys exist: diffraction, objectGuess, probeGuess, xcoords, ycoords. Check optional keys like scan_index. <br> **Reference:** `docs/data_contracts.md` |
| 4.B | **Verify Amplitude vs Intensity**                   | `[D]` | **Why:** Data contract requires amplitude, not intensity. <br> **How:** Verify diffraction values are in reasonable amplitude range (typically 0-1 or 0-sqrt(max_photons)). Check that values are not squared intensities. <br> **Note:** This is a common source of errors. |
| **Section 5: Content Validation Tests** |
| 5.A | **Test Physical Plausibility**                      | `[D]` | **Why:** To catch potential simulation errors. <br> **How:** Verify diffraction patterns are non-zero, have expected statistical properties. Check that patterns show expected Fourier transform characteristics. <br> **Assertions:** `self.assertTrue(np.all(data['diffraction'] >= 0))` |
| 5.B | **[Optional] Test Probe Illumination Effects**      | `[S]` | **Why:** To ensure probe properly modulates object. <br> **How:** Compare simulations with different probe intensities. Brighter probe should yield higher diffraction amplitudes. Verify probe shape affects diffraction patterns. <br> **Note:** This is a stretch goal for physical validation - can be skipped if too complex to implement robustly. |
| **Section 6: Performance & Benchmarking** |
| 6.A | **Create Performance Benchmark Test**               | `[D]` | **Why:** To ensure no significant performance regression. <br> **How:** Time execution for standard dataset (e.g., 1000 images). Compare with baseline timing if available. Set reasonable timeout (e.g., 60 seconds for 1000 images). <br> **Note:** Can be skipped with @unittest.skipIf decorator if needed. |
| 6.B | **[Manual] Test Memory Usage**                      | `[S]` | **Why:** To catch memory leaks or inefficiencies. <br> **How:** Monitor memory usage during large simulation (e.g., 5000 images). Ensure memory usage scales linearly with n_images, not quadratically. <br> **Tool:** Can use `psutil` or `resource` module. <br> **Note:** This is better suited as a manual benchmark during development rather than an automated CI test. Document results in implementation notes. |
| **Section 7: Visual Validation & Integration** |
| 7.A | **Create Visual Validation Script**                 | `[D]` | **Why:** To enable manual inspection of results. <br> **How:** Write script that uses `scripts/tools/visualize_dataset.py` to create plots of test outputs. Include amplitude, phase, and diffraction pattern visualizations. <br> **File:** `tests/simulation/visualize_test_outputs.py` |
| 7.B | **Test Integration with Training Pipeline**         | `[D]` | **Why:** To ensure output works with downstream tools. <br> **How:** Create test that attempts to load simulation output with `ptycho.loader` and verifies it can be used for training. Check that RawData can load the file successfully. <br> **Note:** This ensures end-to-end compatibility. |
| **Section 8: Edge Cases & Error Handling** |
| 8.A | **Test Invalid Input Handling**                     | `[D]` | **Why:** To ensure graceful error messages. <br> **How:** Test with missing keys in input NPZ, mismatched probe/object sizes, invalid gridsize values. Verify helpful error messages are shown. <br> **Example:** Missing objectGuess should show "KeyError: 'objectGuess' not found in input file". |
| 8.B | **Test Boundary Conditions**                        | `[D]` | **Why:** To catch edge case bugs. <br> **How:** Test with minimum values (n_images=1, gridsize=1), maximum reasonable values, non-square arrays if supported. Verify all complete successfully. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: `pytest tests/simulation/test_simulate_and_save.py -v` shows all tests passing with 100% success rate.
3. No regressions are introduced in the existing test suite.
4. Test coverage for the refactored simulate_and_save.py is at least 80%.

## 📝 Notes

- Use `subprocess.run()` to invoke the actual simulate_and_save.py script in tests, ensuring we test the real command-line interface.
- Use `tempfile.TemporaryDirectory()` for test outputs to avoid cluttering the filesystem.
- Consider using `@pytest.mark.parametrize` or `unittest.TestCase.subTest()` for testing multiple parameter combinations.
- The gridsize=2 coordinate validation (task 2.C) is particularly important as it tests the most complex part of the refactoring.
- Performance tests can be marked as optional if they take too long in CI environments.