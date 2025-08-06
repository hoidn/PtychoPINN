# Phase 3: Comprehensive Equivalence Testing Checklist

**Initiative:** High-Performance Patch Extraction Refactoring
**Created:** 2025-08-03
**Phase Goal:** To create a rigorous test suite that proves the new implementation is numerically identical to the old one across all edge cases.
**Deliverable:** A comprehensive test file `tests/test_patch_extraction_equivalence.py` with performance benchmarks.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :-------------------------------------------------
| **Section 0: Preparation & Test Infrastructure**
| 0.A | **Verify Phase 2 Completion**                      | `[ ]` | **Why:** To ensure the dispatcher is working before testing equivalence. <br> **How:** Confirm that `get_image_patches` correctly routes between implementations. Run existing tests with both flag states. Check that logging shows correct implementation selection. <br> **Verify:** Dispatcher works with both `use_batched_patch_extraction=True/False`
| 0.B | **Create Test File Structure**                     | `[ ]` | **Why:** To establish a dedicated test module for equivalence testing. <br> **How:** Create new file `tests/test_patch_extraction_equivalence.py`. Add imports: `unittest`, `numpy as np`, `tensorflow as tf`, `time`, `tracemalloc` (for memory profiling), `parameterized` (if available) or plan for manual parameterization. <br> **File:** `tests/test_patch_extraction_equivalence.py`
| **Section 1: Basic Equivalence Tests**
| 1.A | **Create Test Data Generator**                     | `[ ]` | **Why:** To have consistent, reproducible test data for all tests. <br> **How:** Create helper function `_generate_test_data(obj_size, N, gridsize, batch_size, dtype=np.complex64)` that generates: <br>1. Random complex object with known pattern <br>2. Random scan coordinates within valid bounds <br>3. Proper offset calculations <br>Return dict with all test inputs. <br> **File:** `tests/test_patch_extraction_equivalence.py`
| 1.B | **Implement Basic Equivalence Test**               | `[ ]` | **Why:** To verify both implementations produce identical output for standard cases. <br> **How:** Create `test_basic_equivalence` that: <br>1. Generates test data (e.g., 200x200 object, N=64, gridsize=2, batch=10) <br>2. Calls both `_get_image_patches_iterative` and `_get_image_patches_batched` <br>3. Uses `np.testing.assert_allclose(result1, result2, rtol=1e-6, atol=1e-6)` <br>4. Also checks dtype preservation <br> **File:** `tests/test_patch_extraction_equivalence.py`
| **Section 2: Parameterized Configuration Tests**
| 2.A | **Test Various N Values**                          | `[ ]` | **Why:** To ensure equivalence across different patch sizes. <br> **How:** Create `test_equivalence_various_N` with parameterized tests for N in [16, 32, 64, 128, 256]. For each N: <br>1. Generate appropriate test data <br>2. Run both implementations <br>3. Assert numerical equivalence with 1e-6 tolerance <br> **File:** `tests/test_patch_extraction_equivalence.py`
| 2.B | **Test Various Gridsize Values**                   | `[ ]` | **Why:** To verify correct handling of different channel configurations. <br> **How:** Create `test_equivalence_various_gridsize` for gridsize in [1, 2, 3, 4, 5]. Ensure: <br>1. Correct channel dimension c = gridsize² <br>2. Output shape is (B, N, N, c) for both implementations <br>3. Numerical equivalence maintained <br> **File:** `tests/test_patch_extraction_equivalence.py`
| 2.C | **Test Various Batch Sizes**                       | `[ ]` | **Why:** To ensure batching logic works correctly at different scales. <br> **How:** Create `test_equivalence_various_batch_sizes` for batch_size in [1, 10, 100, 500, 1000]. Include edge case of batch_size=1 (single image). Verify memory efficiency improves with larger batches. <br> **File:** `tests/test_patch_extraction_equivalence.py`
| **Section 3: Edge Case Tests**
| 3.A | **Test Border Coordinates**                        | `[ ]` | **Why:** To verify correct handling when patches are near image boundaries. <br> **How:** Create `test_equivalence_border_cases` that: <br>1. Places scan positions at image corners and edges <br>2. Tests maximum valid offsets <br>3. Ensures padding is handled identically <br>Note: Both implementations should handle padding via the same `hh.pad` call <br> **File:** `tests/test_patch_extraction_equivalence.py`
| 3.B | **Test Empty/Degenerate Cases**                    | `[ ]` | **Why:** To ensure robust handling of edge cases. <br> **How:** Create `test_equivalence_edge_cases` that tests: <br>1. Batch size of 0 (if supported) <br>2. Minimum object size <br>3. All scan positions at same location <br>4. Extreme offset values (within valid range) <br> **File:** `tests/test_patch_extraction_equivalence.py`
| 3.C | **Test Different Data Types**                      | `[ ]` | **Why:** To verify both implementations handle different precisions correctly. <br> **How:** Create `test_equivalence_dtypes` for dtypes [np.complex64, np.complex128]. Ensure: <br>1. Output dtype matches input dtype <br>2. Numerical precision is appropriate for each dtype <br>3. No unnecessary type conversions occur <br> **File:** `tests/test_patch_extraction_equivalence.py`
| **Section 4: Performance Benchmarking**
| 4.A | **Create Performance Benchmark Function**          | `[ ]` | **Why:** To measure and validate the performance improvement hypothesis. <br> **How:** Create `benchmark_performance` function that: <br>1. Accepts config parameters (N, gridsize, batch_size) <br>2. Runs warmup iterations (5x) to stabilize TF <br>3. Times multiple runs (10x) and calculates mean/std <br>4. Returns timing dict for both implementations <br> **File:** `tests/test_patch_extraction_equivalence.py`
| 4.B | **Implement Main Performance Test**                | `[ ]` | **Why:** To verify the 10x performance improvement claim. <br> **How:** Create `test_performance_improvement` that: <br>1. Uses realistic parameters (N=64, gridsize=2, batch=5000) <br>2. Runs benchmark_performance <br>3. Calculates speedup factor <br>4. Asserts speedup >= 10x <br>5. Logs detailed timing results <br> **File:** `tests/test_patch_extraction_equivalence.py`
| 4.C | **Add Memory Profiling Test**                      | `[ ]` | **Why:** To verify memory efficiency claims (< 20% increase). <br> **How:** Create `test_memory_usage` using `tracemalloc`: <br>1. Profile peak memory for both implementations <br>2. Calculate memory ratio <br>3. Assert batched uses < 1.2x memory of iterative <br>4. Log memory usage details <br>Consider testing with large batch sizes. <br> **File:** `tests/test_patch_extraction_equivalence.py`
| **Section 5: Integration & Reporting**
| 5.A | **Create Performance Report Generator**            | `[ ]` | **Why:** To document performance characteristics for future reference. <br> **How:** Create `generate_performance_report` that: <br>1. Runs benchmarks across multiple configurations <br>2. Creates a formatted report with speedup factors <br>3. Includes memory usage comparisons <br>4. Saves to `test_outputs/patch_extraction_performance.txt` <br> **File:** `tests/test_patch_extraction_equivalence.py`
| 5.B | **Add Numerical Difference Analysis**              | `[ ]` | **Why:** To understand the magnitude of any numerical differences. <br> **How:** In equivalence tests, when differences exist: <br>1. Calculate max absolute difference <br>2. Calculate relative differences <br>3. Log statistics about differences <br>4. Ensure all are within acceptable tolerance <br> **File:** `tests/test_patch_extraction_equivalence.py`
| **Section 6: Finalization**
| 6.A | **Run Complete Test Suite**                        | `[ ]` | **Why:** To ensure all tests pass and meet success criteria. <br> **How:** Run `python -m pytest tests/test_patch_extraction_equivalence.py -v`. Verify: <br>1. All equivalence tests pass with atol=1e-6 <br>2. Performance improvement >= 10x <br>3. Memory usage < 1.2x <br>Generate and review performance report. <br> **Command:** `python -m pytest tests/test_patch_extraction_equivalence.py -v`
| 6.B | **Document Test Coverage**                         | `[ ]` | **Why:** To ensure comprehensive test coverage for confidence in production use. <br> **How:** Add module docstring to test file documenting: <br>1. Test coverage matrix (N, gridsize, batch combinations) <br>2. Performance benchmarking methodology <br>3. Tolerance justification (why 1e-6) <br>4. How to run performance reports <br> **File:** `tests/test_patch_extraction_equivalence.py`
| 6.C | **Commit Phase 3 Changes**                         | `[ ]` | **Why:** To checkpoint the comprehensive test suite. <br> **How:** Stage all changes, create commit with message: "Phase 3: Add comprehensive equivalence testing and performance benchmarks". Include performance report in commit message. Push to feature branch. <br> **Command:** `git add -A && git commit -m "Phase 3: Add comprehensive equivalence testing and performance benchmarks" && git push`

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: All equivalence tests pass with tolerance atol=1e-6, and performance improvement of at least 10x is demonstrated.
3. Memory usage increase is less than 20% compared to iterative implementation.
4. A comprehensive performance report has been generated documenting the improvements.