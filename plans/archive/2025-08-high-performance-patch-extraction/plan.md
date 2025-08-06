# R&D Plan: High-Performance Patch Extraction Refactoring

*Created: 2025-08-03*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** High-Performance Patch Extraction Refactoring

**Problem Statement:** The core data preparation function `ptycho.raw_data.get_image_patches` uses a slow, iterative for loop to extract patches one by one. This creates a significant performance bottleneck, especially for large datasets with high gridsize values (e.g., 5000 images * 4 patches/group = 20,000 sequential operations), making data preparation prohibitively slow. The codebase already contains a superior, high-performance, XLA-compatible translation engine (`translate_xla`) that is not being leveraged for this critical task.

**Proposed Solution / Hypothesis:**
- **Solution:** We will refactor `ptycho.raw_data.get_image_patches` to replace the inefficient for loop with a single, batched call to the existing `ptycho.tf_helper.translate` function (which is powered by the memory-efficient `translate_xla` engine). The new implementation will be introduced alongside the old one via a feature flag to allow for rigorous validation before the final switchover.
- **Hypothesis:** By leveraging the existing batched translation engine, we will achieve a significant performance speedup (estimated 10-100x) in the patch extraction step, reduce code complexity, and improve GPU utilization, all while maintaining bit-for-bit identical output to the legacy implementation.

---

## 🛠️ **METHODOLOGY / SOLUTION APPROACH**

This initiative will follow a safe, validation-first refactoring pattern.

1. **Introduce New Implementation:** A new function, `_get_image_patches_batched`, will be created within `raw_data.py`. This function will implement the efficient "translate-then-slice" pattern using `tf_helper.translate`. It will not use a for loop.

2. **Feature Flag Integration:** The existing `get_image_patches` function will be modified to act as a dispatcher. It will use a new configuration parameter (`use_batched_patch_extraction`, default `False`) to decide whether to call the old `_get_image_patches_iterative` (the current loop-based code, which will be moved to its own function) or the new `_get_image_patches_batched`.

3. **Rigorous Equivalence Testing:** A new, dedicated test suite (`tests/test_patch_extraction_equivalence.py`) will be created. This test will be the cornerstone of the validation, running both the iterative and batched implementations with the same inputs and asserting that their outputs are numerically identical to within a very tight tolerance.

4. **Performance Benchmarking:** The test suite will also include a performance benchmark to quantify the speedup and validate the core hypothesis of the initiative.

5. **Deprecation and Cleanup:** Once validation and performance tests pass, the feature flag will be switched to `True` by default. After a stabilization period, the old iterative code and the feature flag will be removed, completing the refactoring.

---

## 🎯 **DELIVERABLES**

1. **Refactored `ptycho/raw_data.py`:** An updated module containing both the new batched and old iterative patch extraction implementations, controlled by a feature flag.

2. **New Equivalence Test Suite:** A new test file, `tests/test_patch_extraction_equivalence.py`, that rigorously validates the correctness and performance of the new implementation against the old one.

3. **Updated Documentation:** Docstrings and relevant guides (e.g., `DEVELOPER_GUIDE.md`) will be updated to reflect the new, high-performance implementation.

4. **Final Cleanup Commit:** A final commit that removes the legacy iterative code and the feature flag after the new implementation is validated and enabled by default.

---

## ✅ **VALIDATION & VERIFICATION PLAN**

This initiative's success is entirely dependent on proving the new implementation is both faster and functionally identical to the old one.

### **Equivalence Testing** (`tests/test_patch_extraction_equivalence.py`):
The core of the validation will be a test class that performs the following for various configurations (different N, gridsize, batch sizes, and dtypes):

1. **Generate Identical Inputs:** Create a source object and a set of random scan coordinates.
2. **Run Both Implementations:**
   - Execute the legacy iterative function (`_get_image_patches_iterative`) with the inputs.
   - Execute the new batched function (`_get_image_patches_batched`) with the exact same inputs.
3. **Assert Numerical Equivalence:**
   - Use `np.testing.assert_allclose` to compare the two output tensors.
   - The tolerance (`rtol`, `atol`) must be set to a very low value (e.g., 1e-6) to ensure they are functionally identical.
4. **Test Edge Cases:** Include tests for coordinates near the image borders to validate padding and boundary handling.

### **Performance Benchmarking:**
- A dedicated test within the new test suite will measure the execution time of both implementations on a realistic, large-scale task (e.g., 5000 images, gridsize=2).
- The test will report the wall-clock time for each and calculate the speedup factor.
- Memory usage profiling will be included to ensure the batched approach doesn't introduce memory regressions.

### **Success Criteria:**
- **Correctness:** All numerical equivalence tests in the new test suite must pass with a tolerance of `atol=1e-6`.
- **Performance:** The new batched implementation must demonstrate at least a 10x performance improvement over the legacy iterative implementation in the benchmark test.
- **Memory:** Peak memory usage should not increase by more than 20% compared to the iterative approach.
- **No Regressions:** All existing tests in the project's test suite must continue to pass.
- **Completeness:** The feature flag is successfully integrated and allows for a safe transition. The final cleanup removes all legacy code.

---

## 🚀 **RISK MITIGATION**

**Risk:** The new batched implementation has subtle differences in boundary handling or interpolation, leading to numerical divergence from the original.
- **Mitigation:** The rigorous equivalence test suite is designed specifically to catch this. The feature flag provides a safe rollback mechanism if an issue is discovered in production.

**Risk:** The batched implementation consumes significantly more GPU memory, breaking existing workflows.
- **Mitigation:** The chosen `translate_xla` engine is memory-efficient (it does not tile the source image). Memory usage will be monitored during testing. The feature flag allows users with memory-constrained systems to temporarily revert to the slower, less memory-intensive iterative method if needed.

**Risk:** The refactoring introduces a bug that is not caught by the tests.
- **Mitigation:** The phased rollout (introduce new code -> test -> enable by default -> remove old code) minimizes the impact window. The ability to toggle the implementation with a single flag provides a powerful debugging tool.

**Risk:** Different floating-point accumulation order might cause slight numerical differences.
- **Mitigation:** The 1e-6 tolerance in tests should catch meaningful differences while allowing for minor floating-point reordering effects.

---

## 📁 **File Organization**

**Initiative Path:** `plans/active/high-performance-patch-extraction/`

**Next Step:** Run `/implementation` to generate the phased implementation plan.