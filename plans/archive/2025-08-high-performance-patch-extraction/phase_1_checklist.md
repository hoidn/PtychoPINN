# Phase 1: Core Batched Implementation Checklist

**Initiative:** High-Performance Patch Extraction Refactoring
**Created:** 2025-08-03
**Phase Goal:** To implement the new batched patch extraction function that replaces the for loop with a single batched translation call.
**Deliverable:** A new `_get_image_patches_batched` function in `ptycho/raw_data.py` with basic unit tests demonstrating functionality.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :-------------------------------------------------
| **Section 0: Preparation & Context Priming**
| 0.A | **Review Key Documents & APIs**                    | `[ ]` | **Why:** To understand the current implementation and available tools before refactoring. <br> **Docs:** `docs/DEVELOPER_GUIDE.md` (Section 4.4 on tensor formats), `ptycho/CLAUDE.md` (tf_helper module overview). <br> **APIs:** Review `ptycho.tf_helper.translate`, `ptycho.tf_helper._channel_to_flat`, `ptycho.tf_helper.pad` functions.
| 0.B | **Analyze Current Implementation**                 | `[ ]` | **Why:** To fully understand the logic being replaced and ensure no edge cases are missed. <br> **How:** Study `get_image_patches` function in `ptycho/raw_data.py` (lines 685-692), focusing on the for loop logic, offset calculations, and tensor manipulations. <br> **File:** `ptycho/raw_data.py`
| 0.C | **Set Up Test Environment**                        | `[ ]` | **Why:** To ensure you can run and test changes immediately. <br> **How:** Verify you can run existing tests with `python -m pytest ptycho/test_raw_data.py -v` (if it exists) or `python -m unittest discover -s tests -p "test_*.py"`. Create `tests/test_raw_data.py` if it doesn't exist.
| **Section 1: Extract Current Implementation**
| 1.A | **Create `_get_image_patches_iterative` Function** | `[ ]` | **Why:** To preserve the current implementation for comparison and fallback. <br> **How:** Copy the existing loop logic (lines 688-692) into a new function `_get_image_patches_iterative(gt_padded, offsets_f, N, B, c)`. Keep the exact same logic including the numpy array conversion. <br> **File:** `ptycho/raw_data.py`
| 1.B | **Add Function Signature & Docstring**             | `[ ]` | **Why:** To document the legacy implementation clearly. <br> **How:** Add proper function signature with type hints and a docstring explaining this is the legacy iterative implementation. Include parameters: `gt_padded` (padded image tensor), `offsets_f` (flat offsets), `N` (patch size), `B` (batch size), `c` (channels). <br> **File:** `ptycho/raw_data.py`
| **Section 2: Implement Batched Version**
| 2.A | **Create `_get_image_patches_batched` Function**   | `[ ]` | **Why:** This is the core performance improvement - replacing the loop with batched operations. <br> **How:** Create new function with same signature as iterative version. Key insight: create a batched tensor by repeating `gt_padded` B*c times, then call `hh.translate` once with all offsets. <br> **File:** `ptycho/raw_data.py`
| 2.B | **Implement Batched Translation Logic**            | `[ ]` | **Why:** To leverage the existing high-performance translation engine. <br> **How:** <br>1. Create batched input: `gt_padded_batch = tf.repeat(gt_padded, B * c, axis=0)` <br>2. Negate offsets: `negated_offsets = -offsets_f[:, :, :, 0]` <br>3. Single translate call: `translated_patches = hh.translate(gt_padded_batch, negated_offsets)` <br>4. Slice to N×N: `patches_flat = translated_patches[:, :N, :N, :]` <br> **File:** `ptycho/raw_data.py`
| 2.C | **Reshape Output to Channel Format**               | `[ ]` | **Why:** To match the expected output format of the original function. <br> **How:** Use `tf.reshape` to convert from flat format `(B*c, N, N, 1)` to channel format `(B, N, N, c)`: `patches_channel = tf.reshape(patches_flat, (B, N, N, c))`. Return as tensor, not numpy array. <br> **File:** `ptycho/raw_data.py`
| **Section 3: Add Configuration Parameter**
| 3.A | **Update ModelConfig Dataclass**                   | `[ ]` | **Why:** To enable feature flag control through the modern configuration system. <br> **How:** Add `use_batched_patch_extraction: bool = False` to the `ModelConfig` dataclass. Place it near other performance-related flags if any exist. <br> **File:** `ptycho/config/config.py`
| 3.B | **Add Legacy Config Mapping**                      | `[ ]` | **Why:** To ensure the configuration flows to modules still using the legacy params system. <br> **How:** In the config-to-params mapping logic (likely in `update_legacy_dict` or similar), add mapping for the new parameter: `params['use_batched_patch_extraction'] = config.model.use_batched_patch_extraction`. <br> **File:** `ptycho/config/config.py` or relevant config mapping location
| **Section 4: Basic Unit Tests**
| 4.A | **Create Test File Structure**                     | `[ ]` | **Why:** To establish proper test organization following project conventions. <br> **How:** Create `tests/test_raw_data.py` if it doesn't exist. Add imports: `unittest`, `numpy as np`, `tensorflow as tf`, and `from ptycho.raw_data import _get_image_patches_iterative, _get_image_patches_batched`. <br> **File:** `tests/test_raw_data.py`
| 4.B | **Write Basic Functionality Test**                 | `[ ]` | **Why:** To verify the batched implementation produces valid output. <br> **How:** Create test case that: <br>1. Creates a simple test image (e.g., 100×100 with known pattern) <br>2. Defines test offsets for a 2×2 grid <br>3. Calls `_get_image_patches_batched` <br>4. Verifies output shape is correct `(B, N, N, c)` <br>5. Checks output is valid (no NaN/Inf values) <br> **File:** `tests/test_raw_data.py`
| 4.C | **Write Shape Validation Test**                    | `[ ]` | **Why:** To ensure the function handles different input configurations correctly. <br> **How:** Create parameterized test with various combinations: <br>- Different N values (32, 64, 128) <br>- Different gridsize values (1, 2, 3) <br>- Different batch sizes (1, 10, 100) <br>Verify output shape matches expected `(B, N, N, gridsize**2)` for each. <br> **File:** `tests/test_raw_data.py`
| 4.D | **Run Tests & Fix Issues**                         | `[ ]` | **Why:** To ensure the implementation is working before moving to the next phase. <br> **How:** Run `python -m pytest tests/test_raw_data.py -v`. Debug any failures. Common issues might include tensor shape mismatches, dtype inconsistencies, or incorrect indexing in the reshape operation. <br> **Verify:** All tests pass
| **Section 5: Finalization**
| 5.A | **Code Formatting & Cleanup**                      | `[ ]` | **Why:** To maintain code quality standards. <br> **How:** Remove any debug print statements, ensure consistent indentation (4 spaces), add appropriate spacing between functions. Run any project linters if configured. <br> **File:** `ptycho/raw_data.py`
| 5.B | **Add Comprehensive Docstrings**                   | `[ ]` | **Why:** To document the new implementation for future developers. <br> **How:** Write detailed docstrings for both `_get_image_patches_iterative` and `_get_image_patches_batched` explaining: purpose, algorithm, parameters, return values, and key differences between them. <br> **File:** `ptycho/raw_data.py`
| 5.C | **Commit Phase 1 Changes**                         | `[ ]` | **Why:** To create a clean checkpoint before moving to Phase 2. <br> **How:** Stage all changes, create commit with message: "Phase 1: Implement core batched patch extraction function with tests". Push to feature branch. <br> **Command:** `git add -A && git commit -m "Phase 1: Implement core batched patch extraction function with tests" && git push`

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: `python -m pytest tests/test_raw_data.py -v` completes with all tests passing.
3. No regressions are introduced in the existing test suite.
4. The new `_get_image_patches_batched` function exists and produces valid output for basic test cases.