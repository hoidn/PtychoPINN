# Phase 2: Feature Flag Integration & Dispatcher Checklist

**Initiative:** High-Performance Patch Extraction Refactoring
**Created:** 2025-08-03
**Phase Goal:** To integrate the feature flag system that allows safe switching between implementations.
**Deliverable:** Modified `get_image_patches` function that acts as a dispatcher between old and new implementations based on configuration.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :-------------------------------------------------
| **Section 0: Preparation & Verification**
| 0.A | **Verify Phase 1 Completion**                      | `[ ]` | **Why:** To ensure all Phase 1 deliverables are in place before building on them. <br> **How:** Confirm that `_get_image_patches_iterative` and `_get_image_patches_batched` exist in `ptycho/raw_data.py`. Run `python -m pytest tests/test_raw_data.py -v` to verify tests pass. <br> **Verify:** Both functions exist and basic tests pass
| 0.B | **Review Configuration Flow**                      | `[ ]` | **Why:** To understand how configuration parameters flow through the system. <br> **How:** Trace the path from `ModelConfig` to `params.cfg`. Review `ptycho/config/config.py` and find where `update_legacy_dict` or similar mapping occurs. <br> **File:** `ptycho/config/config.py`
| **Section 1: Modify Dispatcher Function**
| 1.A | **Refactor `get_image_patches` as Dispatcher**     | `[ ]` | **Why:** To enable runtime selection between implementations based on configuration. <br> **How:** Modify the existing `get_image_patches` function to: <br>1. Keep the same signature and docstring <br>2. Add implementation selection logic before the current code <br>3. Move all current implementation details into the conditional branches <br> **File:** `ptycho/raw_data.py`
| 1.B | **Add Configuration Check Logic**                  | `[ ]` | **Why:** To determine which implementation to use based on the feature flag. <br> **How:** Add logic to check configuration in priority order: <br>```python<br>if config and hasattr(config.model, 'use_batched_patch_extraction'):<br>    use_batched = config.model.use_batched_patch_extraction<br>else:<br>    use_batched = params.get('use_batched_patch_extraction', False)<br>``` <br> **File:** `ptycho/raw_data.py`
| 1.C | **Implement Routing Logic**                        | `[ ]` | **Why:** To call the appropriate implementation based on the flag. <br> **How:** After the padding step, add conditional routing: <br>```python<br>if use_batched:<br>    return _get_image_patches_batched(gt_padded, offsets_f, N, B, c)<br>else:<br>    return _get_image_patches_iterative(gt_padded, offsets_f, N, B, c)<br>``` <br> **File:** `ptycho/raw_data.py`
| **Section 2: Add Logging & Monitoring**
| 2.A | **Import Logging Module**                          | `[ ]` | **Why:** To provide visibility into which implementation is being used. <br> **How:** Add `import logging` at the top of the file if not already present. Get logger: `logger = logging.getLogger(__name__)`. <br> **File:** `ptycho/raw_data.py`
| 2.B | **Add Implementation Selection Logging**           | `[ ]` | **Why:** To help debug and monitor which code path is being used in production. <br> **How:** Add info-level logging after the configuration check: <br>```python<br>logger.info(f"Using {'batched' if use_batched else 'iterative'} patch extraction implementation")<br>``` <br>Also log key parameters: N, gridsize, B for debugging. <br> **File:** `ptycho/raw_data.py`
| **Section 3: Configuration Flow Testing**
| 3.A | **Verify Legacy Config Mapping**                   | `[ ]` | **Why:** To ensure the feature flag works with the legacy params system. <br> **How:** In the config mapping logic, verify that `use_batched_patch_extraction` is properly mapped from ModelConfig to params.cfg. Add the mapping if missing. Test by setting the value and checking `params.get('use_batched_patch_extraction')`. <br> **File:** Configuration mapping location (likely in `ptycho/config/`)
| 3.B | **Create Dispatcher Test**                         | `[ ]` | **Why:** To verify the dispatcher correctly routes based on configuration. <br> **How:** Create test class `TestPatchExtractionDispatcher` in `tests/test_raw_data.py` with two test methods: <br>1. `test_dispatcher_uses_iterative_by_default` - Verify default behavior <br>2. `test_dispatcher_uses_batched_when_enabled` - Mock/set config and verify batched path <br> **File:** `tests/test_raw_data.py`
| **Section 4: Integration Testing**
| 4.A | **Test with Iterative Implementation**             | `[ ]` | **Why:** To ensure the dispatcher doesn't break existing functionality. <br> **How:** Create integration test that: <br>1. Explicitly sets `use_batched_patch_extraction=False` <br>2. Calls `get_image_patches` with real data <br>3. Verifies output matches expected format <br>4. Checks that iterative path was taken (via logs or mocking) <br> **File:** `tests/test_raw_data.py`
| 4.B | **Test with Batched Implementation**               | `[ ]` | **Why:** To verify the new path works through the dispatcher. <br> **How:** Create integration test that: <br>1. Sets `use_batched_patch_extraction=True` <br>2. Calls `get_image_patches` with same test data <br>3. Verifies output format is identical <br>4. Confirms batched path was taken <br> **File:** `tests/test_raw_data.py`
| 4.C | **Test Configuration Priority**                    | `[ ]` | **Why:** To ensure the config object takes precedence over legacy params. <br> **How:** Create test that: <br>1. Sets different values in config object vs params.cfg <br>2. Verifies config object value is used <br>3. Tests behavior when config object is None <br> **File:** `tests/test_raw_data.py`
| **Section 5: Documentation & Finalization**
| 5.A | **Update Function Docstring**                      | `[ ]` | **Why:** To document the new dispatcher behavior. <br> **How:** Update `get_image_patches` docstring to mention: <br>1. It now acts as a dispatcher <br>2. The `use_batched_patch_extraction` configuration parameter <br>3. Default behavior (iterative) <br> **File:** `ptycho/raw_data.py`
| 5.B | **Run Full Test Suite**                            | `[ ]` | **Why:** To ensure no regressions were introduced. <br> **How:** Run complete test suite: `python -m unittest discover -s tests -p "test_*.py"`. Fix any failures. Pay special attention to any tests that use `get_image_patches`. <br> **Verify:** All tests pass
| 5.C | **Commit Phase 2 Changes**                         | `[ ]` | **Why:** To create a clean checkpoint with the feature flag system complete. <br> **How:** Stage all changes, create commit with message: "Phase 2: Add feature flag dispatcher for patch extraction implementations". Push to feature branch. <br> **Command:** `git add -A && git commit -m "Phase 2: Add feature flag dispatcher for patch extraction implementations" && git push`

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: The `get_image_patches` function correctly routes to either implementation based on configuration setting.
3. All tests pass with both `use_batched_patch_extraction=True` and `use_batched_patch_extraction=False`.
4. Logging confirms which implementation is being used at runtime.