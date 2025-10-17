# Phase 2: Parameter Interface Enhancement Checklist

**Created:** 2025-07-19
**Phase Goal:** To enhance the training script interface to intelligently interpret the `--n-images` parameter based on `gridsize`, simplifying the user experience while maintaining clear documentation of the behavior.
**Deliverable:** A modified `scripts/training/train.py` that accepts a single `--n-images` parameter and automatically adapts its behavior based on `gridsize`, with clear logging and documentation.

## 📊 Progress Tracking

**Tasks Completed:** 0 / 18
**Current Status:** 🔴 Not Started
**Started:** -
**Last Updated:** -

## ✅ Task List

### Instructions for Working Through Tasks:
1. Read the complete task description including the How/Why & API Guidance
2. Update task state: `[ ]` → `[~]` (in progress) → `[x]` (complete)
3. Follow the implementation guidance carefully
4. Test each task before marking complete

---

| Task | Description | State | How/Why & API Guidance |
|:-----|:------------|:------|:-----------------------|
| **Section 0: Preparation & Context** |
| 0.1 | **Review Phase Requirements** | `[ ]` | **Why:** Load context before coding to avoid rework.<br>**How:** Read `plan.md` sections on deliverables and success criteria. Read this phase's section in `implementation.md`.<br>**Docs:** Pay attention to "Technical Implementation Details" section.<br>**Output:** You should be able to explain the phase goal and success criteria. |
| 0.2 | **Set Up Development Branch** | `[ ]` | **Why:** Isolate changes for clean Git history.<br>**How:** `git checkout -b feature/grouping-aware-subsampling-phase-2`<br>**Verify:** `git branch` shows new branch with `*` indicator. |
| 0.3 | **Verify Prerequisites** | `[ ]` | **Why:** Ensure Phase 1 outputs are available.<br>**How:** Check that the caching functionality in `ptycho/raw_data.py` works correctly.<br>**API:** Test with `python -c "from ptycho.raw_data import RawData; print('Phase 1 ready')"` |
| **Section 1: Training Script Analysis** |
| 1.1 | **Analyze current training script interface** | `[ ]` | **Why:** Understand current parameter handling before modification.<br>**File:** `scripts/training/train.py`<br>**How:** Study the current argument parsing and data loading logic.<br>**Focus:** Look for how `--n-images` is currently used and where `gridsize` affects behavior.<br>**Output:** Clear understanding of current parameter flow. |
| 1.2 | **Identify parameter interpretation logic location** | `[ ]` | **Why:** Find where to implement intelligent parameter interpretation.<br>**Files:** `scripts/training/train.py`, `ptycho/workflows/components.py`<br>**How:** Trace the flow from command-line argument to data loading call.<br>**Key:** Locate where `n_images` parameter is passed to data loading functions. |
| **Section 2: Intelligent Parameter Interpretation** |
| 2.1 | **Implement parameter interpretation function** | `[ ]` | **Why:** Create centralized logic for `--n-images` interpretation.<br>**File:** `scripts/training/train.py`<br>**Function:** `def interpret_n_images_parameter(n_images: int, gridsize: int) -> tuple[int, str]:`<br>**API Pattern:**<br>```python<br>def interpret_n_images_parameter(n_images: int, gridsize: int) -> tuple[int, str]:<br>    """Interpret --n-images based on gridsize.<br>    <br>    Args:<br>        n_images: User-specified number<br>        gridsize: Current gridsize setting<br>    Returns:<br>        tuple: (actual_n_images, interpretation_message)<br>    """<br>    if gridsize == 1:<br>        return n_images, f"Using {n_images} individual images (gridsize=1)"<br>    else:<br>        total_patterns = n_images * gridsize * gridsize<br>        return n_images, f"Using {n_images} groups ({total_patterns} total patterns, gridsize={gridsize})"<br>```<br>**Test:** Call with various gridsize values. |
| 2.2 | **Add parameter interpretation logging** | `[ ]` | **Why:** Inform users about parameter interpretation behavior.<br>**How:** Add clear log messages that explain whether N refers to individual images or groups.<br>**Log levels:** Use INFO level for user-facing messages.<br>**Messages:**<br>- "Parameter interpretation: --n-images={N} refers to individual images (gridsize=1)"<br>- "Parameter interpretation: --n-images={N} refers to neighbor groups (gridsize={G}, total patterns={T})" |
| 2.3 | **Integrate interpretation into argument parsing** | `[ ]` | **Why:** Apply intelligent interpretation when parsing command-line arguments.<br>**File:** `scripts/training/train.py`<br>**How:** Call interpretation function after parsing args but before data loading.<br>**Integration point:** After `args = parser.parse_args()` but before creating config objects.<br>**Store result:** Update args or create derived variables for clear data flow. |
| **Section 3: Data Loading Integration** |
| 3.1 | **Update load_data calls with interpreted parameters** | `[ ]` | **Why:** Ensure data loading receives correctly interpreted parameters.<br>**File:** `scripts/training/train.py`<br>**How:** Pass the interpreted n_images value to data loading functions.<br>**Check:** Verify that `ptycho/workflows/components.py` `load_data` function receives correct values.<br>**Validation:** Ensure no legacy sequential slicing remains for gridsize > 1. |
| 3.2 | **Remove legacy sequential slicing logic** | `[ ]` | **Why:** Eliminate old sequential selection that created spatial bias.<br>**File:** `ptycho/workflows/components.py`<br>**How:** Check `load_data` function for any remaining `[:n_images]` slicing operations.<br>**Replace:** Ensure all gridsize > 1 cases use the new group-first sampling.<br>**Verify:** No sequential slicing exists outside of the backward-compatible gridsize=1 path. |
| 3.3 | **Add parameter validation** | `[ ]` | **Why:** Catch invalid parameter combinations early.<br>**How:** Add validation logic for edge cases.<br>**Edge cases:**<br>1. `n_images <= 0`<br>2. `gridsize <= 0`<br>3. `n_images > available_groups` (for gridsize > 1)<br>**Error handling:** Provide clear error messages for invalid combinations. |
| **Section 4: Configuration System Integration** |
| 4.1 | **Update configuration dataclasses** | `[ ]` | **Why:** Ensure modern config system supports intelligent parameter interpretation.<br>**File:** `ptycho/config/config.py`<br>**Check:** Verify that `TrainingConfig` and related classes handle the new parameter flow.<br>**API:** Ensure config objects can store both raw and interpreted parameter values.<br>**Backwards compatibility:** Maintain compatibility with existing config files. |
| 4.2 | **Update legacy parameter dictionary** | `[ ]` | **Why:** Maintain compatibility with legacy code that uses `params.cfg`.<br>**How:** Ensure the legacy dictionary receives correctly interpreted values.<br>**File:** `scripts/training/train.py`<br>**Flow:** Modern config → interpreted parameters → legacy dict update<br>**Verify:** Legacy modules that use `params.get('n_images')` receive correct values. |
| **Section 5: Testing & Validation** |
| 5.1 | **Test gridsize=1 parameter handling** | `[ ]` | **Why:** Verify backward compatibility for traditional workflows.<br>**Test cases:**<br>1. `--n-images=100 --gridsize=1` → should use 100 individual images<br>2. Check that logging correctly identifies individual image interpretation<br>3. Verify no caching or group discovery occurs<br>**Command:** Create test script that exercises these scenarios. |
| 5.2 | **Test gridsize>1 parameter handling** | `[ ]` | **Why:** Verify new group-based interpretation works correctly.<br>**Test cases:**<br>1. `--n-images=50 --gridsize=2` → should use 50 groups (200 total patterns)<br>2. `--n-images=25 --gridsize=3` → should use 25 groups (225 total patterns)<br>3. Check that logging correctly explains group interpretation<br>**Expected:** Grouping-aware subsampling from cached groups. |
| 5.3 | **Test parameter edge cases** | `[ ]` | **Why:** Ensure robust error handling for invalid parameters.<br>**Edge cases:**<br>1. `--n-images=0`<br>2. `--n-images=-5`<br>3. Requesting more groups than available<br>**Expected:** Clear error messages, graceful degradation where appropriate. |
| 5.4 | **Integration test with full training pipeline** | `[ ]` | **Why:** Ensure the complete workflow works end-to-end.<br>**How:** Run a short training session with both gridsize=1 and gridsize>1.<br>**Command:** Use test dataset and minimal epochs to verify complete pipeline.<br>**Verify:** Training completes successfully, parameter interpretation is logged correctly. |
| **Section 6: Code Quality & Documentation** |
| 6.1 | **Format and lint code** | `[ ]` | **Why:** Maintain consistent code style.<br>**Commands:**<br>`black scripts/training/train.py`<br>`ruff check scripts/training/train.py`<br>`mypy scripts/training/ --ignore-missing-imports`<br>**Fix:** Address any warnings before proceeding. |
| 6.2 | **Update function docstrings** | `[ ]` | **Why:** Document new parameter interpretation behavior.<br>**Files:** All modified functions in `scripts/training/train.py`<br>**Content:** Explain intelligent parameter interpretation and gridsize-dependent behavior.<br>**Format:** Follow existing docstring conventions in the codebase. |
| **Section 7: Phase Completion** |
| 7.1 | **Run success test** | `[ ]` | **Test:** `ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images=512 --gridsize=2 --nepochs=1 --output_dir test_phase2`<br>**Expected:** Training with `gridsize=2` successfully samples 512 groups (2048 total patterns) and logs this behavior clearly to the user<br>**Verify:** Check log output contains parameter interpretation message<br>**Debug:** If fails, check parameter flow from command line to data loading. |
| 7.2 | **Commit changes** | `[ ]` | **Why:** Create atomic commit for this phase.<br>**Commands:**<br>`git add -A`<br>`git commit -m "[Phase 2] Enhanced training script with intelligent --n-images parameter interpretation"`<br>`git push origin feature/grouping-aware-subsampling-phase-2`<br>**PR Title:** `[Grouping-Aware Subsampling] Phase 2: Intelligent Parameter Interface` |

## 📝 Implementation Notes

*Document decisions and issues as you work:*

### Decisions Made:
- 

### Issues Encountered:
- 

### Performance Observations:
- 

## 🔍 Success Test Details

**Command:** `ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images=512 --gridsize=2 --nepochs=1 --output_dir test_phase2`
**Expected Result:** Training with `gridsize=2` successfully samples 512 groups (2048 total patterns) and logs this behavior clearly to the user
**Actual Result:** <fill this in after running>
**Screenshot/Log:** <paste relevant output>