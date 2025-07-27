# Phase 3: Module Consolidation Checklist

**Initiative:** Codebase Housekeeping
**Created:** 2025-07-22
**Phase Goal:** Consolidate specialized data loaders into a single, unified interface to reduce code duplication.
**Deliverable:** A unified data loading function in `ptycho/loader.py` that handles all previously supported data formats.
**Estimated Duration:** 1 day

## 📊 Progress Tracking

**Tasks:** 0 / 14 completed
**Status:** 🔴 Not Started → 🟡 In Progress → 🟢 Complete
**Started:** -
**Completed:** -
**Actual Duration:** -

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :------------------------------------------------- |
| **Section 0: Analysis & Preparation** |
| 0.1 | **Analyze loaders/als.py functionality** | `[ ]` | **Why:** Understand the ALS loader's interface and data handling before integration.<br>**File:** `loaders/als.py`<br>**Focus:** Study `load_single_object()` function - parameters, return format, data processing<br>**Document:** Note key differences from main loader pattern<br>**Output:** Clear understanding of ALS-specific data handling requirements |
| 0.2 | **Analyze ptycho/xpp.py functionality** | `[ ]` | **Why:** Understand the XPP loader's interface and how it uses main loader functions.<br>**File:** `ptycho/xpp.py`<br>**Focus:** Study the `get_data()` function and how it imports `load_xpp_npz`<br>**Document:** Note the data file path and any XPP-specific configurations<br>**Output:** Understanding of XPP integration pattern and hardcoded dataset usage |
| 0.3 | **Search for all usages of specialized loaders** | `[ ]` | **Why:** Find all import statements and function calls to ensure nothing is missed during consolidation.<br>**Commands:**<br>```bash<br>rg "from loaders\." --type py<br>rg "import.*als" --type py<br>rg "from.*xpp" --type py<br>rg "load_single_object" --type py<br>```<br>**Document:** List all files that import or use these specialized loaders<br>**Output:** Complete inventory of usage locations for update planning |
| **Section 1: Integration into Main Loader** |
| 1.1 | **Add ALS loader function to ptycho/loader.py** | `[ ]` | **Why:** Consolidate ALS functionality into the main loader module.<br>**File:** `ptycho/loader.py`<br>**Implementation:**<br>- Copy `load_single_object` function from `loaders/als.py`<br>- Rename to `load_als_data` for clarity<br>- Ensure it follows the same parameter and return patterns as other loader functions<br>- Add appropriate docstring following project conventions<br>**Dependencies:** Requires 0.1 (ALS analysis) completion |
| 1.2 | **Enhance XPP integration in main loader** | `[ ]` | **Why:** Improve the existing XPP functionality and make it more flexible.<br>**File:** `ptycho/loader.py`<br>**Implementation:**<br>- Review existing `load_xpp_npz` function<br>- Add configuration parameters to make it less hardcoded<br>- Ensure it can handle different XPP dataset files, not just the embedded one<br>- Update docstring to document enhanced flexibility<br>**Dependencies:** Requires 0.2 (XPP analysis) completion |
| 1.3 | **Add unified loader dispatch function** | `[ ]` | **Why:** Provide a single entry point that can auto-detect data format and use appropriate loader.<br>**File:** `ptycho/loader.py`<br>**Function:** `load_dataset_auto(file_path, format_hint=None, **kwargs)`<br>**Implementation:**<br>```python<br>def load_dataset_auto(file_path, format_hint=None, **kwargs):<br>    """Unified loader that auto-detects format or uses hint."""<br>    if format_hint == 'als':<br>        return load_als_data(file_path, **kwargs)<br>    elif format_hint == 'xpp':<br>        return load_xpp_npz(file_path, **kwargs)<br>    else:<br>        # Auto-detection logic or default to generic loader<br>        return load_ptycho_data(file_path, **kwargs)<br>```<br>**Dependencies:** Requires 1.1 and 1.2 (integration) completion |
| **Section 2: Update Import References** |
| 2.1 | **Update imports in files using ALS loader** | `[ ]` | **Why:** Redirect all imports to use the consolidated loader instead of specialized files.<br>**Implementation:** For each file found in 0.3 that imports from `loaders.als`:<br>- Change `from loaders.als import load_single_object` to `from ptycho.loader import load_als_data`<br>- Update function calls from `load_single_object` to `load_als_data`<br>- Test that functionality remains unchanged<br>**Dependencies:** Requires 0.3 (usage search) and 1.1 (ALS integration) completion |
| 2.2 | **Update imports in files using XPP loader** | `[ ]` | **Why:** Ensure XPP usage points to the enhanced main loader functionality.<br>**Implementation:** For each file found in 0.3 that imports XPP functionality:<br>- Update import statements to use main loader<br>- Verify that any hardcoded dataset paths are handled appropriately<br>- Test that XPP workflows continue to function<br>**Dependencies:** Requires 0.3 (usage search) and 1.2 (XPP enhancement) completion |
| 2.3 | **Update ptycho/xpp.py to use consolidated loader** | `[ ]` | **Why:** Make the XPP module a thin wrapper around the consolidated functionality.<br>**File:** `ptycho/xpp.py`<br>**Implementation:**<br>- Update import statement to use enhanced main loader<br>- Simplify the module to focus on XPP-specific configuration<br>- Ensure backward compatibility for any existing usage<br>**Dependencies:** Requires 2.2 (XPP import updates) completion |
| **Section 3: Testing & Validation** |
| 3.1 | **Test ALS data loading functionality** | `[ ]` | **Why:** Verify that ALS data can still be loaded correctly through the consolidated interface.<br>**Test Commands:**<br>```bash<br>python -c "from ptycho.loader import load_als_data; print('ALS loader import successful')"<br>```<br>**If ALS test data exists:**<br>```python<br>from ptycho.loader import load_als_data<br>data = load_als_data('path/to/als/test/file.npz')<br>print(f"Loaded ALS data: {type(data)}")<br>```<br>**Dependencies:** Requires 1.1 and 2.1 (ALS integration and imports) completion |
| 3.2 | **Test XPP data loading functionality** | `[ ]` | **Why:** Verify that XPP data loading continues to work with enhanced loader.<br>**Test Commands:**<br>```bash<br>python -c "from ptycho.loader import load_xpp_npz; print('XPP loader import successful')"<br>```<br>**Test with existing XPP dataset:**<br>```python<br>from ptycho.xpp import ptycho_data, ptycho_data_train<br>print(f"XPP data loaded: {type(ptycho_data)}, train: {type(ptycho_data_train)}")<br>```<br>**Dependencies:** Requires 1.2 and 2.2 (XPP enhancement and imports) completion |
| 3.3 | **Test unified loader dispatch function** | `[ ]` | **Why:** Verify that the auto-detection and format hints work correctly.<br>**Test Commands:**<br>```python<br>from ptycho.loader import load_dataset_auto<br># Test format hints<br>data_als = load_dataset_auto('test_file.npz', format_hint='als')<br>data_xpp = load_dataset_auto('test_file.npz', format_hint='xpp')<br>data_auto = load_dataset_auto('test_file.npz')  # Auto-detection<br>print("All loader dispatch tests passed")<br>```<br>**Dependencies:** Requires 1.3 (dispatch function) completion |
| **Section 4: Cleanup & Documentation** |
| 4.1 | **Remove deprecated loader files** | `[ ]` | **Why:** Complete the consolidation by removing now-redundant specialized loader files.<br>**Files to Remove:**<br>- `loaders/als.py` (functionality moved to main loader)<br>**Commands:**<br>```bash<br>git rm loaders/als.py<br>```<br>**Note:** Keep `ptycho/xpp.py` as it may serve as a configuration/compatibility layer<br>**Dependencies:** Requires all integration and testing (3.1, 3.2, 3.3) completion |
| 4.2 | **Update documentation and docstrings** | `[ ]` | **Why:** Document the new consolidated loader interface and update any references to old loaders.<br>**Files to Update:**<br>- Update docstrings in `ptycho/loader.py` for new functions<br>- Check `ptycho/loader_structure.md` for any references to specialized loaders<br>- Update any inline documentation that mentions the old loader structure<br>**Dependencies:** Requires 4.1 (cleanup) completion |

## 📝 Implementation Notes

*Use this section to document decisions, problems, and solutions during implementation:*

### Decisions Made:
- 

### Problems Encountered:
- 

### Solutions/Workarounds:
- 

### Performance Notes:
- 

### Future Improvements:
- 

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks above are marked `[D]` (Done) ✅
2. Success test passes: The unified loader correctly handles all existing data formats, and all workflows that depend on data loading (training, inference, comparison) function correctly
3. No regressions in existing tests: `python -m unittest discover -s tests`
4. All imports updated to use consolidated loader functions
5. Redundant loader files removed and codebase is DRY (Don't Repeat Yourself)

## 🔗 Quick Links

- R&D Plan: [`plan.md`](plan.md)
- Implementation Plan: [`implementation.md`](implementation.md)  
- Previous Phase: [`phase_2_checklist.md`](phase_2_checklist.md)
- Next Phase: [`phase_final_checklist.md`](phase_final_checklist.md)

---

*Checklist generated on 2025-07-22 for codebase housekeeping initiative Phase 3*