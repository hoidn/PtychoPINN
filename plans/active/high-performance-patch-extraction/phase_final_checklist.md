# Final Phase: Validation & Documentation Checklist

**Initiative:** High-Performance Patch Extraction Refactoring
**Created:** 2025-08-03
**Phase Goal:** Enable the new implementation by default, update documentation, and prepare for legacy code removal.
**Deliverable:** Updated configuration defaults, comprehensive documentation, and a cleanup plan.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :-------------------------------------------------
| **Section 0: Pre-Validation Checks**
| 0.A | **Verify All Previous Phases Complete**            | `[ ]` | **Why:** To ensure all prerequisites are met before finalizing. <br> **How:** Confirm: <br>1. Phase 1: `_get_image_patches_batched` and `_get_image_patches_iterative` exist <br>2. Phase 2: Dispatcher works with feature flag <br>3. Phase 3: All equivalence tests pass, performance >10x <br>Review git log for phase commits. <br> **Command:** `git log --oneline -10`
| 0.B | **Review Performance Report**                      | `[ ]` | **Why:** To document the achieved performance improvements before making default. <br> **How:** Run performance report generator from Phase 3: `python -m pytest tests/test_patch_extraction_equivalence.py::generate_performance_report -v`. Review results to confirm: <br>1. Speedup >= 10x achieved <br>2. Memory usage < 1.2x baseline <br>3. All configurations tested <br> **File:** `test_outputs/patch_extraction_performance.txt`
| **Section 1: Full Regression Testing**
| 1.A | **Run Complete Test Suite - Iterative Mode**       | `[ ]` | **Why:** To establish baseline behavior with current default. <br> **How:** Set environment variable or config to force iterative mode, then run full test suite: <br>`export USE_BATCHED_PATCH_EXTRACTION=false` <br>`python -m unittest discover -s tests -p "test_*.py" -v` <br>Document any test times and results. <br> **Verify:** All tests pass
| 1.B | **Run Complete Test Suite - Batched Mode**         | `[ ]` | **Why:** To ensure no regressions with new implementation. <br> **How:** Set environment to force batched mode: <br>`export USE_BATCHED_PATCH_EXTRACTION=true` <br>`python -m unittest discover -s tests -p "test_*.py" -v` <br>Compare results with iterative baseline. <br> **Verify:** All tests pass, similar or better performance
| 1.C | **Test Real Workflow Scenarios**                   | `[ ]` | **Why:** To validate in realistic usage patterns beyond unit tests. <br> **How:** Run key workflows that use patch extraction: <br>1. Small training run: `ptycho_train --train_data datasets/fly/fly001_transposed.npz --n_images 100 --output_dir test_batch_validation` <br>2. Verify output quality matches expectations <br>3. Check logs confirm batched implementation used <br> **Verify:** Training completes successfully
| **Section 2: Update Configuration Defaults**
| 2.A | **Change Default Value in ModelConfig**            | `[ ]` | **Why:** To enable the performance improvement for all users by default. <br> **How:** In `ptycho/config/config.py`, locate `use_batched_patch_extraction` in ModelConfig and change default from `False` to `True`: <br>`use_batched_patch_extraction: bool = True` <br> **File:** `ptycho/config/config.py`
| 2.B | **Update Legacy Config Default**                   | `[ ]` | **Why:** To ensure consistency across configuration systems. <br> **How:** If there's a separate default initialization for the legacy params system, update it to match. Look for initialization code that sets `params['use_batched_patch_extraction']` and ensure it defaults to `True`. <br> **File:** Configuration initialization location
| 2.C | **Test Default Behavior**                          | `[ ]` | **Why:** To confirm the new default is active without explicit configuration. <br> **How:** Create a minimal test script that: <br>1. Imports and uses `get_image_patches` without setting any config <br>2. Verifies via logs that batched implementation is used <br>3. Confirms output is valid <br> **Verify:** Batched implementation is used by default
| **Section 3: Documentation Updates**
| 3.A | **Update DEVELOPER_GUIDE.md**                      | `[ ]` | **Why:** To document the architectural improvement for future developers. <br> **How:** Add a new subsection in Section 4 (Data Pipeline) or create Section 4.5 titled "High-Performance Patch Extraction": <br>1. Explain the batched approach vs old iterative <br>2. Document performance characteristics (10x+ speedup) <br>3. Note the feature flag for emergency rollback <br>4. Reference the equivalence test suite <br> **File:** `docs/DEVELOPER_GUIDE.md`
| 3.B | **Update Function Docstrings**                     | `[ ]` | **Why:** To ensure code documentation reflects the new implementation. <br> **How:** Update docstrings in `ptycho/raw_data.py`: <br>1. `get_image_patches`: Note it's now a dispatcher, mention performance improvement <br>2. `_get_image_patches_batched`: Mark as the primary implementation <br>3. `_get_image_patches_iterative`: Mark as legacy/deprecated <br>Include performance notes in module docstring. <br> **File:** `ptycho/raw_data.py`
| 3.C | **Add Performance Notes to README**                | `[ ]` | **Why:** To highlight the improvement in project documentation. <br> **How:** If performance optimizations are mentioned in README or CHANGELOG: <br>1. Add note about 10x+ faster patch extraction <br>2. Mention this affects data loading performance <br>3. Note the configurable fallback option <br> **Files:** `README.md`, `CHANGELOG.md` (if exists)
| **Section 4: Deprecation Planning**
| 4.A | **Add Deprecation Warning**                        | `[ ]` | **Why:** To notify users that the iterative implementation will be removed. <br> **How:** In `_get_image_patches_iterative`, add a deprecation warning: <br>```python<br>import warnings<br>warnings.warn(<br>    "The iterative patch extraction implementation is deprecated and will be removed in v2.0. "<br>    "The batched implementation is now default and provides 10x+ speedup.",<br>    DeprecationWarning,<br>    stacklevel=2<br>)<br>``` <br> **File:** `ptycho/raw_data.py`
| 4.B | **Document Deprecation Timeline**                  | `[ ]` | **Why:** To set clear expectations for code removal. <br> **How:** Create or update a DEPRECATIONS.md file with: <br>1. Feature: Iterative patch extraction <br>2. Deprecated: v1.X (current version) <br>3. Removal planned: v2.0 <br>4. Migration: Already automatic, use `use_batched_patch_extraction=False` if issues <br> **File:** `DEPRECATIONS.md` or similar
| 4.C | **Create Follow-up Issue**                         | `[ ]` | **Why:** To track the future cleanup task. <br> **How:** Create a GitHub issue (or project-specific tracking) titled "Remove legacy iterative patch extraction": <br>1. Link to this initiative <br>2. Set milestone for v2.0 <br>3. List files to clean: remove `_get_image_patches_iterative`, feature flag, related tests <br>4. Note: Wait for at least one release cycle <br> **Verify:** Issue created and tracked
| **Section 5: Final Validation**
| 5.A | **Performance Validation with Real Data**          | `[ ]` | **Why:** To confirm improvements on production-scale data. <br> **How:** If available, run patch extraction on a large dataset: <br>1. Time the operation with new default <br>2. Compare with Phase 3 benchmarks <br>3. Verify memory usage is acceptable <br>4. Document results for release notes <br> **Verify:** Performance matches expectations
| 5.B | **Review All Changes**                             | `[ ]` | **Why:** To ensure completeness before final commit. <br> **How:** Review all modified files: <br>1. `git diff --stat` to see all changes <br>2. Ensure no debug code remains <br>3. Verify all TODOs are addressed <br>4. Check documentation is complete <br> **Command:** `git diff --cached`
| **Section 6: Release Preparation**
| 6.A | **Update Release Notes**                           | `[ ]` | **Why:** To communicate the improvement to users. <br> **How:** Prepare release note entry: <br>"**Performance:** Patch extraction now uses batched operations, providing 10-100x speedup for data loading. The old implementation remains available via `use_batched_patch_extraction=False` configuration." <br>Include benchmark results. <br> **File:** Release notes location
| 6.B | **Final Commit**                                   | `[ ]` | **Why:** To complete the initiative with all changes. <br> **How:** Stage all changes and commit: <br>`git add -A` <br>`git commit -m "Final Phase: Enable batched patch extraction by default and update documentation"` <br>Include key metrics in commit message. Push to feature branch. <br> **Command:** `git push origin feature/high-performance-patch-extraction`
| 6.C | **Prepare PR/Merge Request**                       | `[ ]` | **Why:** To integrate the improvement into the main codebase. <br> **How:** Create pull request with: <br>1. Title: "High-Performance Patch Extraction: 10x+ Speedup" <br>2. Link all phase commits <br>3. Include performance benchmarks <br>4. Note the safe rollback option <br>5. Request review from stakeholders <br> **Verify:** PR created and ready for review

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: All project tests pass with the new implementation as default, and documentation accurately reflects the changes.
3. Performance improvements are documented and validated on real data.
4. A clear deprecation path is established for the legacy implementation.
5. The feature branch is ready for merge with a complete pull request.