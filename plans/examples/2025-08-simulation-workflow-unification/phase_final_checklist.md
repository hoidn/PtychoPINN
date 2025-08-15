# Phase Final: Deprecation, Documentation & Cleanup Checklist

**Initiative:** Simulation Workflow Unification
**Created:** 2025-08-02
**Phase Goal:** To add deprecation warnings to the legacy method, update all documentation, and ensure the solution is production-ready.
**Deliverable:** Complete documentation updates, deprecation warnings in place, and all success criteria from the R&D plan verified.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :------------------------------------------------- |
| **Section 0: Preparation & Verification** |
| 0.A | **Verify All Tests Pass**                          | `[ ]` | **Why:** To ensure previous phases are complete and working. <br> **How:** Run full test suite including the new tests from Phase 2: `pytest tests/simulation/test_simulate_and_save.py -v`. Also run any existing simulation-related tests. <br> **Verify:** All tests show green/passing status. |
| 0.B | **Run Final Integration Test**                      | `[ ]` | **Why:** To confirm the core fix works end-to-end. <br> **How:** Run the success test from Phase 1: `python scripts/simulation/simulate_and_save.py --input-file datasets/fly/fly001_transposed.npz --output-file test_sim_final.npz --gridsize 2`. <br> **Verify:** Completes without errors, output file is valid. |
| **Section 1: Deprecation Implementation** |
| 1.A | **Add DeprecationWarning to Legacy Method**         | `[ ]` | **Why:** To guide users away from the problematic legacy code. <br> **How:** Locate `RawData.from_simulation` method in `ptycho/raw_data.py`. Add warning at the start of the method: `warnings.warn("RawData.from_simulation is deprecated and has bugs with gridsize > 1. Use simulate_and_save.py directly.", DeprecationWarning, stacklevel=2)`. <br> **File:** `ptycho/raw_data.py` |
| 1.B | **Search for Legacy Method Usage**                  | `[ ]` | **Why:** To identify any other code that needs updating. <br> **How:** Use grep/ripgrep to search entire codebase: `rg "from_simulation" --type py`. Document all occurrences in a findings report. <br> **Output:** Create list of files and line numbers where method is used. |
| 1.C | **Document Migration Path**                         | `[ ]` | **Why:** To help users transition from the deprecated method. <br> **How:** Create a migration guide section in the deprecation docstring explaining: 1) Why it's deprecated, 2) What to use instead, 3) Key differences in usage. <br> **Example:** "Instead of RawData.from_simulation(), use scripts/simulation/simulate_and_save.py with appropriate command-line arguments." |
| **Section 2: Documentation Updates** |
| 2.A | **Create/Update scripts/simulation/CLAUDE.md**      | `[ ]` | **Why:** To document the new architecture for AI agents. <br> **How:** Create or update CLAUDE.md with: 1) Overview of the refactored workflow, 2) Explanation of the modular approach, 3) Common usage patterns, 4) Troubleshooting guide for tensor shape issues. <br> **File:** `scripts/simulation/CLAUDE.md` |
| 2.B | **Update scripts/simulation/README.md**             | `[ ]` | **Why:** To provide user-facing documentation. <br> **How:** Update README with: 1) Clear explanation of changes, 2) New usage examples for gridsize > 1, 3) Migration notes from old workflow, 4) Performance considerations. Include example commands for common use cases. <br> **File:** `scripts/simulation/README.md` |
| 2.C | **Update Main CLAUDE.md Simulation Section**        | `[ ]` | **Why:** To reflect the unified workflow in project docs. <br> **How:** Review main `CLAUDE.md` for any references to simulation workflow. Update the "Simulating a Dataset" section to emphasize the modular approach and mention the gridsize > 1 support. <br> **File:** `CLAUDE.md` (if simulation section exists) |
| 2.D | **Add Workflow to Tool Selection Guide**            | `[ ]` | **Why:** To help users choose the right simulation approach. <br> **How:** If `docs/TOOL_SELECTION_GUIDE.md` exists, add entry explaining when to use simulate_and_save.py vs other simulation methods. Emphasize it's the correct choice for gridsize > 1. <br> **File:** `docs/TOOL_SELECTION_GUIDE.md` |
| **Section 3: Success Criteria Verification** |
| 3.A | **Verify No Crashes with Gridsize > 1**            | `[ ]` | **Why:** Core success criterion from R&D plan. <br> **How:** Run simulation with various gridsize values: 1, 2, 3. Each should complete successfully. Test with different n_images values as well. <br> **Commands:** `--gridsize 1`, `--gridsize 2`, `--gridsize 3` |
| 3.B | **Verify Data Contract Compliance**                 | `[ ]` | **Why:** Ensure output follows specifications. <br> **How:** Use the data contract test from Phase 2 or manually verify: 1) diffraction is float32 amplitude, 2) All required keys present, 3) Shapes are correct for each gridsize. <br> **Reference:** `docs/data_contracts.md` |
| 3.C | **Verify Performance Benchmarks**                   | `[ ]` | **Why:** Ensure no significant regression. <br> **How:** Run performance test from Phase 2 or manually time execution for 1000 images. Compare with baseline if available. Document results in implementation notes. <br> **Acceptable:** Within 20% of original performance. |
| 3.D | **Verify All Tests Passing**                        | `[ ]` | **Why:** Comprehensive validation. <br> **How:** Run full project test suite: `python -m pytest`. Ensure no regressions in other parts of the codebase. Pay special attention to any tests that use simulation functionality. |
| **Section 4: Code Quality & Cleanup** |
| 4.A | **Remove Debug/Development Code**                   | `[ ]` | **Why:** To prepare for production use. <br> **How:** Review the refactored simulate_and_save.py for any debug print statements, commented-out code, or TODO comments. Remove or properly address them. <br> **File:** `scripts/simulation/simulate_and_save.py` |
| 4.B | **Ensure Proper Error Messages**                    | `[ ]` | **Why:** To help users troubleshoot issues. <br> **How:** Review error handling in simulate_and_save.py. Ensure all error messages are informative and suggest solutions. Test with invalid inputs to verify messages. |
| 4.C | **Add Version/Change Comments**                     | `[ ]` | **Why:** To track when changes were made. <br> **How:** Add comments to modified files indicating the refactoring date and purpose. Example: `# Refactored 2025-08-02: Replaced monolithic from_simulation with modular workflow to fix gridsize > 1` |
| **Section 5: Final Integration & Archival** |
| 5.A | **Create Implementation Summary**                   | `[ ]` | **Why:** To document what was accomplished. <br> **How:** Write a brief summary of the implementation including: 1) What was changed, 2) Why it was changed, 3) Key technical decisions, 4) Any remaining limitations. Save as `implementation_summary.md` in the initiative folder. |
| 5.B | **Update Initiative Status**                        | `[ ]` | **Why:** To track project progress. <br> **How:** Prepare to update PROJECT_STATUS.md to mark this initiative as complete. Note the completion date and key deliverables. This will be done after all tasks are complete. |
| 5.C | **Prepare for Code Review**                         | `[ ]` | **Why:** To ensure quality before merging. <br> **How:** Create a summary of all changed files, run a final diff to review all modifications. Ensure commit messages are clear and reference the initiative. Prepare PR description if needed. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes: All R&D plan success criteria are verified as complete, documentation is updated, and deprecation warnings are properly displayed when using the legacy method.
3. No regressions are introduced in the existing test suite.
4. All documentation clearly explains the new workflow and migration path.

## 📝 Notes

- The deprecation warning should be informative but not alarm users unnecessarily - the legacy method still works for gridsize=1.
- Documentation should emphasize the benefits of the new modular approach, not just the bug fix.
- Consider creating a simple diagram showing the new workflow if it would help users understand the changes.
- The implementation summary (5.A) should be concise but comprehensive enough for future maintainers to understand what changed and why.