# Final Phase: Validation, Documentation, and Cleanup Checklist

**Initiative:** Probe Parameterization Study (Corrected)  
**Created:** 2025-08-01  

**Phase Goal:** To validate the new workflow end-to-end, update all relevant documentation to reflect the new best practices, and clean up all obsolete files from the previous attempt.

**Deliverable:** A fully validated and documented workflow, an updated `PROJECT_STATUS.md`, and a clean repository state.

## ✅ **Task List**

**Instructions:**
- Work through tasks in order. Dependencies are noted in the guidance column.
- The "How/Why & API Guidance" column contains all necessary details for implementation.
- Update the State column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| **Section 1: End-to-End Validation** | | | |
| 1.A | Verify Phase 1 and 2 completion | [ ] | **Why:** Ensure all dependencies are ready before final validation.<br>**How:** Confirm all tools exist: `scripts/tools/create_hybrid_probe.py`, `ptycho/workflows/simulation_utils.py`, enhanced `scripts/simulation/simulate_and_save.py` with `--probe-file`. Verify Phase 2 scripts exist: `scripts/studies/prepare_2x2_study.py`, `scripts/studies/run_2x2_study.sh`.<br>**Verify:** All files present and tools pass basic functionality tests |
| 1.B | Run end-to-end pipeline validation | [ ] | **Why:** Validate the complete two-stage workflow works correctly.<br>**How:** Execute full pipeline: 1) Run `python scripts/studies/prepare_2x2_study.py --output-dir validation_test --quick-test`, 2) Run `bash scripts/studies/run_2x2_study.sh --study-dir validation_test --quick-test`. Verify 4 conditions created: gs1_idealized, gs1_hybrid, gs2_idealized, gs2_hybrid. Monitor for any failures or configuration issues.<br>**Command:** Complete two-stage pipeline execution |
| 1.C | Validate gridsize isolation success | [ ] | **Why:** Confirm the critical gridsize configuration bug is fixed.<br>**How:** Examine logs in `validation_test/gs1_*/` and `validation_test/gs2_*/` directories. Verify: 1) gs1 conditions show gridsize=1 in training logs, 2) gs2 conditions show gridsize=2 in training logs, 3) No cross-contamination between runs, 4) All 4 models trained successfully.<br>**Critical:** This validates the primary bug fix |
| 1.D | Performance comparison validation | [ ] | **Why:** Ensure the workflow produces meaningful scientific results.<br>**How:** Check that `compare_models.py` generated valid metrics for all 4 conditions. Verify reconstruction quality meets expected thresholds (PSNR > 20 dB). Examine output files are properly structured and contain expected keys.<br>**Files:** Check metrics.csv and reconstructions.npz in each condition directory |
| **Section 2: Documentation Updates** | | | |
| 2.A | Update DEVELOPER_GUIDE.md with gridsize lessons | [ ] | **Why:** Document the critical architectural lesson for future developers.<br>**How:** Add new section to `docs/DEVELOPER_GUIDE.md` explaining: 1) The gridsize configuration bug, 2) Why process isolation is necessary, 3) Best practices for multi-condition studies, 4) The two-stage architecture pattern. Include code examples of correct vs incorrect approaches.<br>**File:** `docs/DEVELOPER_GUIDE.md` |
| 2.B | Update COMMANDS_REFERENCE.md with new tools | [ ] | **Why:** Make new tools discoverable to users.<br>**How:** Add entries to `docs/COMMANDS_REFERENCE.md` for: 1) `create_hybrid_probe.py` with usage examples, 2) Enhanced `simulate_and_save.py` with `--probe-file` option, 3) `prepare_2x2_study.py` and `run_2x2_study.sh` workflow. Include parameter descriptions and typical use cases.<br>**File:** `docs/COMMANDS_REFERENCE.md` |
| 2.C | Update TOOL_SELECTION_GUIDE.md with 2x2 workflow | [ ] | **Why:** Help users choose the right workflow for their studies.<br>**How:** Add section to `docs/TOOL_SELECTION_GUIDE.md` describing when to use the 2x2 study workflow. Include decision matrix: single probe study vs multi-probe comparison, when process isolation is needed, computational requirements and time estimates.<br>**File:** `docs/TOOL_SELECTION_GUIDE.md` |
| 2.D | Create gridsize lessons learned document | [ ] | **Why:** Preserve institutional knowledge about this critical bug.<br>**How:** Create `docs/GRIDSIZE_CONFIGURATION_LESSONS.md` documenting: 1) The original bug manifestation, 2) Root cause analysis, 3) The solution (process isolation), 4) Prevention strategies, 5) Testing approaches for similar issues. Include specific code examples.<br>**File:** `docs/GRIDSIZE_CONFIGURATION_LESSONS.md` |
| **Section 3: Code Cleanup** | | | |
| 3.A | Identify obsolete files from failed attempts | [ ] | **Why:** Clean up experimental code that is no longer needed.<br>**How:** Review project directory for files matching patterns: `*probe_study_*`, `*corrected*`, `*test*`, `*temp*`. Identify scripts/directories that were part of the failed original implementation. Create list of files to remove, excluding the successful `probe_study_FULL/` directory.<br>**Target:** Temporary/experimental directories and scripts |
| 3.B | Remove obsolete experimental scripts | [ ] | **Why:** Prevent confusion and reduce repository clutter.<br>**How:** Delete identified obsolete files/directories. Keep: 1) Final successful results in `probe_study_FULL/`, 2) New permanent tools created in Phases 1-2, 3) This planning directory. Remove: old monolithic scripts, temporary test directories, failed implementation attempts.<br>**Safety:** Backup or git commit before deletion |
| 3.C | Update scripts/studies/ organization | [ ] | **Why:** Ensure the studies directory is well-organized with new tools.<br>**How:** Verify `scripts/studies/` contains: 1) New `prepare_2x2_study.py` and `run_2x2_study.sh`, 2) Updated `CLAUDE.md` and `README.md` documentation, 3) Clean organization with obsolete scripts removed. Update documentation to reflect current tool set.<br>**Files:** Clean and documented `scripts/studies/` directory |
| **Section 4: Project Archival** | | | |
| 4.A | Archive successful probe study results | [ ] | **Why:** Preserve the successful experimental results permanently.<br>**How:** Ensure `probe_study_FULL/` directory is complete with: 1) Final report (`2x2_study_report_final.md`), 2) All experimental data and results, 3) Probe files and visualizations, 4) Clear README explaining the contents. Consider compressing large data files if needed.<br>**Directory:** `probe_study_FULL/` with complete results |
| 4.B | Update PROJECT_STATUS.md initiative entry | [ ] | **Why:** Accurately reflect the completed initiative and new capabilities.<br>**How:** Update the "Probe Parameterization Study" entry in `docs/PROJECT_STATUS.md`: 1) Confirm status as Complete, 2) Update deliverables to reflect actual tools created, 3) Add note about the gridsize bug fix, 4) Ensure planning documents are correctly linked.<br>**File:** `docs/PROJECT_STATUS.md` |
| 4.C | Archive planning documents | [ ] | **Why:** Preserve the complete planning and implementation record.<br>**How:** Verify `plans/archive/2025-08-probe-parameterization/` contains: 1) All planning documents (plan.md, implementation.md, checklists), 2) Clear file organization, 3) Updated status indicating completion. Ensure documents are linked correctly from PROJECT_STATUS.md.<br>**Directory:** Complete archived planning documents |
| **Section 5: Final Verification** | | | |
| 5.A | Run complete test suite | [ ] | **Why:** Ensure no regressions were introduced by the changes.<br>**How:** Execute full project test suite: `python -m unittest discover -s tests -p "test_*.py"`. All existing tests should pass. Pay particular attention to simulation and evaluation tests. Fix any failures before marking complete.<br>**Command:** `python -m unittest discover -s tests -p "test_*.py"` |
| 5.B | Verify new tools integration | [ ] | **Why:** Confirm all new tools work correctly in the integrated codebase.<br>**How:** Test each new tool individually: 1) `create_hybrid_probe.py` with sample data, 2) `simulate_and_save.py --probe-file`, 3) `prepare_2x2_study.py --quick-test`, 4) `run_2x2_study.sh --quick-test`. All should execute without errors.<br>**Verification:** All tools function correctly |
| 5.C | Documentation consistency check | [ ] | **Why:** Ensure all documentation is accurate and cross-referenced correctly.<br>**How:** Review all updated documentation files for: 1) Correct file paths and script names, 2) Accurate command examples, 3) Proper cross-references and links, 4) Consistent terminology and style. Fix any inconsistencies found.<br>**Files:** All updated documentation files |

---

## 🎯 **Success Criteria**

This phase is complete when:

1. **All tasks in the table above are marked [D] (Done).**
2. **End-to-end validation succeeds:** The complete `prepare_2x2_study.py` followed by `run_2x2_study.sh` workflow runs successfully.
3. **Gridsize bug confirmed fixed:** Logs demonstrate correct gridsize usage with no configuration leakage between runs.
4. **Documentation is comprehensive:** All new tools and lessons learned are properly documented.
5. **Repository is clean:** Obsolete files removed, successful results preserved, planning documents archived.
6. **No regressions:** Full test suite passes and all new tools integrate correctly.