# Command: /phase-checklist <phase-number>

**Goal:** Generate a detailed checklist for a specific phase based on the implementation plan.

---

## ⚠️ **IMPORTANT: YOUR TASK AS THE AI AGENT**

Your task is to **GENERATE A PHASE CHECKLIST AND SAVE IT TO A FILE**. Do not execute the tasks in the checklist.

Your process is:
1. **Parse Phase Number:** Extract the phase number from the command (e.g., `/phase-checklist 1` → phase 1)
2. **Read Context:**
   - Read `PROJECT_STATUS.md` to get current initiative path
   - Read `<path>/implementation.md` to understand the phase details
3. **Generate Detailed Checklist:** Break down the phase into specific, actionable tasks
4. **Save File:** Save to `<path>/phase_<n>_checklist.md`
5. **Present Results:** Show the saved file path and contents

---

## 📋 **PHASE CHECKLIST GENERATION STRATEGY**

### For Each Phase Type:

#### Implementation Phases (1, 2, etc.)
- Break high-level tasks into specific code changes
- Include file paths and function names
- Add test cases for each feature
- Include verification steps

#### Final Phase (Validation & Documentation)
- List specific tests to run
- Document files to update
- Include integration verification
- Add archival steps

### Task Granularity Guidelines:
- Each task: 15-60 minutes of work
- Total tasks per phase: 8-20 tasks
- Group related tasks into sections

### Task Description Format:
```markdown
- [ ] **X.Y - <Action verb> <specific item>**
  - **File:** `<exact/path/to/file.py>`
  - **Details:** <What exactly to do>
  - **Verify:** <How to check it's done>
```

---

## 📝 **DETAILED CHECKLIST TEMPLATE**

```markdown
# Phase <N>: <Phase Name> Checklist

**Initiative:** <Initiative name from plan>
**Created:** <Today's date YYYY-MM-DD>
**Phase Goal:** <Copy from implementation plan>
**Deliverable:** <Copy from implementation plan>

## ✅ Task List

### Instructions:
1.  Work through tasks in order. Dependencies are noted in the guidance column.
2.  The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3.  Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       -
| :-- | :------------------------------------------------- | :---- | :-------------------------------------------------
| **Section 0: Preparation & Context Priming**
| 0.A | **Review Key Documents & APIs**                    | `[ ]` | **Why:** To load the necessary context and technical specifications before coding. <br> **Docs:** `docs/refactor/eval_enhancements/plan_eval_enhancements.md`, `docs/DEVELOPER_GUIDE.md` Section 5. <br> **APIs:** `skimage.metrics.structural_similarity`, `numpy.linalg.lstsq`, `ptycho.evaluation.frc50`.
| 0.B | **Identify Target Files for Modification**| `[ ]` | **Why:** To have a clear list of files that will be touched during this phase. <br> **Files:** `ptycho/evaluation.py` (Modify - core function updates), `ptycho/evaluation.py` (Add new functions for plane fitting).
| **Section 1: SSIM Integration**
| 1.A | **Add SSIM import and data range calculation**                   | `[ ]` | **Why:** SSIM requires proper data range specification for accurate calculation. <br> **How:** Add `from skimage.metrics import structural_similarity` import. Create helper function to calculate data range: `data_range = arr.max() - arr.min()`. <br> **File:** `ptycho/evaluation.py`.
| ... | ...                                                | ...   | ...
| **Section 5: Finalization**
| 5.A | **Code Formatting & Linting**                      | `[ ]` | **Why:** To maintain code quality and project standards. <br> **How:** Review code for consistent indentation, remove any debug prints, ensure proper docstrings for new functions.
| 5.B | **Update Function Docstring**                           | `[ ]` | **Why:** Document new parameters and functionality. <br> **How:** Update `eval_reconstruction` docstring to document `phase_align_method` parameter and new SSIM return values. Add docstrings for any new helper functions.

---

## 🎯 Success Criteria

**This phase is complete when:**
1.  All tasks in the table above are marked `[D]` (Done).
2.  The phase success test passes: `<specific command from implementation.md>`
3.  No regressions are introduced in the existing test suite.
