# Command: /complete-phase

**Goal:** Verify the completion of the current phase, mark it as complete, and prepare the next phase.

---

## 🚀 **EXECUTION STRATEGY**

**As the AI agent, follow these steps precisely:**

1. **Read Project Status:**
   - Read `PROJECT_STATUS.md` from the project root
   - Identify the current active initiative and its path
   - Note the current phase name

2. **Verify Implementation Plan:**
   - Navigate to the initiative path (e.g., `plans/active/<n>/`)
   - Read `implementation.md`
   - **CRITICAL:** Verify the file contains `<!-- ACTIVE IMPLEMENTATION PLAN -->`
   - If not found, STOP and report error

3. **Locate Current Phase:**
   - Find the current phase in the implementation plan
   - Identify its success test criteria
   - Note the phase checklist filename

4. **VERIFY PHASE COMPLETION:**
   - Read the current phase checklist
   - Count completed vs. total tasks
   - Execute the success test command/verification
   - **Decision Point:**
     - ✅ **If Success Test Passes:** Continue to step 5
     - ❌ **If Success Test Fails:** Execute failure protocol (see below)

5. **Update Planning Documents:**
   - **Implementation Plan Updates:**
     - Mark current phase as complete: `[x]`
     - Update progress percentages
     - Update "Current Phase" to next phase name
   - **PROJECT_STATUS.md Updates:**
     - Update current phase name
     - Update progress percentage
     - Update next milestone

6. **Prepare Next Phase:**
   - If there's a next phase:
     - Generate its checklist using the template below
     - Save to `<initiative-path>/phase_<n>_checklist.md`
   - If this was the final phase:
     - Proceed to initiative completion (step 7)

7. **Complete Initiative (if applicable):**
   - Archive the initiative:
     - Move `plans/active/<n>/` to `plans/archive/<YYYY-MM>-<n>/`
   - Update PROJECT_STATUS.md:
     - Move initiative to "Completed Initiatives"
     - Clear "Current Active Initiative"
   - Run the engagement script for next steps

---

## ❌ **FAILURE PROTOCOL**

If the success test fails:

1. **STOP IMMEDIATELY** - Do not modify any files

2. **Generate Diagnostic Report:**
   ```markdown
   ## Phase Verification Failed
   
   **Phase:** <Current Phase Name>
   **Expected:** <Success test description>
   **Actual:** <What was found instead>
   
   ### Incomplete Tasks:
   - [ ] Task X (N% complete)
   - [ ] Task Y (N% complete)
   
   ### Diagnostic Commands:
   ```<command to check status>```
   ```<another diagnostic command>```
   
   ### Suggested Actions:
   1. Check if <specific file> exists
   2. Verify <specific function> works correctly
   3. Review error logs in <location>
   ```

3. **Wait for user guidance** before proceeding

---

## 📋 **PHASE CHECKLIST TEMPLATE**

```markdown
# Phase <N>: <Phase Name> Checklist

**Created:** <Today's date>
**Phase Goal:** <Copy from implementation plan>
**Deliverable:** <Copy from implementation plan>

## 📊 Progress Tracking

**Tasks Completed:** 0 / <total>
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
| 0.2 | **Set Up Development Branch** | `[ ]` | **Why:** Isolate changes for clean Git history.<br>**How:** `git checkout -b feature/<initiative>-phase-<n>`<br>**Verify:** `git branch` shows new branch with `*` indicator. |
| 0.3 | **Verify Prerequisites** | `[ ]` | **Why:** Ensure previous phase outputs are available.<br>**How:** Check that <specific files/functions from previous phase> exist and work.<br>**API:** Test with `python -c "from module import function; print(function.__doc__)"` |
| **Section 1: <Core Implementation Area>** |
| 1.1 | **<Specific implementation task>** | `[ ]` | **Why:** <Reason this is needed>.<br>**File:** `src/module/file.py`<br>**How:** Create function `def new_function(param: Type) -> ReturnType:`<br>**API Pattern:**<br>```python<br>def function_name(param: str) -> dict:<br>    """One-line description.<br>    <br>    Args:<br>        param: Description<br>    Returns:<br>        Description<br>    """<br>    # Implementation here<br>```<br>**Test:** `pytest tests/test_module.py::test_function_name -v` |
| 1.2 | **<Related task>** | `[ ]` | **Why:** <Business/technical reason>.<br>**Depends on:** Task 1.1 must be complete.<br>**How:** Modify existing function to call new function from 1.1.<br>**Key change:**<br>```python<br># Before<br>result = old_logic()<br><br># After  <br>result = new_function(param)<br>```<br>**Gotcha:** Remember to handle the edge case where `param` is None. |
| **Section 2: Testing & Validation** |
| 2.1 | **Unit Tests for <Feature>** | `[ ]` | **Why:** Verify correctness before integration.<br>**File:** `tests/test_<module>.py`<br>**Test cases to implement:**<br>1. Normal case: `input="valid"` → `output={"status": "ok"}`<br>2. Edge case: `input=""` → `output={"status": "empty"}`<br>3. Error case: `input=None` → `raises ValueError`<br>**Template:** Use existing test pattern in file.<br>**Run:** `pytest tests/test_<module>.py -v` |
| 2.2 | **Integration Test** | `[ ]` | **Why:** Ensure feature works with rest of system.<br>**How:** Add test case to `tests/integration/test_workflow.py`<br>**Scenario:** <Describe real-world usage><br>**Verify:** All existing integration tests still pass. |
| **Section 3: Code Quality** |
| 3.1 | **Format and Lint Code** | `[ ]` | **Why:** Maintain consistent code style.<br>**Commands:**<br>`black src/<module>/ tests/`<br>`ruff check src/<module>/`<br>`mypy src/<module>/ --ignore-missing-imports`<br>**Fix:** Address any warnings before proceeding. |
| 3.2 | **Update Documentation** | `[ ]` | **Why:** Keep docs in sync with code.<br>**Files to update:**<br>- Docstrings in all new/modified functions<br>- `docs/API.md` if public API changed<br>- `README.md` if user-facing behavior changed<br>**Verify:** Generate docs with `make docs` and review. |
| **Section 4: Phase Completion** |
| 4.1 | **Run Success Test** | `[ ]` | **Test:** <Copy exact command from implementation plan><br>**Expected:** <Expected output><br>**Why:** This proves the phase deliverable works.<br>**Debug:** If fails, check <specific log file> for errors. |
| 4.2 | **Commit Changes** | `[ ]` | **Why:** Create atomic commit for this phase.<br>**Commands:**<br>`git add -A`<br>`git commit -m "[Phase <n>] <deliverable description>"`<br>`git push origin feature/<initiative>-phase-<n>`<br>**PR Title:** `[<Initiative>] Phase <n>: <description>` |

## 📝 Implementation Notes

*Document decisions and issues as you work:*

### Decisions Made:
- 

### Issues Encountered:
- 

### Performance Observations:
- 

## 🔍 Success Test Details

**Command:** `<exact command from implementation plan>`
**Expected Result:** <detailed expected output>
**Actual Result:** <fill this in after running>
**Screenshot/Log:** <paste relevant output>
```

---

## 📊 **PROGRESS CALCULATION**

When updating progress:

```python
# Phase Progress
phase_tasks_done = <count of [x] in checklist>
phase_tasks_total = <total task count>
phase_progress = (phase_tasks_done / phase_tasks_total) * 100

# Overall Progress
phases_complete = <count of [x] phases>
phases_total = <total phase count>
# Weight current phase progress
overall_progress = (phases_complete / phases_total) * 100 + 
                  (phase_progress / phases_total)
```

Update progress bars:
- Empty: ░░░░░░░░░░░░░░░░
- 25%:   ████░░░░░░░░░░░░
- 50%:   ████████░░░░░░░░
- 75%:   ████████████░░░░
- 100%:  ████████████████

---

## 🎯 **END-OF-INITIATIVE ENGAGEMENT SCRIPT**

When all phases are complete:

```markdown
🎉 **Initiative Complete!**

**Initiative:** <Name>
**Duration:** <Start date> to <End date> (<N> days)
**Phases Completed:** <N>

### Key Achievements:
- <Main deliverable 1>
- <Main deliverable 2>
- <Main deliverable 3>

The initiative has been archived to: `plans/archive/<YYYY-MM>-<n>/`

---

### What's Next?

Please describe your next objective. Consider these categories:

✨ **New Capability/Algorithm**
   - Example: "Add support for 3D reconstruction"
   - Example: "Implement adaptive learning rate"

⚡ **Performance & Optimization**
   - Example: "Reduce memory usage by 50%"
   - Example: "Speed up data loading"

🔧 **Data Pipeline & Tooling**
   - Example: "Automate experiment tracking"
   - Example: "Create visualization dashboard"

✅ **Validation & Verification**
   - Example: "Add comprehensive benchmarking"
   - Example: "Create regression test suite"

⚙️ **Refactoring & Technical Debt**
   - Example: "Modernize legacy module"
   - Example: "Improve error handling"

**Your next objective:** 
```

---

## 🔄 **RECOVERY PROCEDURES**

### If phase checklist is missing:
1. Check if it exists with different name
2. Offer to generate it from implementation plan
3. Verify phase number is correct

### If implementation plan is corrupted:
1. Check for backup in git history
2. Verify correct file path
3. Look for `<!-- ACTIVE IMPLEMENTATION PLAN -->` marker

### If PROJECT_STATUS.md is missing:
1. Offer to recreate from implementation plan
2. Search for backup or template
3. Initialize new status file
