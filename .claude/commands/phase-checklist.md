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
**Estimated Duration:** <Copy from implementation plan>

## 📊 Progress Tracking

**Tasks:** 0 / <total> completed
**Status:** 🔴 Not Started → 🟡 In Progress → 🟢 Complete
**Started:** -
**Completed:** -
**Actual Duration:** -

## ✅ Task List

### Section 0: Phase Preparation
- [ ] **0.1 - Review phase requirements**
  - **Files:** 
    - `plans/active/<n>/plan.md` (objectives)
    - `plans/active/<n>/implementation.md` (this phase)
  - **Details:** Understand deliverables and success criteria
  - **Verify:** Can explain phase goal in one sentence

- [ ] **0.2 - Set up development branch**
  - **Commands:**
    ```bash
    git checkout main
    git pull origin main
    git checkout -b feature/<initiative>-phase-<n>
    ```
  - **Verify:** `git branch` shows new branch

- [ ] **0.3 - Verify prerequisites**
  - **Details:** Ensure previous phase outputs exist
  - **Verify:** <Specific files/functions from previous phase work>

### Section 1: <Main Implementation Area>
<For each high-level task from implementation plan, create 2-4 specific subtasks>

- [ ] **1.1 - <Specific implementation task>**
  - **File:** `src/<module>/<file>.py`
  - **Function/Class:** `<name>`
  - **Details:** 
    - <Specific change 1>
    - <Specific change 2>
  - **Code hint:**
    ```python
    # Example structure
    def new_function(param: Type) -> ReturnType:
        """<Docstring>."""
        pass
    ```
  - **Verify:** <Specific test command or check>

- [ ] **1.2 - <Related implementation task>**
  - **File:** `src/<module>/<file>.py`
  - **Details:** <What to implement>
  - **Dependencies:** Requires 1.1 completion
  - **Verify:** <How to test>

### Section 2: Testing & Validation
- [ ] **2.1 - Unit tests for <feature from Section 1>**
  - **File:** `tests/test_<module>.py`
  - **Test cases:**
    - Normal operation: <input> → <expected output>
    - Edge case: <input> → <expected output>
    - Error case: <input> → <expected error>
  - **Command:** `pytest tests/test_<module>.py::test_<function>`
  - **Verify:** All tests pass

- [ ] **2.2 - Integration test**
  - **File:** `tests/integration/test_<feature>.py`
  - **Details:** Test interaction with <other module>
  - **Verify:** Integration test passes

- [ ] **2.3 - Manual verification**
  - **Steps:**
    1. Run `python scripts/<script>.py --test`
    2. Check output contains <expected string>
    3. Verify <side effect> occurred
  - **Verify:** Output matches expectations

### Section 3: Code Quality & Documentation  
- [ ] **3.1 - Code formatting and linting**
  - **Commands:**
    ```bash
    black src/<module>/ tests/
    ruff check src/<module>/ tests/
    mypy src/<module>/ (if using type hints)
    ```
  - **Verify:** No errors or warnings

- [ ] **3.2 - Update docstrings and comments**
  - **Files:** All modified files
  - **Details:** 
    - Add docstrings to new functions
    - Update existing docstrings if behavior changed
    - Add inline comments for complex logic
  - **Verify:** Every public function has a docstring

- [ ] **3.3 - Update type hints (if applicable)**
  - **Files:** All new functions
  - **Verify:** `mypy` passes without errors

### Section 4: Phase Finalization
- [ ] **4.1 - Run phase success test**
  - **Test:** <Copy exact success test from implementation plan>
  - **Expected:** <Expected outcome>
  - **Actual:** <To be filled during execution>
  - **Verify:** Test passes as expected

- [ ] **4.2 - Update progress tracking**
  - **Files:**
    - This checklist (mark tasks complete)
    - Consider updating PROJECT_STATUS.md progress
  - **Verify:** All tasks marked complete

- [ ] **4.3 - Commit and push changes**
  - **Commands:**
    ```bash
    git add -A
    git commit -m "[Phase <n>] <Description of deliverable>"
    git push origin feature/<initiative>-phase-<n>
    ```
  - **Verify:** Changes pushed successfully

- [ ] **4.4 - Create pull request (if applicable)**
  - **Title:** `[<Initiative>] Phase <n>: <Phase name>`
  - **Description:** Link to implementation plan and list key changes
  - **Verify:** PR created and CI passes

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
1. All tasks above are marked complete ✅
2. Success test passes: `<specific command>`
3. No regressions in existing tests
4. Code is committed and pushed
5. Documentation is updated

## 🔗 Quick Links

- R&D Plan: [`plan.md`](plan.md)
- Implementation Plan: [`implementation.md`](implementation.md)
- Previous Phase: [`phase_<n-1>_checklist.md`](phase_<n-1>_checklist.md) *(if applicable)*
- Next Phase: [`phase_<n+1>_checklist.md`](phase_<n+1>_checklist.md) *(if applicable)*

---

*Checklist generated on <date> by /phase-checklist command*
```

---

## 💡 **PHASE-SPECIFIC CUSTOMIZATION**

### Phase 1 (Usually Core Implementation)
- Focus on basic functionality
- More detailed code structure hints
- Emphasis on getting something working

### Phase 2+ (Usually Extensions/Integration)  
- Build on Phase 1 foundations
- More integration tests
- Cross-module interactions

### Final Phase (Always Validation & Documentation)
- Comprehensive testing checklist
- All documentation updates
- Performance verification
- Archive preparation tasks

---

## 🎯 **TASK GENERATION HEURISTICS**

1. **From "implement X" → Specific tasks:**
   - Create/modify the main function
   - Add error handling
   - Create unit tests
   - Add integration point

2. **From "refactor Y" → Specific tasks:**
   - Analyze current structure
   - Create new structure
   - Migrate functionality
   - Update all references
   - Verify behavior unchanged

3. **From "add support for Z" → Specific tasks:**
   - Define new interface/API
   - Implement core logic
   - Add configuration options
   - Create examples
   - Test edge cases

---

## ⚡ **QUICK COMMAND REFERENCE**

```bash
# Generate checklist for phase 1
/phase-checklist 1

# Generate checklist for phase 2  
/phase-checklist 2

# Generate final phase checklist
/phase-checklist final
```

