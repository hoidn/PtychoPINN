# Command: /complete-phase-gemini-full

**Goal:** Have Gemini perform complete phase verification and prepare next phase while Claude manages files.

---

## ⚠️ **YOUR ROLE AS CLAUDE**

You are a **FILE MANAGER** only:
1. Identify current phase and success criteria
2. Have Gemini verify everything
3. Update files based on Gemini's verdict
4. Either advance to next phase or report failures

**DO NOT** make any verification decisions yourself.

---

## 🚀 **EXECUTION WORKFLOW**

### Step 1: Load Current State (Claude)

1. Read `PROJECT_STATUS.md` for current phase
2. Read implementation plan for success test
3. Read current phase checklist
4. Note which files were modified

### Step 2: Full Verification by Gemini (User Executes)

Generate this command:

```bash
gemini -p "@./ Perform COMPLETE phase verification and next phase preparation.

CURRENT PHASE: Phase [N] - [Name]
SUCCESS TEST: [From implementation plan]
MODIFIED FILES: [List from checklist]
PHASE GOAL: [From implementation plan]

Perform comprehensive verification:

## 1. IMPLEMENTATION VERIFICATION

Check EVERY task from the phase checklist:
- Are all required functions implemented?
- Do they follow the specified signatures?
- Is error handling complete?
- Are edge cases covered?

For each issue found, specify:
- Severity: BLOCKER | WARNING | SUGGESTION
- Location: file:line
- Issue: [specific problem]
- Fix: [specific solution]

## 2. TEST VERIFICATION

Analyze test coverage:
- What % of new code has tests?
- Which functions lack test coverage?
- Are test assertions meaningful?
- Do tests cover edge cases?

Run these test commands and report results:
\`\`\`bash
[Unit test command]
[Integration test command]
[Success test from plan]
\`\`\`

## 3. INTEGRATION VERIFICATION

Check integration points:
- Do all imports work?
- Are APIs correctly integrated?
- Any breaking changes to existing code?
- Performance impact?

## 4. CODE QUALITY CHECK

Verify:
- Code follows project patterns (show examples)
- Documentation is complete
- Type hints are present
- No security issues
- No performance antipatterns

## 5. SUCCESS TEST EXECUTION

Run: [success test command]
Expected: [expected output]
Actual: [report actual output]
VERDICT: PASS | FAIL

## 6. OVERALL PHASE VERDICT

Based on ALL above checks:
**PHASE STATUS: COMPLETE | INCOMPLETE**

If INCOMPLETE, list all BLOCKERS that must be fixed.

If COMPLETE, prepare for next phase:

## 7. NEXT PHASE PREPARATION (only if COMPLETE)

[If there is a next phase in the implementation plan]

Analyze Phase [N+1] requirements and provide:

### Detailed Task Breakdown
Break down into 10-25 specific tasks with:
- Exact file:line locations
- Current code that needs changing
- New code to write
- Test cases needed
- Integration points

### Risk Analysis
- Technical challenges
- Dependencies on Phase [N] outputs
- Potential breaking changes
- Performance considerations

### Time Estimate
Based on:
- Code complexity analysis
- Similar changes in codebase
- Number of integration points
- Test requirements

Total estimate: [X] hours

## 8. GEMINI VERIFICATION SUMMARY

**Phase [N] Verification Results:**
- Implementation: [PASS/FAIL - X issues]
- Tests: [PASS/FAIL - X% coverage]  
- Integration: [PASS/FAIL - X issues]
- Quality: [PASS/FAIL - X issues]
- Success Test: [PASS/FAIL]

**Overall Verdict:** [COMPLETE/INCOMPLETE]

**Next Phase Readiness:** [Ready to proceed / Blockers must be fixed]

END OF VERIFICATION - Output the complete analysis."
```

### Step 3: Process Gemini's Verdict (Claude)

Read Gemini's output and check the **PHASE STATUS** and **Overall Verdict**.

#### If COMPLETE:

1. **Update implementation.md:**
   - Mark current phase `[x]`
   - Update Current Phase to next phase
   - Update progress percentage

2. **Update PROJECT_STATUS.md:**
   - Update current phase name
   - Update progress bar
   - Add verification timestamp

3. **If next phase exists:**
   - Save next phase prep as `phase_<n+1>_prep.md`
   - Tell user to run `/phase-checklist-gemini-full <n+1>`

4. **If final phase complete:**
   - Move `plans/active/<n>/` to `plans/archive/<date>-<n>/`
   - Update PROJECT_STATUS.md
   - Ask user for next initiative

#### If INCOMPLETE:

1. **Create fix checklist:**
   Save Gemini's blockers as `phase_<n>_fixes.md`

2. **Report to user:**
   ```
   ❌ Phase [N] verification FAILED
   
   Gemini found [X] BLOCKERS that must be fixed:
   [List blockers]
   
   Fix checklist saved to: phase_<n>_fixes.md
   
   After fixing, run /complete-phase-gemini-full again.
   ```

3. **DO NOT** update any progress tracking

### Step 4: Final User Communication

For successful completion:
```
✅ Phase [N] VERIFIED COMPLETE by Gemini

Verification Summary:
- Implementation: PASS
- Tests: PASS ([X]% coverage)
- Integration: PASS  
- Quality: PASS
- Success Test: PASS

✅ Progress updated in all tracking files
✅ Next phase preparation saved

Next step: Run `/phase-checklist-gemini-full [N+1]` to generate the detailed checklist.
```

---

## 🎯 **BENEFITS OF FULL GEMINI VERIFICATION**

1. **Comprehensive Checks** - Every line of code verified
2. **Real Test Execution** - Actually runs the tests
3. **Integration Analysis** - Checks entire system impact
4. **Quality Assurance** - Pattern compliance verified
5. **Next Phase Prep** - Based on current state analysis

---

## ⚠️ **CRITICAL RULES FOR CLAUDE**

- **NEVER** decide if phase is complete yourself
- **ONLY** act on Gemini's verdict
- **NEVER** modify Gemini's analysis
- **ALWAYS** require COMPLETE status to proceed
- **IF** ambiguous, default to INCOMPLETE

---

## 🔄 **SPECIAL CASES**

### Partial Success
If Gemini reports some passes but has WARNINGS:
- Still mark INCOMPLETE if any BLOCKERS
- Create fix list for BLOCKERS only
- Note WARNINGS in progress tracking

### Performance Regression
If Gemini reports performance issues:
- Treat as BLOCKER if >20% degradation
- Create specific performance fix checklist
- Include profiling commands

### Missing Tests
If Gemini reports <80% test coverage:
- Treat as BLOCKER
- Generate specific test checklist
- Include test templates from codebase
