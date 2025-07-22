### **File: `.claude/commands/implementation.md` (Corrected Revision, No Placeholders)**

```markdown
# Command: /implementation

**Goal:** Generate and save a phased implementation plan document, including Git state tracking, based on the most recent R&D plan and established decomposition principles.

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**THIS COMMAND MUST FOLLOW THIS EXACT SEQUENCE:**
1.  You MUST read `PROJECT_STATUS.md` to identify the current initiative.
2.  You MUST read the corresponding `plan.md`.
3.  You MUST execute the Git commands in the "State Persistence Logic" section to capture the feature branch, baseline branch, and baseline commit hash.
4.  You MUST decompose the work into logical phases, following the "PHASE DECOMPOSITION GUIDELINES".
5.  You MUST generate the `implementation.md` file using the provided template, populating all Git state fields correctly.
6.  You MUST update `PROJECT_STATUS.md` to advance the initiative to Phase 1.

**DO NOT:**
-   ❌ Generate the implementation plan without first capturing the Git state.
-   ❌ Ignore the phase decomposition guidelines when creating the plan.
-   ❌ Use complex or untested shell commands.

**EXECUTION CHECKPOINT:** Before saving the `implementation.md` file, you must verify that the `Baseline Commit Hash` field contains a valid Git commit hash, not an error message or an empty string.

---

## 🤖 **CONTEXT: YOU ARE CLAUDE CODE**

You are Claude Code, an autonomous command-line tool. You will execute the Git commands and file operations described below directly and without human intervention to create the implementation plan.

---

## 📋 **YOUR EXECUTION WORKFLOW**

### Step 1: Read Context
-   Read `PROJECT_STATUS.md` to get the current initiative path.
-   Read `<path>/plan.md` to understand the project goals and technical details.

### Step 2: 🔴 MANDATORY - Capture Git State
-   Execute the shell commands provided in the "State Persistence Logic" section below to determine the feature branch, baseline branch, and baseline commit hash.

### Step 3: Decompose Work into Phases
-   Analyze the "Core Capabilities" and "Technical Implementation Details" from the `plan.md`.
-   Using the **"PHASE DECOMPOSITION GUIDELINES"** below, break the work down into a sequence of 2-5 logical phases. Each phase should have a clear goal and a verifiable deliverable.

### Step 4: Generate and Save Implementation Plan
-   Generate the full content for the implementation plan using the "IMPLEMENTATION PLAN TEMPLATE" below.
-   Populate the "Git Workflow Information" section with the values you captured in Step 2.
-   Populate the "IMPLEMENTATION PHASES" section with the phases you designed in Step 3.
-   Save the content to `<initiative-path>/implementation.md`.

### Step 5: Update Project Status
-   Update the `PROJECT_STATUS.md` file using the "PROJECT STATUS UPDATE" section as a guide.

### Step 6: Confirm and Present
-   Announce that the implementation plan has been created and the project status has been updated.
-   Present the full content of the generated `implementation.md` for the user's review.

---

## 🔒 **State Persistence Logic (Revised for Robustness)**

You must execute the following shell commands. This robust, sequential approach is tested to work in your environment.

```bash
# 1. Get the current feature branch name
feature_branch=$(git rev-parse --abbrev-ref HEAD)

# 2. Determine the baseline branch using a safe, sequential method.
if git rev-parse --verify main >/dev/null 2>&1; then
    baseline_branch="main"
elif git rev-parse --verify master >/dev/null 2>&1; then
    baseline_branch="master"
elif git rev-parse --verify develop >/dev/null 2>&1; then
    baseline_branch="develop"
else
    baseline_branch=$(git rev-parse --abbrev-ref HEAD@{upstream} | sed 's/.*\///' || echo "main")
fi
echo "Baseline branch determined as: $baseline_branch"

# 3. Get the commit hash of that baseline branch
baseline_hash=$(git rev-parse "$baseline_branch")
echo "Baseline commit hash: $baseline_hash"
```

---

## 💡 **PHASE DECOMPOSITION GUIDELINES**

When breaking work into phases, you **MUST** follow these principles:

1.  **Each phase must produce a verifiable deliverable.**
    -   Bad: "Refactor code"
    -   Good: "Refactored module with all original unit tests passing"

2.  **Phases should be logically independent when possible.**
    -   Can the phase be tested on its own?
    -   Does it produce something that can be reviewed independently?

3.  **Consider natural boundaries in the work:**
    -   **Data First:** Data model changes → Business logic changes → API/UI changes.
    -   **Foundation First:** Core logic → Error handling → Performance optimization.
    -   **Backend then Frontend:** API implementation → Frontend integration.

4.  **Size phases appropriately:**
    -   Aim for phases that represent approximately 1-2 days of focused work.
    -   A simple initiative might have only one implementation phase.
    -   A complex initiative should not exceed 4 implementation phases.

5.  **The final phase is always "Validation & Documentation".**
    -   This phase is not for new features.
    -   It must include tasks for comprehensive end-to-end testing, verifying all success criteria from the R&D plan, and updating all relevant documentation.

---

## 템플릿 & 가이드라인 (Templates & Guidelines)

### **IMPLEMENTATION PLAN TEMPLATE**
*This is the template for the content of `implementation.md`.*
```markdown
<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** <Name from R&D Plan>
**Initiative Path:** `plans/active/<initiative-name>/`

---
## Git Workflow Information
**Feature Branch:** <Value of $feature_branch from logic above>
**Baseline Branch:** <Value of $baseline_branch from logic above>
**Baseline Commit Hash:** <Value of $baseline_hash from logic above>
**Last Phase Commit Hash:** <Value of $baseline_hash from logic above>
---

**Created:** <Current Date, e.g., 2025-07-20>
**Core Technologies:** Python, NumPy, TensorFlow, scikit-image

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

- **`plan.md`** - The high-level R&D Plan
  - **`implementation.md`** - This file - The Phased Implementation Plan
    - `phase_1_checklist.md` - Detailed checklist for Phase 1
    - `phase_2_checklist.md` - Detailed checklist for Phase 2
    - `phase_final_checklist.md` - Checklist for the Final Phase

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** <Synthesize a one-sentence summary from the R&D Plan's objective.>

**Total Estimated Duration:** <Sum of phase estimates, e.g., 3 days>

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Logic Implementation**

**Goal:** To implement the foundational data structures and core algorithms for the new feature.

**Deliverable:** A new module `src/core/new_feature.py` with passing unit tests for all public functions.

**Estimated Duration:** 1 day

**Key Tasks:**
- Create the data processing function `process_data()`.
- Implement the main `calculate_metric()` algorithm.
- Add comprehensive unit tests covering normal operation, edge cases, and error handling.

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** `pytest tests/core/test_new_feature.py` completes with 100% pass rate.

---

### **Phase 2: Integration with Main Application**

**Goal:** To integrate the new core logic into the main application workflow.

**Deliverable:** An updated `src/main.py` that correctly calls the new module and produces the expected output.

**Estimated Duration:** 1 day

**Key Tasks:**
- Import the new module in `src/main.py`.
- Add a command-line flag `--enable-new-feature`.
- Modify the main loop to call `new_feature.process_data()` when the flag is present.

**Dependencies:** Requires Phase 1 completion.

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:** Running `python src/main.py --enable-new-feature --input data.txt` produces a valid output file.

---

### **Final Phase: Validation & Documentation**

**Goal:** Validate the complete implementation, update all relevant documentation, and ensure performance meets requirements.

**Deliverable:** A fully tested and documented feature, ready for production use.

**Estimated Duration:** 1 day

**Key Tasks:**
- Run end-to-end integration tests using a real-world dataset.
- Verify all success criteria from the R&D plan are met.
- Update `README.md` with instructions on how to use the `--enable-new-feature` flag.
- Add a new section to `docs/features.md` detailing the new capability.

**Dependencies:** All previous phases complete.

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** All R&D plan success criteria are verified as complete.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [ ] **Phase 1:** Core Logic Implementation - 0% complete
- [ ] **Phase 2:** Integration with Main Application - 0% complete
- [ ] **Final Phase:** Validation & Documentation - 0% complete

**Current Phase:** Phase 1: Core Logic Implementation
**Overall Progress:** ░░░░░░░░░░░░░░░░ 0%

---

## 🚀 **GETTING STARTED**

1.  **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist.
2.  **Begin Implementation:** Follow the checklist tasks in order.
3.  **Track Progress:** Update task states in the checklist as you work.
4.  **Request Review:** Run `/complete-phase` when all Phase 1 tasks are done to generate a review request.

---

## ⚠️ **RISK MITIGATION**

**Potential Blockers:**
- **Risk:** The external API dependency might have a lower rate limit than expected.
  - **Mitigation:** Implement client-side caching and exponential backoff for retries.
- **Risk:** The new algorithm may be too computationally expensive.
  - **Mitigation:** Profile the code early in Phase 1 and identify optimization opportunities.

**Rollback Plan:**
- **Git:** Each phase will be a separate, reviewed commit on the feature branch, allowing for easy reverts.
- **Feature Flag:** The `--enable-new-feature` flag allows the new code to be disabled in production if issues arise.
```

### **PROJECT STATUS UPDATE**
*Update these fields in `PROJECT_STATUS.md`.*
```markdown
**Current Phase:** Phase 1: Core Logic Implementation
**Progress:** ░░░░░░░░░░░░░░░░ 0%
**Next Milestone:** A new module `src/core/new_feature.py` with passing unit tests.
**Implementation Plan:** `plans/active/<initiative-name>/implementation.md`
```
```
