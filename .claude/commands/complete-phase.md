### **File: `.claude/commands/complete-phase.md` (Revised and Hardened)**

```markdown
# Command: /complete-phase

**Goal:** Manage the end-of-phase transition using a formal review cycle. This command now operates in two distinct modes, determined by the presence of a review file.

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**You MUST operate in one of two modes. You are not allowed to mix them.**

**Mode 1: Request Review (Default)**
*   **Trigger:** No `review_phase_N.md` file exists for the current phase.
*   **Action:** You MUST generate a `review_request_phase_N.md` file containing a `git diff` and then HALT.

**Mode 2: Process Review**
*   **Trigger:** A `review_phase_N.md` file EXISTS for the current phase.
*   **Action:** You MUST read the review, parse the `VERDICT`, and then either commit the changes (on `ACCEPT`) or report the required fixes (on `REJECT`).

**DO NOT:**
-   ❌ Commit any code without a `VERDICT: ACCEPT` from a review file.
-   ❌ Generate a new review request if a review file already exists.
-   ❌ Mark a phase as complete if the verdict is `REJECT` or if the commit fails.

---

## 🤖 **CONTEXT: YOU ARE CLAUDE CODE**

You are Claude Code, an autonomous agent. You will execute the Git and file commands below to manage the phase completion and review process. You will handle all steps without human intervention.

---

## 📋 **YOUR EXECUTION WORKFLOW**

### Step 1: Determine Current Mode
-   Read `PROJECT_STATUS.md` to get the current initiative path and phase number (`N`).
-   Check if the file `<path>/review_phase_N.md` exists.
-   If it exists, proceed to **Mode 2: Process Review**.
-   If it does not exist, proceed to **Mode 1: Request Review**.

---

### **MODE 1: REQUEST REVIEW**

#### Step 1.1: Read State and Generate Diff
-   Read `<path>/implementation.md` to get the `Last Phase Commit Hash`. This is your diff base.
-   Run the following command to generate the diff. This uses a tested pattern that is robust.

```bash
# Ensure a temporary directory exists
mkdir -p ./tmp

# Extract the baseline commit hash for the diff
# Note: Using awk is a simple, tested way to extract the value
diff_base=$(grep 'Last Phase Commit Hash:' <path>/implementation.md | awk '{print $4}')

# Generate the diff against the baseline hash, excluding .ipynb files
# Using pathspec magic syntax (requires Git 1.9+)
git diff "${diff_base}"..HEAD -- . ':(exclude)*.ipynb' ':(exclude)**/*.ipynb' > ./tmp/phase_diff.txt
```

#### Step 1.2: Generate Review Request File
-   Create a new file: `<path>/review_request_phase_N.md`.
-   Populate it using the "REVIEW REQUEST TEMPLATE" below. You must embed the content of `plan.md`, `implementation.md`, `phase_N_checklist.md`, and `./tmp/phase_diff.txt` using the robust sequential `echo`/`cat` pattern.

#### Step 1.3: Notify and Halt
-   Inform the user that the review request is ready at `<path>/review_request_phase_N.md`.
-   Instruct them to have it reviewed and to create the `review_phase_N.md` file with a clear verdict.
-   **HALT.** Your task for this run is complete.

---

### **MODE 2: PROCESS REVIEW**

#### Step 2.1: Read and Parse Review File
-   Read the file `<path>/review_phase_N.md`.
-   Find the line starting with `VERDICT:`. Extract the verdict (`ACCEPT` or `REJECT`).
-   If no valid verdict is found, report an error and stop.

#### Step 2.2: 🔴 MANDATORY - Conditional Execution (On `ACCEPT`)
-   If `VERDICT: ACCEPT`, you MUST execute the following sequence of commands precisely.

```bash
# 1. Add all changes to staging
git add -A

# 2. Commit the changes for this phase
#    Note: The deliverable description should be extracted from implementation.md
phase_deliverable="<Extract Deliverable from implementation.md for the current phase>"
git commit -m "Phase N: $phase_deliverable"

# 3. Verify the commit was successful
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Git commit failed. Halting."
    exit 1
fi

# 4. Capture the new commit hash for state update
new_hash=$(git rev-parse HEAD)
echo "New commit hash is: $new_hash"
```

-   **Update State:** Modify `<path>/implementation.md`, replacing the old `Last Phase Commit Hash` with the `$new_hash`.
-   **Finalize Phase:** Mark the current phase as complete in all status documents (`implementation.md`, `PROJECT_STATUS.md`).
-   **Prepare Next Phase:** If this is not the final phase, generate the checklist for Phase N+1. If it is the final phase, archive the initiative.
-   **Report Success:** Announce that the phase was accepted, committed, and that the next phase is ready.

#### Step 2.3: Conditional Execution (On `REJECT`)
-   If `VERDICT: REJECT`, extract all lines from the "Required Fixes" section of the review file.
-   Present these fixes clearly to the user.
-   Instruct the user to address the feedback and then run `/complete-phase` again to generate a new review request.
-   **HALT.** Make no changes to Git or status files.

---

## 템플릿 & 가이드라인 (Templates & Guidelines)

### **REVIEW REQUEST TEMPLATE**
*This is the content for the agent-generated `review_request_phase_N.md`.*
```markdown
# Review Request: Phase <N> - <Phase Name>

**Initiative:** <Initiative Name>
**Generated:** <YYYY-MM-DD HH:MM:SS>

This document contains all necessary information to review the work completed for Phase <N>.

## Instructions for Reviewer

1.  Analyze the planning documents and the code changes (`git diff`) below.
2.  Create a new file named `review_phase_N.md` in this same directory (`<path>/`).
3.  In your review file, you **MUST** provide a clear verdict on a single line: `VERDICT: ACCEPT` or `VERDICT: REJECT`.
4.  If rejecting, you **MUST** provide a list of specific, actionable fixes under a "Required Fixes" heading.

---
## 1. Planning Documents

### R&D Plan (`plan.md`)
<The full content of plan.md is embedded here>

### Implementation Plan (`implementation.md`)
<The full content of implementation.md is embedded here>

### Phase Checklist (`phase_N_checklist.md`)
<The full content of the current phase_N_checklist.md is embedded here>

---
## 2. Code Changes for This Phase

**Baseline Commit:** `<Last Phase Commit Hash from implementation.md>`
**Current Branch:** `<current feature branch name>`
**Changes since last phase:**
*Note: Jupyter notebook (.ipynb) files are excluded from this diff for clarity*

```diff
<The full output of the 'git diff' command is embedded here>
```
```

### **REVIEW FILE TEMPLATE (for human reviewers)**
*This is the expected format of the human-created `review_phase_N.md`.*
```markdown
# Review: Phase <N> - <Phase Name>

**Reviewer:** <Reviewer's Name>
**Date:** <YYYY-MM-DD>

## Verdict

**VERDICT: ACCEPT**

---
## Comments

The implementation looks solid. The new module is well-tested and follows project conventions.

---
## Required Fixes (if REJECTED)

*(This section would be empty for an ACCEPT verdict)*
- **Fix 1:** In `src/module/file.py`, the error handling for `function_x` is incomplete. It must also catch `KeyError`.
- **Fix 2:** The unit test `tests/test_module.py::test_function_x_edge_case` does not assert the correct exception type.
```

---
## 📊 **SAMPLE INTERACTION**

### ✅ Correct Execution:
```
User: /complete-phase

You: "Phase 1 checklist is complete. A review file was not found, so I will now generate a review request."

     [You execute 'git diff ... -- . ':(exclude)*.ipynb' ':(exclude)**/*.ipynb'', then generate 'review_request_phase_1.md']

You: "✅ Review request for Phase 1 has been generated at:
       `plans/active/my-initiative/review_request_phase_1.md`
       Please have it reviewed. Once the review is complete and saved as
       `review_phase_1.md` with a verdict, run this command again to process it."

[... Human review happens, 'review_phase_1.md' is created with VERDICT: ACCEPT ...]

User: /complete-phase

You: "Review file for Phase 1 found. Processing review..."
     [You read 'review_phase_1.md' and find 'VERDICT: ACCEPT']
You: "Verdict is ACCEPT. Committing changes and finalizing phase."

     [You execute 'git add', 'git commit', 'git rev-parse HEAD']
     [You update 'implementation.md' with the new commit hash]
     [You update 'PROJECT_STATUS.md' and generate 'phase_2_checklist.md']

You: "✅ Phase 1 has been accepted and committed.
       - Commit hash: <new_hash>
       - The checklist for Phase 2 is now available at: `plans/active/my-initiative/phase_2_checklist.md`"
```
```
