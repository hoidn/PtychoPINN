### **File: `.claude/commands/customplan.md` (Revised and Hardened)**

```markdown
# Command: /customplan [initiative-description]

**Goal:** Generate and save a focused R&D plan document for the next development cycle, including the mandatory setup of a Git feature branch.

**Usage:**
- `/customplan Add multi-trial statistics to generalization study`

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**THIS COMMAND MUST FOLLOW THIS EXACT SEQUENCE:**
1.  You MUST engage the user to get a clear objective.
2.  You MUST generate a `kebab-case` initiative name.
3.  You MUST capture the baseline branch name (`main`, `master`, etc.) *before* creating a new branch.
4.  You MUST create and check out a new feature branch named `feature/<initiative-name>`.
5.  You MUST then generate the `plan.md` document.
6.  You MUST update `PROJECT_STATUS.md` with the new initiative and branch information.

**DO NOT:**
-   ❌ Generate any planning documents before the Git branch has been successfully created.
-   ❌ Proceed if any Git command fails. Report the error and stop.
-   ❌ Forget to update `PROJECT_STATUS.md`.

**EXECUTION CHECKPOINT:** Before generating the `plan.md` file, you must verify that you are on a new feature branch.

---

## 🤖 **CONTEXT: YOU ARE CLAUDE CODE**

You are Claude Code, an autonomous command-line tool. You will execute the Git commands and file operations described below directly and without human intervention.

---

## 📋 **YOUR EXECUTION WORKFLOW**

### Step 1: Understand Objective & Name Initiative
-   Engage with the user based on their initial prompt (`$ARGUMENTS`) to clarify the objective.
-   Generate a concise, descriptive, `kebab-case` name for the initiative.

### Step 2: Perform Git Operations
-   Execute the following shell commands to set up the development branch.

```bash
# 1. Generate the initiative name (e.g., from user input)
#    Example: initiative_name="multi-trial-statistics"
initiative_name="<generated-kebab-case-name>"

# 2. Get the current branch name as the baseline reference.
#    This is the branch you will diff against later.
ref_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Baseline branch identified as: $ref_branch"

# 3. Create and check out the new feature branch.
feature_branch="feature/$initiative_name"
git checkout -b "$feature_branch"

# 4. Verify successful branch creation.
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" = "$feature_branch" ]; then
    echo "✅ Successfully created and checked out new branch: $feature_branch"
else
    echo "❌ ERROR: Failed to create or switch to the new feature branch."
    exit 1
fi
```

### Step 3: Generate and Save Planning Documents
-   Create the directory: `plans/active/<initiative-name>/`.
-   Generate the content for `plan.md` using the "R&D PLAN TEMPLATE" below.
-   Save the content to `plans/active/<initiative-name>/plan.md`.

### Step 4: Update Project Status
-   Update the `PROJECT_STATUS.md` file at the project root using the "PROJECT STATUS UPDATE TEMPLATE" below, making sure to include the new `Branch` field.

### Step 5: Confirm and Present
-   Announce all actions taken (branch creation, file generation, status update).
-   Present the full content of the generated `plan.md` for the user's review.

---

## 템플릿 & 가이드라인 (Templates & Guidelines)

### **R&D PLAN TEMPLATE**
*This template is for the content of `plan.md`.*
```markdown
# R&D Plan: <Initiative Name in Title Case>

*Created: <Today's date in YYYY-MM-DD format>*

## 🎯 **OBJECTIVE & HYPOTHESIS**
... (template content remains the same as previous version) ...
---

## ✅ **VALIDATION & VERIFICATION PLAN**
... (template content remains the same as previous version) ...
---

## 📁 **File Organization**

**Initiative Path:** `plans/active/<initiative-name>/`

**Next Step:** Run `/implementation` to generate the phased implementation plan.
```

### **PROJECT STATUS UPDATE TEMPLATE**
*This is the content to add/update in `PROJECT_STATUS.md`.*
```markdown
## 📍 Current Active Initiative

**Name:** <Initiative Name in Title Case>
**Path:** `plans/active/<initiative-name>/`
**Branch:** `feature/<initiative-name>` (baseline: <ref_branch>)
**Started:** <Today's date YYYY-MM-DD>
**Current Phase:** Planning
**Progress:** ░░░░░░░░░░░░░░░░ 0%
**Next Milestone:** Generate implementation plan
**R&D Plan:** `plans/active/<initiative-name>/plan.md`
**Implementation Plan:** *To be created*
```

---

## 📊 **SAMPLE INTERACTION**

### ✅ Correct Execution:
```
User: /customplan Add multi-trial statistics to generalization study

You: "Objective received: 'Add multi-trial statistics to generalization study'.
       I will create a new initiative named 'multi-trial-statistics'.
       First, I will set up the feature branch."

     [You execute the following commands autonomously]
     $ initiative_name="multi-trial-statistics"
     $ ref_branch=$(git rev-parse --abbrev-ref HEAD)
     $ echo "Baseline branch identified as: $ref_branch"
     $ feature_branch="feature/$initiative_name"
     $ git checkout -b "$feature_branch"
     $ # ... verification command ...

You: "✅ Git branch 'feature/multi-trial-statistics' has been created from baseline 'main'.
       Now generating the R&D plan and updating the project status..."

     [You create the directory, generate plan.md, and update PROJECT_STATUS.md]

You: "✅ Initiative 'Multi-Trial Statistics' has been created.
       - **R&D Plan saved to:** `plans/active/multi-trial-statistics/plan.md`
       - **Project status has been updated.**
       - **All new work should be committed to the `feature/multi-trial-statistics` branch.**

       Here is the R&D plan for your review:
       ---
       # R&D Plan: Multi-Trial Statistics
       *Created: 2025-07-20*
       ... (full plan content) ..."
```

### ❌ What NOT to do:
```
User: /customplan Add multi-trial statistics

You: "Okay, here is the plan for that initiative:"
     [Generates plan.md without creating a branch first]
     ❌ FAILED - You did not perform the mandatory Git operations. The initiative is not properly isolated.
```
```
