# Command: /implementation

**Goal:** Generate and save a phased implementation plan document based on the most recent R&D plan.

---
## ⚠️ **IMPORTANT: YOUR TASK AS THE AI AGENT**

Your **only task** is to **GENERATE A MARKDOWN DOCUMENT AND SAVE IT TO A FILE**. Do **NOT** execute any of the steps described in the plan you create.

Your process is:
1.  **Read the R&D Plan:** First, read the most recent R&D plan document to fully understand the project's objective, deliverables, and core capabilities.
2.  **Decompose into Logical Phases:** Based on the R&D plan, break down the work into a logical sequence of distinct, verifiable phases. **The number of phases is NOT fixed.** A simple task might have only one implementation phase followed by the finalization phase. A complex one might have several intermediate phases. Each phase must produce a concrete deliverable.
3.  **Define a Tailored Finalization Phase:** The very last phase must always be a "Finalization" or "Validation & Documentation" phase. Its tasks should be *specific* to the changes made in the preceding phases, not a generic checklist.
4.  **Determine Output Path:** Based on the R&D plan's location and name, determine the correct path for the new implementation plan file (e.g., `docs/studies/multirun/implementation_statistical_generalization.md`).
5.  **Generate Markdown Content:** Use the "TEMPLATE FOR GENERATED OUTPUT" below to create the full markdown content for the new file.
6.  **Save the File:** Use the `Edit` tool to create and save the generated markdown content to the path you determined.
7.  **Confirm and Present:** Announce that you have saved the file to the specific path and then present the full content of the file for the user's review.

---
## **TEMPLATE FOR GENERATED OUTPUT**
**(Use this template to create the `implementation_*.md` file)**

```markdown
<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** <Copy from R&D Plan>
*   _e.g., "Statistical Generalization Study Enhancement"_

**Core Technologies:** <Copy from R&D Plan or infer>
*   _e.g., "Bash, Python, Pandas, Matplotlib"_

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

*   **`<path/to/plan_doc.md>`** (The high-level R&D Plan)
    *   **`implementation_<initiative_name>.md`** (This file - The Phased Implementation Plan)
        *   `phase_1_checklist_<initiative_name>.md` (Detailed checklist for Phase 1)
        *   ... (Checklists for any intermediate phases)
        *   `phase_final_checklist_<initiative_name>.md` (Checklist for the Final Phase)

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** <A one-sentence summary of the final state after all phases are complete>
*   _e.g., "To create a fully automated workflow for studying model generalization with statistical rigor by running multiple trials and visualizing mean performance with standard deviation."_

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: <Name of the First Logical Component>**
*   _e.g., "Multi-Trial Execution Framework"_

**Goal:** <A concise description of this phase's objective>
*   _e.g., "To update the `run_complete_generalization_study.sh` script to support configurable multi-trial runs."_

**Deliverable:** <The specific, tangible output of this phase>
*   _e.g., "A modified `run_complete_generalization_study.sh` that correctly accepts and utilizes a `--num-trials` argument."_

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_1_checklist_<initiative_name>.md`

**Success Test:** <How you know this phase is successfully completed>
*   _e.g., "All tasks in the Phase 1 checklist are marked as done. Running the script with `--num-trials 2 --dry-run` shows commands with nested `trial_1` and `trial_2` output directories."_

---

<!-- (Add 0 or more intermediate phases here as needed. If the task is simple, this section can be omitted entirely.) -->
### **Phase 2: <Name of the Second Logical Component (if needed)>**
*   ...

---

### **Final Phase: <Validation & Documentation / Finalization>**

**Goal:** To validate the complete workflow and update all relevant documentation.

**Deliverable:** A fully tested and documented feature, with project status updated.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_final_checklist_<initiative_name>.md`

**Key Tasks Summary:**
*   <A *specific* list of validation tasks based on what was changed>
    *   _e.g., "Run a small-scale end-to-end study using the new `--num-trials` flag."_
    *   _e.g., "Verify the final plot contains shaded error bands."_
*   <A *specific* list of documentation updates based on what was changed>
    *   _e.g., "Update `docs/studies/QUICK_REFERENCE.md` with the new `--num-trials` flag."_
    *   _e.g., "Update the `run_complete_generalization_study.sh` header comments."_
*   Update `PROJECT_STATUS.md` to mark the initiative as complete.

**Success Test:** All tasks in the final phase checklist are marked as done. The new feature is fully integrated and documented.

---

## 📝 **PHASE TRACKING**

- [ ] **Phase 1:** <Name of Phase 1> (see `phase_1_checklist_<initiative_name>.md`)
- ... (List any intermediate phases here)
- [ ] **Final Phase:** <Name of Final Phase> (see `phase_final_checklist_<initiative_name>.md`)

**Current Phase:** Phase 1: <Name of Phase 1>
**Next Milestone:** <A description of the deliverable for the current phase>
*   _e.g., "A modified `run_complete_generalization_study.sh` that accepts `--num-trials`."_
```
