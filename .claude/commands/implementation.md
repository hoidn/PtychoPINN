# Command: /implementation

**Goal:** Generate and save a phased implementation plan document based on the most recent R&D plan.

---
## ⚠️ **IMPORTANT: YOUR TASK AS THE AI AGENT**

Your **only task** is to **GENERATE A MARKDOWN DOCUMENT AND SAVE IT TO A FILE**. Do **NOT** execute any of the steps described in the plan you create.

Your process is:
1.  **Read the R&D Plan:** First, read the most recent R&D plan document (e.g., `docs/studies/plan_model_generalization.md`) to fully understand the project's objective.
2.  **Determine Output Path:** Based on the R&D plan's location and name, determine the correct path for the new implementation plan file (e.g., `docs/studies/implementation_<initiative_name>.md`).
3.  **Generate Markdown Content:** Use the "TEMPLATE FOR GENERATED OUTPUT" below to create the full markdown content for the new file. **Crucially, ensure the `<!-- ACTIVE IMPLEMENTATION PLAN -->` header is included at the very top.**
4.  **Save the File:** Use the `Edit` tool to create and save the generated markdown content to the path you determined in Step 2.
5.  **Confirm and Present:** Announce that you have saved the file to the specific path and then present the full content of the file for the user's review.

---
## **TEMPLATE FOR GENERATED OUTPUT**
**(Use this template to create the `implementation_<initiative_name>.md` file)**

```markdown
<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** <Copy from R&D Plan>
*   _e.g., "Model Generalization Study: Performance vs. Training Set Size"_

**Core Technologies:** <Copy from R&D Plan or infer>
*   _e.g., "Bash, Python, Pandas, Matplotlib"_

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

*   **`<path/to/plan_doc.md>`** (The high-level R&D Plan)
    *   **`implementation_<initiative_name>.md`** (This file - The Phased Implementation Plan)
        *   `phase_1_checklist.md` (Detailed checklist for Phase 1)
        *   `phase_2_checklist.md` (Detailed checklist for Phase 2)
        *   `phase_3_checklist.md` (Detailed checklist for Phase 3)

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** <A one-sentence summary of the final state after all phases are complete>
*   _e.g., "To create a fully automated workflow for studying model generalization by comparing performance vs. training set size."_

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: <Name of the Foundational Component>**
*   _e.g., "Enhance Single-Run Capability"_

**Goal:** <A concise description of this phase's objective>
*   _e.g., "To update the core `run_comparison.sh` script to support configurable training and testing set sizes."_

**Deliverable:** <The specific, tangible output of this phase>
*   _e.g., "A modified `run_comparison.sh` that correctly accepts and utilizes `--n-train-images` and `--n-test-images` arguments."_

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] phase_1_checklist.md`

**Key Tasks Summary:**
*   <A high-level summary of the main tasks for this phase>
    *   _e.g., "Add named argument parsing to `run_comparison.sh`."_
    *   _e.g., "Forward new arguments to the underlying Python scripts."_
    *   _e.g., "Verify Python scripts correctly use the `n_images` parameter."_

**Success Test:** <How you know this phase is successfully completed>
*   _e.g., "All tasks in `phase_1_checklist.md` are marked as done. Running `run_comparison.sh --n-train-images 512` successfully completes and logs show only 512 images were used for training."_

**Duration:** <Estimated time, e.g., 1 day>

---

### **Phase 2: <...>**
*   ...

---

## 📝 **PHASE TRACKING**

- [ ] **Phase 1:** <Name of Phase 1> (see `phase_1_checklist.md`)
- [ ] **Phase 2:** <Name of Phase 2> (see `phase_2_checklist.md`)
- [ ] **Phase 3:** <Name of Phase 3> (see `phase_3_checklist.md`)

**Current Phase:** Phase 1: <Name of Phase 1>
**Next Milestone:** <A description of the deliverable for the current phase>
*   _e.g., "A modified `run_comparison.sh` that accepts and uses `--n-train-images` and `--n-test-images`."_
