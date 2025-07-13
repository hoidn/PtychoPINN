# Command: /complete-phase

**Goal:** Verify the completion of the current phase, mark it as complete, and prepare the next phase.

---
## 🚀 **EXECUTION STRATEGY**

**As the AI agent, follow these steps precisely:**

1.  **Context Priming (Read R&D Plan):** First, find and read the main R&D plan for this initiative. It will be located in the `docs/studies/` directory and named `plan_*.md`.
2.  **Find and Read the Active Implementation Plan:**
    *   Search for the file named `implementation_*.md` within the `docs/studies/` directory.
    *   **Verify the file:** Read the first few lines of the file. It **MUST** contain the header `<!-- ACTIVE IMPLEMENTATION PLAN -->`. This is the official plan file. Do not look for another one.
    *   Read the full content of this verified plan file.
3.  **Determine the Current Phase:**
    *   Scan the "PHASE TRACKING" section of the plan.
    *   Identify the phase listed next to "**Current Phase:**". This is the phase we need to verify.
4.  **VERIFY PHASE COMPLETION (CRITICAL STEP):**
    *   Find the "**Success Test:**" description for the "Current Phase" in the implementation plan.
    *   **Execute the necessary commands or read the necessary files to verify that this success test has been met.** For example, if the test is "Running a script produces a specific output," then run that script. If the test is "A file contains specific changes," then read that file.
    *   **Decision Point:**
        *   **If the Success Test passes:** Announce that the verification was successful and proceed to **Step 5**.
        *   **If the Success Test fails:** **STOP.** Announce that the current phase is not yet complete. Report the discrepancy between the expected success test and the actual state of the repository. Do not modify any files. Await further instructions from the user.
5.  **Update the Implementation Plan File (only after verification):**
    *   Use the `Edit` tool to modify the implementation plan file.
    *   Find the "Next Phase" (the first `[ ]` item after the current one).
    *   Change the status of the "Current Phase" from `[ ]` to `✅`.
    *   Update the "Current Phase" line to reflect the name of the "Next Phase".
6.  **Create the Next Phase Checklist:**
    *   If there is a "Next Phase", use the "Checklist Template" to generate its checklist and save it to `docs/studies/`.
    *   If all phases are now complete, proceed to **Step 7**.
7.  **Engage User for Next Steps (if all phases are complete):**
    *   Announce that all planned phases are complete.
    *   Use the "END-OF-PLAN ENGAGEMENT SCRIPT" to ask the user for the next objective.

---

## 🤔 **END-OF-PLAN ENGAGEMENT SCRIPT**

**(Only run this if all planned phases are complete)**

**Ask the user:**

1.  "All planned phases for the current objective are complete! What is the next major objective or area for improvement?"

2.  **Present relevant categories with examples:**
    *   ✨ **New Capability/Algorithm:** _e.g., "Implement a new baseline model for comparison," "Add support for a different type of experimental data."_
    *   ⚡ **Performance & Optimization:** _e.g., "Refactor the data loading to be faster," "Reduce the memory footprint of the model."_
    *   🔧 **Data Pipeline & Tooling:** _e.g., "Create a new tool for data visualization," "Automate the data preparation steps."_
    *   ✅ **Validation & Verification:** _e.g., "Add more sophisticated evaluation metrics (like FRC)," "Create a comprehensive regression test suite for the simulation module."_
    *   ⚙️ **Refactoring & Technical Debt:** _e.g., "Remove the legacy `params.cfg` system completely," "Refactor `model.py` to be more modular."_

---

## 📋 **CHECKLIST TEMPLATE FOR NEXT PHASE**

**(Use this template to generate the content for the new `phase_N_checklist.md` file)**

```markdown
### **Agent Implementation Checklist: <Name of Phase>**

**Overall Goal for this Phase:** <Copy the "Goal" from the main implementation plan for this phase>

**Instructions for Agent:**
1.  Copy this checklist into your working memory.
2.  Update the `State` for each item as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).
3.  Follow the `How/Why & API Guidance` column carefully for implementation details.

---

| ID  | Task Description                                   | State | How/Why & API Guidance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               -
| :-- | :------------------------------------------------- | :---- | :-------------------------------------------------
| **Section 0: Preparation & Context Priming**
| 0.A | **Review Key Documents & APIs**                    | `[ ]` | **Why:** To load the necessary context and technical specifications before coding. <br> **Docs:** `<Link to R&D Plan>`, `<Link to relevant DEVELOPER_GUIDE.md section>`. <br> **APIs:** `<e.g., ptycho.tf_helper.reassemble_position>`, `<e.g., numpy.allclose>`.
| 0.B | **Identify Target Files for Modification/Creation**| `[ ]` | **Why:** To have a clear list of files that will be touched during this phase. <br> **Files:** `<e.g., src/diffusepipe/voxelization/global_voxel_grid.py (Modify)>`, `<e.g., tests/voxelization/test_global_voxel_grid.py (Create)>`.
| **Section 1: <Logical Group of Tasks, e.g., Core Logic Implementation>**
| 1.A | **<First specific, small task>**                   | `[ ]` | **Why:** <Reason for this task>. <br> **How:** <Specific implementation details, function signatures, or code snippets>. <br> **File:** `<path/to/relevant/file.py>`.
| ... | ...                                                | ...   | ...
| **Section 2: <Another Logical Group, e.g., Unit Testing>**
| 2.A | **Test <Feature from 1.A>**                        | `[ ]` | **Why:** To verify the correctness of the first piece of logic. <br> **How:** <Describe the test case. e.g., "Create a test with mock inputs where X=Y and assert the output is Z.">. <br> **File:** `<path/to/test_file.py>`.
| ... | ...                                                | ...   | ...
| **Section N: Finalization**
| N.A | **Code Formatting & Linting**                      | `[ ]` | **Why:** To maintain code quality. <br> **How:** Run the project's standard formatters (e.g., Black) and linters (e.g., Ruff) on all modified files.
| N.B | **Update Documentation**                           | `[ ]` | **Why:** To keep project documentation in sync with the code. <br> **How:** <e.g., "Add a docstring to the new function. Update the `DEVELOPER_GUIDE.md` to reflect the new API.">
