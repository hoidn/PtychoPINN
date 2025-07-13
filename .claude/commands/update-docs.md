# Command: /update-docs

**Goal:** Review the most recent code changes and update all relevant project documentation to reflect them, incorporating user-provided suggestions.

**Usage:** `/update-docs [optional suggested edits]`

**Example:**
`/update-docs The main change was adding the --n-train-images flag to run_comparison.sh. We should update its docstring and also create a new README in the scripts/studies/ directory to explain the new generalization study workflow.`

**User's suggested changes and edits:**
$ARGUMENTS

---
## 🚀 **EXECUTION STRATEGY**

**As the AI agent, follow this documentation update checklist precisely:**

1.  **Identify Recent Changes & User Suggestions:**
    *   First, check if the user provided any suggestions above (these came from the `/update-docs` command arguments).
    *   If no arguments were provided, review the most recent code changes or ask the user: "What specific feature or change do I need to document?"
2.  **Analyze Impact:** Based on the recent changes and the user's suggestions, analyze which parts of the project are affected.
3.  **Systematically Review and Update Documentation:** Go through the checklist in the "DOCUMENTATION REVIEW CHECKLIST" section below. For each item, check if an update is needed, paying close attention to the user's suggestions. If an update is needed, perform it. If not, state that you've checked it and no changes were required.

---

## **DOCUMENTATION REVIEW CHECKLIST**

### **1. High-Level Project Guides**

*   [ ] **`README.md` (Root Level):**
    *   **Check:** Does the new feature change the primary installation or usage instructions? Is it a major new capability that should be highlighted in the overview?
    *   **Action (if needed):** Update the "Usage" or "Features" sections.

*   [ ] **`docs/DEVELOPER_GUIDE.md`:**
    *   **Check:** Does the change introduce a new architectural principle, a critical workflow, or a lesson learned (an "anti-pattern")? Does it affect the data pipeline or evaluation methods?
    *   **Action (if needed):** Add a new section or update an existing one to reflect the new best practices or architectural components.

*   [ ] **`CLAUDE.md`:**
    *   **Check:** Does the change introduce a new core directive for the AI? Does it change how the AI should interact with the codebase or data formats?
    *   **Action (if needed):** Add or update a directive. _e.g., "You MUST now use the `new_function()` for all evaluations."_

### **2. Tool and Script Documentation**

*   [ ] **Script `README.md` Files:**
    *   **Check:** Was a new script directory created (e.g., `scripts/studies/`)? Does it have a `README.md` explaining its purpose and workflow?
    *   **Action (if needed):** Create or update the `README.md` for the relevant directory (e.g., `scripts/tools/README.md`, `scripts/studies/README.md`). Provide clear usage examples.

*   [ ] **Shell Script Header Comments:**
    *   **Check:** Were any shell scripts (`.sh`) created or modified?
    *   **Action (if needed):** Update the header comments in the script to explain its purpose, arguments, and provide up-to-date usage examples.

### **3. Code-Level Documentation (Docstrings)**

*   [ ] **Python Module Docstrings:**
    *   **Check:** Were any new Python files (`.py`) created?
    *   **Action (if needed):** Add a module-level docstring at the top of the new file explaining its overall purpose and the tools it provides.

*   [ ] **Function and Class Docstrings:**
    *   **Check:** Were any new public functions or classes added or modified?
    *   **Action (if needed):** Add or update the docstrings to clearly explain the purpose, arguments (`Args:`), return values (`Returns:`), and any errors raised (`Raises:`). Include a simple usage example if helpful.

### **4. Data Format Documentation**

*   [ ] **`docs/data_contracts.md`:**
    *   **Check:** Did the changes introduce a new data format, add/remove keys from an NPZ file, or change the shape/type of any arrays?
    *   **Action (if needed):** Update the data contracts document to reflect the new canonical format. This is a critical step.

### **5. Final Review**

*   [ ] **Consistency Check:**
    *   **Check:** Read through all the changes you've made. Is the terminology consistent? Do the examples in different documents align with each other?
    *   **Action (if needed):** Correct any inconsistencies.
