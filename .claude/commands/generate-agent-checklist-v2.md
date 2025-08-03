# Command: /generate-agent-checklist-gemini

**Goal:** Autonomously generate and execute a plan to ensure every project module has a high-quality, standardized docstring. This involves creating new docstrings where missing and improving existing ones.

**Usage:**
`/generate-agent-checklist-gemini "I want every .py module in the ptycho library to have a public-interface-focused docstring."`

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**YOUR ROLE IS AN AUTONOMOUS ORCHESTRATOR AND FILE MANAGER. YOU DO NOT PERFORM THE ANALYSIS.**
1.  You MUST gather the user's high-level objective from the command arguments.
2.  You MUST run `repomix` to create a complete, fresh snapshot of the codebase context.
3.  You MUST build a structured prompt file (`doc-plan-prompt.md`) to delegate the analysis and planning to Gemini.
4.  You MUST execute `gemini -p "@doc-plan-prompt.md"`.
5.  You MUST parse Gemini's structured response to create the three critical state files: `modules_prioritized.txt`, `dependency_report.txt`, and `docstring_progress.md`.
6.  You MUST then proceed to Phase 2 (Sub-Agent Orchestration) using these Gemini-generated files as the source of truth.

**DO NOT:**
-   ❌ Run `find`, `pydeps`, or any other manual analysis tool. Gemini is now responsible for all of Phase 1.
-   ❌ Create the plan yourself. Your job is to run the process and manage the state.
-   ❌ Proceed to Phase 2 if Gemini's analysis fails or returns an invalid format.

---

## 🤖 **YOUR EXECUTION WORKFLOW**

### **Phase 1: Gemini-Powered Strategic Analysis**

In this phase, you'll execute commands to have Gemini analyze the codebase and generate a prioritized plan for adding docstrings.

#### **Step 1.A: Aggregate Codebase Context**

First, create a complete snapshot of the project for Gemini to analyze. This uses repomix to gather all relevant Python and markdown files while excluding unnecessary content like notebooks, review files, and archived plans.

```bash
# The user's high-level objective is in $ARGUMENTS
npx repomix@latest . \
  --include "ptycho/**/*.py,*.md,docs/**/*.md,.claude/**/*.md,plans/active/**/*.md" \
  -i "**/*.ipynb,build/**,node_modules/**,dist/**,*.lock,**/review_request*.md,plans/archive/**,plans/examples/**,tensorflow/**"

if [ ! -s ./repomix-output.xml ]; then
    echo "❌ ERROR: Repomix failed to generate the codebase context. Aborting."
    exit 1
fi
echo "✅ Codebase context aggregated into repomix-output.xml."
```

#### **Step 1.B: Build Gemini Prompt File**

Create a structured prompt file that delegates the analysis task to Gemini. This uses an append-only approach to avoid complex string substitutions.

```bash
# Start building the prompt file
cat > ./doc-plan-prompt.md << 'EOF'
<task>
You are an expert Staff Engineer. Your task is to analyze an entire codebase and create a prioritized plan for adding module-level docstrings.

<steps>
<1>
Analyze the <user_notes> and the full <codebase_context>. Consider architecture.md, data contracts, and other relevant documents. Map out high-level component relationships, 
data flow, and any other relevant information.
</1>
<2>
Identify all target Python modules (.py files in the ptycho/ directory, excluding __init__.py).
</2>
<3>
Analyze the dependencies between these modules to determine a safe documentation order. Foundational modules (those with few dependencies) should come first. 
Generate a detailed module-level dependency graph and summaries of key workflows. Be particularly precise about type and shape of data flowing between modules. Be particualarly
precise about any state-dependencies.
</3>
<4>
Generate a prioritized list of modules and a summary of their dependencies, strictly adhering to the format specified in <output_format>. For each module, include a short description of its role, key public APIs, and behavior. 
</4>
</steps>

<user_notes>
EOF

# Append the user's objective
echo "$ARGUMENTS" >> ./doc-plan-prompt.md

# Append closing tags
echo "</user_notes>" >> ./doc-plan-prompt.md

echo "" >> ./doc-plan-prompt.md

echo "<output_format>" >> ./doc-plan-prompt.md
echo "---ANALYSIS---" >> ./doc-plan-prompt.md
echo "[your analysis of the codebase, component relationships, and any other relevant information.]" >> ./doc-plan-prompt.md
echo "---PRIORITIZED_MODULES_START---" >> ./doc-plan-prompt.md
echo "[List of module file paths, one per line, sorted from least to most dependent.]" >> ./doc-plan-prompt.md
echo "---PRIORITIZED_MODULES_END---" >> ./doc-plan-prompt.md
echo "" >> ./doc-plan-prompt.md
echo "---DEPENDENCY_REPORT_START---" >> ./doc-plan-prompt.md
echo '[A human-readable summary of key dependencies. For example: "ptycho/loader.py depends on ptycho/raw_data.py, ptycho/tf_helper.py"]' >> ./doc-plan-prompt.md
echo "---DEPENDENCY_REPORT_END---" >> ./doc-plan-prompt.md
echo "</output_format>" >> ./doc-plan-prompt.md
echo "</task>" >> ./doc-plan-prompt.md

echo "<codebase_context>" >> ./doc-plan-prompt.md

# Append the repomix output
cat ./repomix-output.xml >> ./doc-plan-prompt.md

# Append the closing context and output format
echo "</codebase_context>" >> ./doc-plan-prompt.md

echo "✅ Successfully built structured prompt file: ./doc-plan-prompt.md"
```

#### **Step 1.C: Execute Gemini Analysis**

Run Gemini with the prompt file to generate the prioritized module list and dependency report.

```bash
# Execute Gemini with the fully-formed prompt file and capture response
GEMINI_RESPONSE=$(gemini -p "@./doc-plan-prompt.md") || {
    echo "❌ ERROR: Gemini command failed"
    exit 1
}

# Save the raw response for debugging if needed
echo "$GEMINI_RESPONSE" > ./gemini_response_raw.txt
```

#### **Step 1.D: Create State Files from Gemini's Output**

Parse Gemini's structured response to create the three state files that will guide Phase 2.

```bash
# Parse the Gemini response that was captured in the previous step

# Create modules_prioritized.txt
awk '/---PRIORITIZED_MODULES_START---/,/---PRIORITIZED_MODULES_END---/' ./gemini_response_raw.txt | sed '1d;$d' > modules_prioritized.txt

# Create dependency_report.txt
awk '/---DEPENDENCY_REPORT_START---/,/---DEPENDENCY_REPORT_END---/' ./gemini_response_raw.txt | sed '1d;$d' > dependency_report.txt

# Verify that the files were created
if [ ! -s modules_prioritized.txt ]; then
    echo "❌ ERROR: Gemini failed to return a prioritized module list. Aborting."
    exit 1
fi

# Create the progress tracker from the prioritized list
( echo "# Docstring Progress Tracker"; echo ""; cat modules_prioritized.txt | while read -r line; do echo "- [ ] \`$line\`"; done ) > docstring_progress.md

echo "✅ Gemini analysis complete. State files created:"
echo "  - modules_prioritized.txt"
echo "  - dependency_report.txt"
echo "  - docstring_progress.md"
```

---

### **Phase 2: Sub-Agent Orchestration**

*(You will now begin the execution loop, delegating tasks to sub-agents with the updated, smarter instructions.)*

| ID | Task Description | State | How/Why & API Guidance |
| :-- | :--- | :--- | :--- |
| 2.A | **Orchestrate Documentation of Each Module** | `[ ]` | **Why:** To process each module independently by delegating to specialized sub-agents. <br> **How:** Begin a loop. For each file path in `modules_prioritized.txt`: <br> 1. **Invoke a new, single-purpose "Authoring Sub-Agent."** <br> 2. Provide it with the updated instructions from the **"Sub-Agent Instructions: Docstring Authoring (v6)"** section. <br> 3. Pass the specific module's file path and the `dependency_report.txt` file as context. <br> 4. After the sub-agent successfully completes, mark the corresponding item as done in `docstring_progress.md` and proceed to the next module in the loop. |

---

### **Phase 3: Final Verification & Commit**

*(You will execute these final steps after the loop in Phase 2 is complete.)*

| ID | Task Description | State | How/Why & API Guidance |
| :-- | :--- | :--- | :--- |
| 3.A | **Verify All Modules are Documented** | `[ ]` | **Why:** To ensure no modules were missed by the sub-agents. <br> **How:** Run a script that reads `modules_prioritized.txt` and checks that each file now starts with a `"""` docstring. |
| 3.B | **Invoke Verification Sub-Agent** | `[ ]` | **Why:** To ensure docstrings are consistent and architecturally sound. <br> **How:** Invoke a final "Verification Sub-Agent" with the instructions from the **"Sub-Agent Instructions: Final Verification"** section. |
| 3.C | **Run Automated Style Linting** | `[ ]` | **Why:** To enforce a consistent documentation style. <br> **How:** Install and run `pydocstyle`. <br> ```bash <br> pip install pydocstyle && pydocstyle ptycho/ <br> ``` |
| 3.D | **Final Code Commit** | `[ ]` | **Why:** To save the completed documentation work. <br> **How:** Stage all the modified Python files and commit them with a detailed message reflecting the new capability. <br> ```bash <br> git add ptycho/**/*.py <br> git commit -m "docs: Add or improve module-level docstrings via AI agent" -m "Ensures all core library modules have a high-quality, standardized docstring. Creates new docstrings where missing and refactors existing ones to meet project standards." <br> ``` |

---

### **Sub-Agent Instructions: Docstring Authoring (v6)**

*(Orchestrator: You will provide these new, smarter instructions to each sub-agent you invoke in Phase 2.A.)*

**Your Goal:** Ensure the specified Python module has a single, high-quality, developer-focused module-level docstring that adheres to the project's standards. This involves either **creating a new docstring** if one is missing, or **reviewing and improving an existing one**.

**Your Context:**
*   **Target Module:** `<path/to/module.py>` (Its full content is available in the `repomix` context)
*   **Dependency Report:** `ptycho/dependency_report.txt`

**Your Workflow:**

**1. Assessment & Triage:**
   - **Action:** Examine the source code of the Target Module.
   - **Check:** Does a module-level docstring (one that starts the file, using `"""` or `'''`) already exist?

**2.a. If Docstring is Missing (Creation Workflow):**
   - **Analysis:** Perform dependency analysis to define the module's public API and its consumers.
   - **Drafting:** Write a new docstring from scratch, strictly adhering to the **"Hardened Docstring Template"**.
   - **Verification:** Run a script to ensure your new docstring is under the 15% size limit. Refactor for conciseness if needed.
   - **Finalization:** Insert the new docstring at the top of the target file.

**2.b. If Docstring Exists (Review & Refactor Workflow):**
   - **Analysis:** Critically evaluate the existing docstring against the principles, the **"Hardened Docstring Template"**, and the **"Docstring Anti-Patterns"**.
   - **Identify Gaps:** Determine what is missing or incorrect. Does it lack a usage example? Is the architectural role unclear? Does it fail to mention data contracts?
   - **Refactor:** Create a new, improved version of the docstring.
     - You **MUST** preserve any valuable, accurate information from the original.
     - You **MUST** fix all identified gaps and anti-patterns.
     - The final output **MUST** be 100% compliant with the "Hardened Docstring Template," regardless of the original's structure.
   - **Verification & Replacement:** Ensure the refactored docstring meets the 15% size limit, then replace the old docstring in the file with your new, improved version.

---

### **Hardened Docstring Template (for Authoring Sub-Agent)**

*Your docstring must follow the structure and quality of the examples below. Choose the example that best fits the nature of the module you are documenting.*

---
#### **Example 1: For Modules with Complex Logic & Hidden State (e.g., `raw_data.py`)**

```python
"""
Ptychography data ingestion and scan-point grouping.

This module serves as the primary ingestion layer for the PtychoPINN data pipeline.
It is responsible for taking raw ptychographic data and wrapping it in a `RawData` object.
Its most critical function, `generate_grouped_data()`, assembles individual scan
points into physically coherent groups for training.

Architecture Role:
    Raw NPZ file -> raw_data.py (RawData) -> Grouped Data Dict -> loader.py
"""

"""
Public Interface:
    `RawData.generate_grouped_data(N, K=4, nsamples=1, ...)`
        - Purpose: The core function for sampling and grouping scan points.
        - Critical Behavior (Conditional on `params.get('gridsize')`):
            - **If `gridsize == 1`:** Performs simple sequential slicing.
            - **If `gridsize > 1`:** Implements a robust "group-then-sample"
              strategy to avoid spatial bias.
        - Key Parameters:
            - `nsamples` (int): For `gridsize=1`, this is the number of images.
              For `gridsize>1`, this is the number of *groups*.
"""

"""
Workflow Usage Example:
    ```python
    from ptycho.raw_data import RawData
    from ptycho import params

    # 1. Instantiate RawData from a raw NPZ file's contents.
    raw_data = RawData(xcoords=data['xcoords'], ...)

    # 2. Set the external state that controls the module's behavior.
    params.set('gridsize', 2)

    # 3. Generate the grouped data dictionary.
    grouped_data_dict = raw_data.generate_grouped_data(N=64, nsamples=1000)
    ```
"""

"""
Architectural Notes & Dependencies:
- This module has a critical implicit dependency on the global `params.get('gridsize')`
  value, which completely changes its sampling algorithm.
- It automatically creates a cache file (`*.groups_cache.npz`) to accelerate
  subsequent runs.
"""
```

---
#### **Example 2: For Modules Defined by Data/Tensor Transformations (e.g., `tf_helper.py`)**

```python
"""
Low-level TensorFlow operations for ptychographic data manipulation.

This module provides a suite of high-performance, tensor-based functions for
the core computational tasks in the PtychoPINN pipeline, primarily patch
extraction, reassembly, and tensor format conversions. It is a foundational
library used by the data pipeline, model, and evaluation modules.
"""

"""
Key Tensor Formats:
This module defines and converts between three standard data layouts for batches
of ptychographic patches:

- **Grid Format:** `(B, G, G, N, N, 1)`
  - Represents patches organized in their spatial grid structure.
- **Channel Format:** `(B, N, N, G*G)`
  - Stacks patches in the channel dimension. Required for CNN input.
- **Flat Format:** `(B*G*G, N, N, 1)`
  - Each patch is an independent item in the batch.
"""

"""
Public Interface:
    `reassemble_position(obj_tensor, global_offsets, M=10)`
        - **Purpose:** The primary function for stitching patches back into a full
          object image based on their precise, non-uniform scan coordinates.
        - **Algorithm:** Uses a batched shift-and-sum operation with automatic
          memory management for large datasets.
        - **Parameters:**
            - `obj_tensor` (Tensor): Complex patches in `Flat Format`.
            - `global_offsets` (Tensor): The `(y, x)` scan coordinates for each patch.
            - `M` (int): The size of the central region of each patch to use for
              the reassembly, which helps avoid edge artifacts.
"""

"""
Usage Example:
    This example shows the canonical `Grid -> Channel -> Flat -> Reassembly`
    workflow that this module enables.

    ```python
    import ptycho.tf_helper as hh
    import tensorflow as tf

    # 1. Start with data in Grid Format. Shape: (10, 2, 2, 64, 64, 1)
    patch_grid = tf.random.normal((10, 2, 2, 64, 64, 1))
    
    # 2. Convert to Channel Format for a CNN. Shape: (10, 64, 64, 4)
    patch_channels = hh.grid_to_channel(patch_grid)
    
    # ... (model processing) ...

    # 3. Convert to Flat Format for reassembly. Shape: (40, 64, 64, 1)
    patches_flat = hh.channel_to_flat(patch_channels)

    # 4. Reassemble the flat patches into a final image.
    scan_coords = tf.random.uniform((40, 1, 1, 2), maxval=100)
    reconstructed_image = hh.reassemble_position(patches_flat, scan_coords, M=20)
    ```
"""
```

---

### **Docstring Anti-Patterns (To Be Avoided by Sub-Agents)**

Your generated docstrings will be rejected if they contain the following:

*   **Vague Summaries:** Avoid generic phrases like "This module contains helper functions" or "Utilities for data processing." Be specific about its role.
*   **Marketing Language:** Do not use subjective fluff like "critical," "essential," "high-performance," or specific speedup numbers. Instead, explain *how* it is performant (e.g., "Uses a batched algorithm to manage memory").
*   **Implementation Details:** Do not explain the line-by-line logic of the code. Focus on the public contract: what goes in, what comes out, and what it's for.
*   **Isolated Examples:** Do not provide usage examples that are just a single function call with placeholder variables. The example must show a realistic interaction between modules.
*   **Inaccurate Consumer Lists:** Do not guess which modules use this one. The dependency report is the source of truth.

---

### **Sub-Agent Instructions: Final Verification**

*(Orchestrator: You will provide these instructions to the sub-agent you invoke in Phase 3.B.)*

**Your Goal:** To perform a final consistency and architectural accuracy check on all newly created or updated docstrings.

**Your Context:**
*   The list of all documented modules: `modules_prioritized.txt`
*   The full dependency map: `dependency_report.txt`
*   The PtychoPINN architecture understanding from `docs/DEVELOPER_GUIDE.md` and `docs/architecture.md`

**Your Workflow:**
1.  **Read All Docstrings:** Load the module-level docstring from every file listed in `modules_prioritized.txt`.
2.  **Cross-Reference Architecture Claims:** For each docstring:
    *   Verify "primary consumers" claims against actual dependency data in `ptycho/dependency_report.txt`.
    *   Check that architectural role descriptions align with the actual system design.
    *   Validate that workflow examples show realistic integration patterns.
3.  **Identify Inconsistencies:**
    *   Module claims to be used by X, but dependency report shows no such link.
    *   Usage examples show patterns not actually used in the codebase.
    *   Circular or contradictory architectural role descriptions.
    *   Incorrect data flow or integration claims.
4.  **Generate Report:** Create `docstring_consistency_report.md` with:
    *   **Pass/Fail Summary:** Overall assessment.
    *   **Inconsistencies Found:** Specific issues requiring fixes.
    *   **Architecture Accuracy:** Assessment of architectural claims.
    *   **Recommendations:** Suggested improvements for consistency.
5.  **Report Findings:** Return the path to the generated report. The Orchestrator will decide if fixes are needed before proceeding.


