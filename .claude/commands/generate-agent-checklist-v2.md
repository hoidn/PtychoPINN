# Command: /generate-agent-checklist-gemini

**Goal:** Autonomously generate and execute a plan to ensure every project module has a high-quality, standardized docstring. This involves creating new docstrings where missing and improving existing ones.

**Usage:**
`/generate-agent-checklist-gemini "I want every .py module in the ptycho library to have a public-interface-focused docstring."`

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**YOUR ROLE IS AN AUTONOMOUS ORCHESTRATOR AND FILE MANAGER. YOU DO NOT PERFORM THE ANALYSIS.**
1.  You MUST gather the user's high-level objective from the command arguments.
2.  You MUST run `repomix` to create a complete, fresh snapshot of the codebase context.
3.  You MUST build a structured prompt file (`tmp/doc-plan-prompt.md`) to delegate the analysis and planning to Gemini.
4.  You MUST execute `gemini -p "@tmp/doc-plan-prompt.md"`.
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
  -i "**/*.ipynb,build/**,node_modules/**,dist/**,*.lock,**/review_request*.md,plans/archive/**,plans/examples/**,tensorflow/**,tmp/**"

if [ ! -s ./repomix-output.xml ]; then
    echo "❌ ERROR: Repomix failed to generate the codebase context. Aborting."
    exit 1
fi
echo "✅ Codebase context aggregated into repomix-output.xml."
```

#### **Step 1.B: Build Gemini Prompt File**

Create a structured prompt file that delegates the analysis task to Gemini. This uses an append-only approach to avoid complex string substitutions.

```bash
# This script builds a highly-structured prompt file for Gemini to generate a
# phased, strategic plan for adding module-level docstrings.

# The prompt is constructed incrementally to safely handle user input and large
# context files.

# --- Step 1: Start building the prompt file with the core task and persona ---
cat > ./tmp/doc-plan-prompt.md << 'EOF'
<task>
You are an expert Staff Engineer specializing in large-scale codebase analysis and technical documentation strategy. Your primary skills are in understanding system architecture, data flow, state management, and API contracts.

Your task is to perform an architectural deep-dive of the provided codebase and generate a comprehensive, prioritized documentation strategy. Your final report must be structured as a phased, actionable work plan.

<persona>
- You think in terms of data contracts, not just function calls.
- You are obsessed with identifying and documenting state dependencies (e.g., global configurations) because they are a primary source of bugs.
- You prioritize clarity in the public API of each module.
- You believe a documentation plan is only as good as the analysis that underpins it.
</persona>

<thinking_workflow>
To produce the final report, you MUST follow this internal thought process:

1.  **Full Analysis:** Systematically enumerate and analyze every target Python module (`.py` files in `ptycho/`, excluding `__init__.py`). For each module, gather the details required by the per-module template in the output format.
2.  **Strategic Grouping:** After analyzing all modules, identify logical, thematic phases for the documentation effort (e.g., "Phase 1: Configuration & State", "Phase 2: Core Physics & Tensor Ops", "Phase 3: Data Pipeline", etc.). Assign each module to one of these strategic phases.
3.  **Final Report Generation:** Construct the final output. First, write down your strategic grouping and rationale. Then, present the detailed module-by-module plan, organized under the phase headings you just defined.
</thinking_workflow>

<user_objective>
EOF

# --- Step 2: Append the user's high-level objective from the command arguments ---
echo "$ARGUMENTS" >> ./tmp/doc-plan-prompt.md

# --- Step 3: Append the closing tag for the objective and the detailed output format instructions ---
# We use 'cat >>' to append the next large static block.
cat >> ./tmp/doc-plan-prompt.md << 'EOF'
</user_objective>

<output_format>
Your final output must be a single, comprehensive report starting with your strategic analysis, followed by the detailed, phased plan.

<analysis_and_strategy>
This section must contain your high-level strategic thinking.

**Architectural Pillars:**
[Briefly describe the 3-4 main architectural pillars you identified.]

**Proposed Documentation Phases:**
[List the strategic phases you've decided on and provide a 1-sentence rationale for each. For example:]
- **Phase 1: Configuration & Foundational Utilities:** Document the core state management and widely used helpers first, as they are dependencies for everything else.
- **Phase 2: Core Physics & Tensor Operations:** Focus on the stable, foundational modules that define the system's scientific and computational contracts.
- **Phase 3: Data Ingestion Pipeline:** Document the flow of data from raw files to model-ready tensors.
- **Phase 4: Model & Training Workflows:** Document the central model architecture and the high-level orchestrators.
</analysis_and_strategy>

<prioritized_documentation_plan>
This section must contain the detailed module-by-module plan, grouped under the phase headings you defined above. Each module entry must strictly adhere to the following Markdown format.

---
## Phase 1: Configuration & Foundational Utilities

- **File:** `[path/to/module.py]`
  - **Priority:** `[CRITICAL | HIGH | MEDIUM | LOW]`
  - **Role & Architectural Significance:** A concise summary of this module's purpose and its importance in the system's architecture.
  - **Key Public API(s):** List the most important public functions or classes.
  - **Data Flow Contract (Inputs/Outputs):**
    - **Input:** Describe the primary data consumed, including source, type, and shape.
    - **Output:** Describe the primary data produced, including destination, type, and shape.
  - **State Dependencies:** Explicitly identify any dependencies on external or global state. If none, state "None."
  - **Dependencies (Internal):** List key internal `ptycho` modules this module imports.
  - **Consumers (Internal):** List key internal `ptycho` modules that import this one.
  - **Docstring Action Plan:** `[NO_CHANGE | IMPROVE | REWRITE | CREATE]` - Provide a concrete, actionable recommendation. (e.g., "CREATE: The module is undocumented.", "IMPROVE: Add a workflow example showing interaction with `loader.py`.", "REWRITE: The current docstring is outdated and incorrectly describes the algorithm. Must update to reflect the 'sample-then-group' logic.").

## Phase 2: Core Physics & Tensor Operations
... (and so on for each phase and module)
</prioritized_documentation_plan>
</output_format>

<gold_standard_example>
Here is an example of the expected quality and detail for a single module entry under a phase heading. Your analysis for every module must match this level of depth.

## Phase 3: Data Ingestion Pipeline

- **File:** `ptycho/raw_data.py`
  - **Priority:** `HIGH`
  - **Role & Architectural Significance:** The primary data ingestion layer. Its key architectural role is to abstract away raw file formats and enforce physical coherence of scan positions *before* they enter the main ML pipeline, which is crucial for the validity of `gridsize > 1` training.
  - **Key Public API(s):** `RawData` class, `RawData.generate_grouped_data()`.
  - **Data Flow Contract (Inputs/Outputs):**
    - **Input:** Raw `.npz` files containing NumPy arrays. Key arrays include `'diffraction'` with shape `(num_scans, N, N)` and `'xcoords'`/`'ycoords'` with shape `(num_scans,)`.
    - **Output:** A dictionary of grouped NumPy arrays consumed by `ptycho.loader`. The output shape is critically dependent on `params.get('gridsize')`:
      - **If `gridsize > 1`**: Arrays are in "Channel Format", e.g., the `X` (diffraction) array has shape `(nsamples, N, N, gridsize**2)`.
      - **If `gridsize == 1`**: Arrays represent individual patches, e.g., the `X` (diffraction) array has shape `(nsamples, N, N, 1)`.
  - **State Dependencies:** Critically dependent on the global `ptycho.params.get('gridsize')` which algorithmically changes its grouping strategy from sequential slicing to a robust "sample-then-group" method.
  - **Dependencies (Internal):** `ptycho.params`, `ptycho.config.config`, `ptycho.tf_helper`.
  - **Consumers (Internal):** `ptycho.loader`, `ptycho.workflows.components`.
  - **Docstring Action Plan:** `REWRITE` - The current docstring is outdated. It incorrectly describes the algorithm as "group-then-sample" and must be updated to reflect the current, performance-optimized "sample-then-group" logic. The critical state dependency on `gridsize` and the conditional output shapes must also be explicitly documented.
</gold_standard_example>
EOF

# --- Step 4: Add the opening tag for the codebase context ---
echo "<codebase_context>" >> ./tmp/doc-plan-prompt.md

# --- Step 5: Append the full codebase context from the repomix output file ---
cat ./repomix-output.xml >> ./tmp/doc-plan-prompt.md

# --- Step 6: Append the closing tags to finalize the prompt file ---
echo "</codebase_context>" >> ./tmp/doc-plan-prompt.md
echo "</task>" >> ./tmp/doc-plan-prompt.md

echo "✅ Successfully built structured prompt file: ./tmp/doc-plan-prompt.md"
```

#### **Step 1.C: Execute Gemini Analysis**

Run Gemini with the prompt file to generate the prioritized module list and dependency report.

```bash
# Execute Gemini with the fully-formed prompt file and capture response
GEMINI_RESPONSE=$(gemini -p "@./tmp/doc-plan-prompt.md") || {
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
| 2.A | **Orchestrate Documentation of Each Module** | `[ ]` | **Why:** To process each module independently by delegating to specialized sub-agents. <br> **How:** Begin a loop. For each file path in `modules_prioritized.txt`: <br> 1. **Invoke a new, single-purpose "Authoring Sub-Agent."** <br> 2. Provide it with the FULL instructions from the **"Sub-Agent Instructions: Docstring Authoring (v6)"** section, including Gemini's recommentation / analysis relevant to that module. <br> 3. Pass the specific module's file path and the `dependency_report.txt` file as context. <br> 4. After the sub-agent successfully completes, mark the corresponding item as done in `docstring_progress.md` and proceed to the next module in the loop. |

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
*   **Additional Context:** [Gemini's analysis and recommendations for this module]

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
     - You **MUST** improve the docstring, unless you believe it is already perfect.
     - You should ensure the docstring is no longer than 15% of the original file size, but occasionally exceeding this limit is acceptable
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
*   **Marketing Language:** Do not use subjective fluff like "critical," "essential," "high-performance," or specific speedup numbers. Instead, explain *how* it is performant (e.g., "Uses a batched algorithm to manage memory"). Words you are NEVER allowed to use: "critical," "essential," "high-performance," "fast," "efficient," "optimized," "comprehensive", you get the idea.
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


