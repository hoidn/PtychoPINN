### Generating a High-Quality Documentation Plan **

**CRITICAL: YOUR SOLE TASK IS TO GENERATE A NEW, SELF-CONTAINED IMPLEMENTATION CHECKLIST FILE NAMED `docstring_initiative_checklist.md`.**

**DO NOT MODIFY THIS PROMPT FILE. DO NOT PERFORM THE DOCUMENTATION YOURSELF. YOUR ONLY DELIVERABLE IS THE NEW CHECKLIST FILE.**

---

**Your Role and Objective:**
You are an expert Staff Software Engineer and AI Agent Orchestrator. You have been given a high-level goal from a user. Your task is to transform that goal into a detailed, executable project plan.

**User's Goal:**
> "I want every `.py` module (not counting scripts) to have a docstring <15 percent of the module size that documents the module's public interface: i.e., gives sufficient information on how that module is used / is to be used in other parts of the code or as a public api."

---

### **CRITICAL: Your Mandatory Workflow**

You must follow this three-step process to generate the content for the new checklist file.

#### **Step 1: Codebase Analysis & Dependency Mapping**

Before you can create the plan, you must deeply understand the codebase's architecture.

1.  **Define Scope:** Determine the list of all target modules (e.g., all `.py` files in `ptycho/` excluding `__init__.py`).
2.  **Map Dependencies:** Use static analysis (`pydeps`) and targeted inspection (`grep`) to build a complete map of which modules import from others.
3.  **Synthesize Architectural Roles:** Based on the dependency map, categorize each module (e.g., "Core Data Pipeline," "Low-Level Utility," "Configuration"). This analysis is mandatory and will inform the quality of your plan.

#### **Step 2: Design the Execution Strategy**

Based on your analysis, design a robust, multi-phase plan for the documentation initiative.

1.  **Structure the Plan:** The plan must be broken down into three phases:
    *   Phase 1: Analysis & Scoping (codifying the work from your Step 1).
    *   Phase 2: Implementation (using a sub-agent for each module).
    *   Phase 3: Final Verification (ensuring consistency and quality).
2.  **Define Sub-Agent Instructions:** You will create a detailed set of instructions for the "Authoring Sub-Agents." These instructions must be prescriptive and include the two high-quality docstring examples to guide the agent's writing style for different types of modules.

#### **Step 3: Synthesize the Final Checklist File**

Your final action is to generate the new file.

1.  Create a new file named `docstring_initiative_checklist.md`.
2.  Populate this file with the complete, self-contained implementation plan, using the strict template provided in the "Final Deliverable" section below.

---

### **The Final Deliverable: The Checklist File**

Your final output **MUST** be a single markdown file named `docstring_initiative_checklist.md`. The content of this file **MUST** follow the template below.

---
<!-- This is the template for the file you will create: docstring_initiative_checklist.md -->

# Agent Implementation Checklist: Module Docstring Initiative

**Initiative:** Comprehensive Module Docstring Generation
**Created:** <Date>
**Phase Goal:** To add a high-quality, concise, public-interface-focused docstring to every non-script `.py` module in the `ptycho/` library, ensuring each docstring is less than 15% of the module's size.
**Deliverable:** A fully documented `ptycho/` library with consistent, useful module-level docstrings and a passing `pydocstyle` verification check.

## ✅ Task List

### <!-- Main Agent Instructions -->
**Instructions for the Main Agent:**
1.  Work through the phases in order. Do not proceed to the next phase until the previous one is fully complete.
2.  For **Phase 2**, you will act as an orchestrator. For each module listed, you will spawn a dedicated sub-agent with the specific instructions provided.
3.  Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID | Task Description | State | How/Why & API Guidance |
| :-- | :--- | :--- | :--- |
| **Phase 1: Analysis & Scoping**
| 1.A | **Generate Definitive List of Target Modules** | `[ ]` | **Why:** To create a master list of all modules that require a docstring. <br> **How:** Execute the following command from the project root and save the output. <br> **Command:** <br> `find ptycho -name "*.py" -not -name "__init__.py" > modules_to_document.txt` <br> **Verify:** The file `modules_to_document.txt` should exist and contain a list of `.py` files. |
| 1.B | **Generate Static Dependency Map** | `[ ]` | **Why:** To provide the necessary context for all sub-agents to understand each module's public interface and role. <br> **How:** Install `pydeps` (`pip install pydeps`) and run the following commands to generate both visual and text-based dependency reports. <br> **Commands:** <br> `pydeps ptycho --cluster -o ptycho/dependency_graph.svg` <br> `pydeps ptycho --no-output --show-deps > ptycho/dependency_report.txt` <br> **Verify:** The files `dependency_graph.svg` and `dependency_report.txt` exist in the `ptycho/` directory. |
| 1.C | **Create a Progress Tracking Checklist** | `[ ]` | **Why:** To track the completion status of each sub-agent's task. <br> **How:** Create a new markdown file named `docstring_progress.md`. Copy the contents of `modules_to_document.txt` into it and format it as a checklist. <br> **Example:** <br> `- [ ] ptycho/config/config.py` <br> `- [ ] ptycho/params.py` <br> `...` |
| **Phase 2: Sub-Agent Orchestration for Docstring Implementation**
| 2.A | **Spawn Sub-Agents for Each Module** | `[ ]` | **Why:** To process each module independently and in parallel if possible. <br> **How:** For each file path listed in `modules_to_document.txt`, invoke a sub-agent with the specific "Sub-Agent Instructions" provided below. Pass the module's file path and the paths to the dependency reports (`ptycho/dependency_report.txt` and `ptycho/dependency_graph.svg`) as context. As each sub-agent completes its task, update the `docstring_progress.md` checklist. |
| **Phase 3: Final Verification & Consistency Pass**
| 3.A | **Verify All Modules are Documented** | `[ ]` | **Why:** To ensure no modules were missed. <br> **How:** Write a script that reads `modules_to_document.txt` and checks that each file now starts with a `"""` docstring. The script should fail if any module is undocumented. |
| 3.B | **Run Cross-Reference and Consistency Check** | `[ ]` | **Why:** To ensure the docstrings are not just present, but are consistent and reference each other correctly. <br> **How:** Invoke a final "Verification Sub-Agent" with the instructions below. This agent's task is to read *all* the new docstrings and the dependency map to ensure they form a coherent whole. |
| 3.C | **Run Automated Docstring Style Linting** | `[ ]` | **Why:** To enforce a consistent documentation style across the entire project. <br> **How:** Install `pydocstyle` (`pip install pydocstyle`) and run it on the `ptycho` directory. <br> **Command:** <br> `pydocstyle ptycho/` <br> **Verify:** The command should report no errors, or only minor, acceptable warnings. |
| 3.D | **Final Code Commit** | `[ ]` | **Why:** To save the completed documentation work to the repository. <br> **How:** Stage all the modified Python files and commit them. <br> **Command:** <br> `git add ptycho/**/*.py` <br> `git commit -m "docs: Add comprehensive module-level docstrings\n\n- Documents the public interface for all core library modules.\n- Follows a consistent format with usage examples.\n- Docstring size is constrained to <15% of module size."` |

---

### <!-- Sub-Agent Instructions -->
**Sub-Agent Instructions: Docstring Authoring (v4 - Final Hardened Version)**

**Your Goal:** Write a single, high-quality, developer-focused module-level docstring for the specified Python module.

**Your Context:**
*   **Target Module:** `<path/to/module.py>`
*   **Dependency Report:** `ptycho/dependency_report.txt`
*   **Architecture Context:** You MUST use the provided architectural summary to inform your writing.

**Your Guiding Principles (Non-Negotiable):**
1.  **Adapt to the Module's Nature:** You MUST analyze the module and determine its primary characteristic. Is it defined by complex conditional logic (like the `raw_data.py` example) or by its data transformations and tensor shape contracts (like the `tf_helper.py` example)? Your docstring's focus MUST reflect this.
2.  **Data Contracts are King:** If the module's primary purpose is to transform data shapes, you MUST explicitly document the input and output tensor formats and shapes.
3.  **Explain Parameter *Effects*:** For public functions, explain the *effect* of critical parameters on the behavior of the system.
4.  **Realistic Workflow Examples:** Your usage example MUST be a practical, multi-step snippet that shows how the module interacts with its primary consumers and dependencies.

**Your Workflow:**
1.  **Analysis:** Perform the dependency analysis to define the module's exact public API and its consumers. You MUST also scan the target module's source code for any imports from or calls to the legacy `ptycho.params` module. If found, you MUST investigate how this external state alters the module's behavior.
2.  **Drafting:** Write the docstring, strictly adhering to the **"Hardened Docstring Template"** below. You MUST choose the most appropriate style based on the two provided examples and fill out every section. If you identified any hidden dependencies, you MUST document them in the **"Architectural Notes & Dependencies"** section.
3.  **Constraint Verification:** Run a script to ensure your docstring is under the 15% size limit. Refactor for conciseness if it fails.
4.  **Anti-Pattern Review:** Before finalizing, you MUST review the **"Docstring Anti-Patterns"** section below and ensure your docstring does not violate any of them.
5.  **Finalization:** Insert the docstring into the target file.

---

### **Hardened Docstring Template (Mandatory Structure)**

*Your docstring must follow the structure and quality of the examples below. Choose the example that best fits the nature of the module you are documenting.*

---
#### **Example 1: For Modules with Complex Logic & Hidden State (e.g., `raw_data.py`)**

This style is best for modules whose behavior is controlled by complex conditional logic or external state.

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

This style is best for utility modules that perform a series of data shape and type transformations.

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

### **Sub-Agent Instructions: Final Verification Agent**

*(These are the instructions for the agent spawned in Phase 3.B)*

**Your Goal:** To perform a final consistency and architectural accuracy check on all newly created docstrings.

**Your Context:**
*   The list of all documented modules: `modules_to_document.txt`
*   The full dependency map: `ptycho/dependency_report.txt`
*   The PtychoPINN architecture understanding from `docs/DEVELOPER_GUIDE.md` and `docs/architecture.md`

**Your Workflow:**
1.  **Read All Docstrings:** Load the module-level docstring from every file listed in `modules_to_document.txt`.
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

**Verification Criteria:**
- **Architectural Accuracy:** Do the docstrings correctly describe each module's role in the PtychoPINN system?
- **Dependency Consistency:** Do "consumer" claims match actual import relationships?
- **Workflow Realism:** Do usage examples reflect actual integration patterns in the codebase?
- **Cross-References:** Do modules that work together reference each other appropriately?
