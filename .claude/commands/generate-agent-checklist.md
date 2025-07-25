# **Runbook: Comprehensive Module Documentation (v5)**

**CRITICAL META-INSTRUCTION: READ THIS FIRST**

You are the top-level orchestrator. Your task is to execute the steps in this runbook **sequentially, in your current context.**

You **MUST NOT** delegate this entire runbook to a single sub-task (e.g., `Task(Document all modules)`). Delegating the entire process will cause a critical loss of state and context, leading to failure. You are the main thread of execution. You will delegate only when explicitly instructed to do so for small, specific sub-tasks (like documenting a single file).

**Your immediate first action is to begin with Phase 1, Task 1.A.**

---

**Your Role and Objective:**
You are the autonomous Orchestrator for this documentation initiative. You will manage the entire project from start to finish by executing the phases below.

**User's Goal:**
> "I want every `.py` module (not counting scripts) to have a docstring <15 percent of the module size that documents the public interface: i.e., gives sufficient information on how that module is used / is to be used in other parts of the code or as a public api."

---

### **State Management Between Steps**

The files you create in Phase 1 (e.g., `modules_prioritized.txt`, `dependency_report.txt`) are the **state** for this entire execution. You must ensure they are available in your working directory for all subsequent steps. Do not lose track of these files.

---

### **Phase 1: Strategic Analysis & Prioritization**

*(You will execute these commands directly.)*

| ID | Task Description | State | How/Why & API Guidance |
| :-- | :--- | :--- | :--- |
| 1.A | **Generate List of Target Modules** | `[ ]` | **Why:** To create a master list of all modules that require a docstring. <br> **How:** Execute the following command now. <br> ```bash <br> find ptycho -name "*.py" -not -name "__init__.py" > modules_to_document.txt <br> ``` <br> **Verify:** The file `modules_to_document.txt` must exist before proceeding. |
| 1.B | **Generate Dependency Map** | `[ ]` | **Why:** To provide context for prioritization and for the sub-agents. <br> **How:** Execute the following commands now. If `pydeps` is not installed, install it first. <br> ```bash <br> pip install pydeps <br> pydeps ptycho --cluster -o ptycho/dependency_graph.svg <br> pydeps ptycho --no-output --show-deps > ptycho/dependency_report.txt <br> ``` <br> **Verify:** The file `dependency_report.txt` must exist before proceeding. |
| 1.C | **Prioritize Modules by Dependency** | `[ ]` | **Why:** To create an intelligent execution order. Foundational modules (least dependent) must be documented first. <br> **How:** You must now write and execute a script (e.g., Python) that reads `dependency_report.txt` and `modules_to_document.txt` to produce a new, sorted list. <br> **Output:** A new file, `modules_prioritized.txt`. <br> **Example Logic:** <br> ```python <br> # Pseudocode for the script you will write and run now. <br> dependencies = parse_pydeps_report('dependency_report.txt') <br> modules = read_file_lines('modules_to_document.txt') <br> sorted_modules = sorted(modules, key=lambda m: len(dependencies.get(m, []))) <br> write_lines_to_file('modules_prioritized.txt', sorted_modules) <br> ``` <br> **Verify:** The file `modules_prioritized.txt` must exist before proceeding. |
| 1.D | **Create Progress Tracker** | `[ ]` | **Why:** To track the completion status of each sub-agent's task. <br> **How:** Create a new markdown file named `docstring_progress.md` containing the contents of `modules_prioritized.txt` formatted as a checklist. |

---

### **Phase 2: Sub-Agent Orchestration**

*(You will now begin a loop and delegate tasks one by one.)*

| ID | Task Description | State | How/Why & API Guidance |
| :-- | :--- | :--- | :--- |
| 2.A | **Orchestrate Documentation of Each Module** | `[ ]` | **Why:** To process each module independently by delegating to specialized sub-agents. <br> **How:** Begin a loop. For each file path in `modules_prioritized.txt`: <br> 1. **Invoke a new, single-purpose "Authoring Sub-Agent."** <br> 2. Provide it with the instructions from the **"Sub-Agent Instructions: Docstring Authoring"** section below. <br> 3. Pass the specific module's file path and the `dependency_report.txt` file as context. <br> 4. After the sub-agent successfully completes, mark the corresponding item as done in `docstring_progress.md` and proceed to the next module in the loop. |

---

### **Phase 3: Final Verification & Commit**

*(After the loop in Phase 2 is complete, you will execute these final steps.)*

| ID | Task Description | State | How/Why & API Guidance |
| :-- | :--- | :--- | :--- |
| 3.A | **Verify All Modules are Documented** | `[ ]` | **Why:** To ensure no modules were missed by the sub-agents. <br> **How:** Run a script that reads `modules_prioritized.txt` and checks that each file now starts with a `"""` docstring. The script must fail if any module is undocumented. |
| 3.B | **Invoke Verification Sub-Agent** | `[ ]` | **Why:** To ensure docstrings are consistent and architecturally sound. <br> **How:** Invoke a final "Verification Sub-Agent" with the instructions from the **"Sub-Agent Instructions: Final Verification"** section below. Pass it the `dependency_report.txt` file as context. You must review its findings and apply any necessary fixes. |
| 3.C | **Run Automated Style Linting** | `[ ]` | **Why:** To enforce a consistent documentation style across the entire project. <br> **How:** Install `pydocstyle` (`pip install pydocstyle`) and run it on the `ptycho` directory. <br> ```bash <br> pydocstyle ptycho/ <br> ``` <br> **Verify:** The command should report no errors. You must fix any reported issues. |
| 3.D | **Final Code Commit** | `[ ]` | **Why:** To save the completed documentation work to the repository. <br> **How:** Stage all the modified Python files and commit them with a detailed message. <br> ```bash <br> git add ptycho/**/*.py <br> git commit -m "docs: Add comprehensive module-level docstrings" -m "Adds public-interface-focused docstrings to all core library modules, following a consistent format with usage examples and architectural context. Docstring size is constrained to <15% of module size." <br> ``` |

---

### **Sub-Agent Instructions: Docstring Authoring (v5)**

*(Orchestrator: You will provide these instructions to each sub-agent you invoke in Phase 2.A.)*

**Your Goal:** Write a single, high-quality, developer-focused module-level docstring for the specified Python module.

**Your Context:**
*   **Target Module:** `<path/to/module.py>`
*   **Dependency Report:** `ptycho/dependency_report.txt`

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

**Your Goal:** To perform a final consistency and architectural accuracy check on all newly created docstrings.

**Your Context:**
*   The list of all documented modules: `modules_prioritized.txt`
*   The full dependency map: `ptycho/dependency_report.txt`
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
