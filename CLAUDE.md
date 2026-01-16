# CLAUDE.md

This file provides guidance to Claude when working with the PtychoPINN repository.

## 🚨 CRITICAL: Data File Management

**NEVER commit data files (NPZ, HDF5, etc.) to Git!** See <doc-ref type="critical">docs/DATA_MANAGEMENT_GUIDE.md</doc-ref> for mandatory practices.

## ⚠️ Core Project Directives

<directive level="critical" purpose="Understand current project focus">
  Before starting any new task, you **MUST** first read the project status tracker: 
  <doc-ref type="status">docs/PROJECT_STATUS.md</doc-ref>.
  When reading or modifying files in a subdir of the project root, if a CLAUDE.md file exists 
  in that subdir, you **MUST** read that CLAUDE.md.
</directive>

<directive level="important" purpose="Avoid modifying stable core logic">
  The core ptychography physics simulation and the TensorFlow model architecture are considered stable and correct. **Do not modify the core logic in `<code-ref type="module">ptycho/model.py</code-ref>`, `<code-ref type="module">ptycho/diffsim.py</code-ref>`, or `<code-ref type="module">ptycho/tf_helper.py</code-ref>` unless explicitly asked.**
</directive>

<directive level="important" purpose="Prioritize data validation over model debugging">
  Most errors in this project stem from incorrect input data formats, not bugs in the model code. Before debugging the model, **always verify the input data structure first.**
</directive>

<directive level="guidance" purpose="Use established workflows">
  The `scripts/` directory contains high-level, tested workflows. Use these as entry points for tasks like training and simulation. Prefer using these scripts over writing new, low-level logic.
</directive>

<directive level="guidance" purpose="Use configuration for parameter changes">
  Changes to experimental parameters (e.g., learning rate, image size) should be made via configuration files (`.yaml`) or command-line arguments, not by hardcoding values in the Python source.
</directive>

<directive level="critical" purpose="Ensure all new documentation is discoverable">
  When you create any new documentation file (`.md`), you **MUST** ensure it is discoverable by linking to it from at least one existing, high-level document. This is not optional. Follow these rules to determine where to add the link:

  1.  **Identify the Document Type:**
      *   Is it a new high-level **R&D Plan** or **Implementation Plan**?
      *   Is it a user-facing **Workflow Guide** (like a `README.md` for a script)?
      *   Is it a core **Architectural Guide** (like `DEVELOPER_GUIDE.md`)?
      *   Does it modify a **Data Contract**?

  2.  **Add a Link Based on Type:**
      *   **For R&D/Implementation Plans:** The primary link **MUST** be added to `<doc-ref type="status">docs/PROJECT_STATUS.md</doc-ref>` under the relevant initiative.
      *   **For Workflow Guides:** A link **MUST** be added to the "Key Workflows & Scripts" section in `<doc-ref type="guide">CLAUDE.md</doc-ref>`.
      *   **For Core Architectural Guides:** A link **MUST** be added to either `<doc-ref type="guide">CLAUDE.md</doc-ref>` or `<doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>`, whichever is more contextually appropriate.
      *   **For Data Contract Changes:** You **MUST** update `<doc-ref type="contract">docs/data_contracts.md</doc-ref>`.

  3.  **Use XML Tags for Links:** All new links that you add for discoverability **MUST** use the `<doc-ref>` or `<code-ref>` XML tagging system to ensure they are machine-parsable.
</directive>

<directive level="critical" purpose="Enforce Test-Driven Development">
  For any task involving new feature implementation or bug fixing, you **MUST** follow a Test-Driven Development (TDD) methodology. Your implementation plan should explicitly detail the Red-Green-Refactor cycles.

  1.  **RED:** First, propose and write a fine-grained unit test that will fail but would pass if the feature were implemented or the bug were fixed.
  2.  **GREEN:** Then, propose and write the minimal implementation code required to make that specific test pass.
  3.  **REFACTOR:** Finally, propose any refactoring to clean up the code while ensuring the test still passes.

  For a canonical example of this process, refer to the case study on fixing the baseline model's `gridsize > 1` bug in the `<doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>`.
</directive>

## Project Overview

PtychoPINN is a TensorFlow-based implementation of physics-informed neural networks (PINNs) for ptychographic reconstruction. It combines a U-Net-like deep learning model with a differentiable physics layer to achieve rapid, high-resolution reconstruction from scanning coherent diffraction data.

### For Developers

Developers looking to contribute to the codebase or understand its deeper architectural principles should first read the **<doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>**. It contains critical information on the project's design, data pipeline, and best practices.

## 1. Getting Started: Environment & Verification

First, set up the environment and run a verification test to ensure the system is working correctly.

```bash
# 1. Create and activate conda environment
conda create -n ptycho python=3.10
conda activate ptycho

# 2. Install the package in editable mode
pip install -e .

# 3. Run a verification test with known-good data
# This proves the model and environment are set up correctly.
# It uses a small number of images for a quick test.
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_groups 512 --output_dir verification_run
```

If the verification run completes and creates files in the `verification_run/` directory, the environment is correct.

## ⚠️ CRITICAL: Parameter Initialization

**Before calling any data loading functions**, you MUST initialize the legacy params:

```python
from ptycho.config.config import update_legacy_dict
config = setup_configuration(args, yaml_path)
update_legacy_dict(params.cfg, config)  # ← REQUIRED before data operations!
```

**Common failure:** Shape `(*, 64, 64, 1)` instead of `(*, 64, 64, 4)` with gridsize=2
**Cause:** `params.cfg['gridsize']` not initialized
**Solution:** See <doc-ref type="troubleshooting">docs/TROUBLESHOOTING.md#shape-mismatch-errors</doc-ref>

## 2. Development & Testing Strategy

### Automated Testing

The project maintains a comprehensive suite of automated tests located in the top-level `tests/` directory. All new tests should be added there following the `test_*.py` naming convention.

**Test Templates:**
- **GridSize Testing:** Use `tests/test_template_gridsize.py` as a template for tests involving different gridsize values
- Shows proper `params.cfg` initialization to avoid shape mismatch bugs

To run all tests, execute the following command from the project root:
```bash
python -m unittest discover tests/
```

A key validation mechanism is the integration test in `tests/test_integration_workflow.py`. This test simulates the complete user workflow (train → save → load → infer) and serves as the primary validation for the model save/load cycle across separate processes.

For more detailed information on the testing strategy, different test types, and how to contribute new tests, refer to the **<doc-ref type="guide">docs/TESTING_GUIDE.md</doc-ref>**.

## 3. Key Workflows & Scripts

This project provides several high-level scripts to automate common tasks. For detailed usage, see the documentation in each script's directory.

**📚 For comprehensive documentation navigation, see <doc-ref type="index">docs/index.md</doc-ref>**

- **Data Management:** See <doc-ref type="critical">docs/DATA_MANAGEMENT_GUIDE.md</doc-ref> **⚠️ MUST READ**
- **Testing Guide:** See <doc-ref type="guide">docs/TESTING_GUIDE.md</doc-ref> **⚠️ NEW**
- **Troubleshooting:** See <doc-ref type="guide">docs/TROUBLESHOOTING.md</doc-ref> **⚠️ NEW - Debug shape mismatches & config issues**
- **Params Quick Reference:** See <doc-ref type="guide">docs/QUICK_REFERENCE_PARAMS.md</doc-ref> **⚠️ NEW - params.cfg initialization cheatsheet**
- **Training:** See <doc-ref type="workflow-guide">scripts/training/CLAUDE.md</doc-ref> and <doc-ref type="workflow-guide">scripts/training/README.md</doc-ref>
- **Inference:** See <doc-ref type="workflow-guide">scripts/inference/CLAUDE.md</doc-ref> and <doc-ref type="workflow-guide">scripts/inference/README.md</doc-ref>
- **Evaluation:** See <doc-ref type="workflow-guide">scripts/evaluation/README.md</doc-ref> **⚠️ NEW - Single model evaluation with metrics**
- **Simulation:** See <doc-ref type="workflow-guide">scripts/simulation/CLAUDE.md</doc-ref> and <doc-ref type="workflow-guide">scripts/simulation/README.md</doc-ref>
- **Sampling Guide:** See <doc-ref type="guide">docs/SAMPLING_USER_GUIDE.md</doc-ref> for independent sampling control **⚠️ NEW**
- **Reconstruction (Pty-Chi):** See <doc-ref type="workflow-guide">scripts/reconstruction/ptychi_reconstruct_tike.py</doc-ref> for pty-chi reconstruction workflow
- **Pty-chi Migration Guide:** See <doc-ref type="guide">docs/PTYCHI_MIGRATION_GUIDE.md</doc-ref> for replacing Tike with faster pty-chi reconstruction
- **Data Preprocessing Tools:** See <doc-ref type="workflow-guide">scripts/tools/CLAUDE.md</doc-ref> and <doc-ref type="workflow-guide">scripts/tools/README.md</doc-ref>
- **Model Comparison & Studies:** See <doc-ref type="workflow-guide">scripts/studies/CLAUDE.md</doc-ref>, <doc-ref type="workflow-guide">scripts/studies/README.md</doc-ref> and <doc-ref type="workflow-guide">scripts/studies/QUICK_REFERENCE.md</doc-ref>
- **Experimental Datasets:** See <doc-ref type="guide">docs/FLY64_DATASET_GUIDE.md</doc-ref>
- **Configuration Guide:** See <doc-ref type="guide">docs/CONFIGURATION_GUIDE.md</doc-ref>
- **Tool Selection Guide:** See <doc-ref type="guide">docs/TOOL_SELECTION_GUIDE.md</doc-ref>
- **Complete Workflow Guide:** See <doc-ref type="guide">docs/WORKFLOW_GUIDE.md</doc-ref>
- **Commands Reference:** See <doc-ref type="guide">docs/COMMANDS_REFERENCE.md</doc-ref>
- **Model Comparison Guide:** See <doc-ref type="guide">docs/MODEL_COMPARISON_GUIDE.md</doc-ref>
- **Project Organization Guide:** See <doc-ref type="guide">docs/PROJECT_ORGANIZATION_GUIDE.md</doc-ref>
- **Documentation Generation Guide:** See <doc-ref type="guide">docs/DOCUMENTATION_GENERATION_GUIDE.md</doc-ref>
- **Initiative Workflow Guide:** See <doc-ref type="workflow-guide">docs/INITIATIVE_WORKFLOW_GUIDE.md</doc-ref>
- **Gemini-Powered Initiative Commands:** For complex codebases (>5K lines), use `/customplan-gemini-full`, `/implementation-gemini-full`, `/phase-checklist-gemini-full`, and `/complete-phase-gemini-full` for comprehensive codebase analysis and exact code generation
- **Documentation Navigation:** See <doc-ref type="workflow-guide">docs/CLAUDE.md</doc-ref>
- **Core Library Development:** See <doc-ref type="workflow-guide">ptycho/CLAUDE.md</doc-ref>
- **GridSize Inference Troubleshooting:** See <doc-ref type="troubleshooting">docs/GRIDSIZE_INFERENCE_GOTCHAS.md</doc-ref>
- **GridSize and n_groups Interaction:** See <doc-ref type="critical">docs/GRIDSIZE_N_GROUPS_GUIDE.md</doc-ref> **⚠️ CRITICAL**

## 4. Key Workflows & Commands

### Training a Model

```bash
# Train using a YAML configuration file (preferred method)
ptycho_train --config configs/fly_config.yaml

# Train by specifying files and parameters directly
ptycho_train --train_data_file <path/to/train.npz> --test_data_file <path/to/test.npz> --output_dir <output_path> --n_groups 5000
```

### Running Inference

```bash
# Run inference on a test dataset using a trained model
ptycho_inference --model_path <path/to/model_dir> --test_data <path/to/test.npz> --output_dir <inference_output>
```

### Evaluating a Single Model

```bash
# Evaluate a trained model with comprehensive metrics
ptycho_evaluate --model-dir <path/to/model_dir> --test-data <path/to/test.npz> --output-dir <evaluation_output>

# Example with sampling and custom settings
ptycho_evaluate --model-dir my_model --test-data test.npz --output-dir eval_results --n-test-groups 1000 --phase-align-method plane
```

### Simulating a Dataset

```bash
# Direct simulation tool - simulate data from an existing object/probe file
python scripts/simulation/simulate_and_save.py \
    --input-file <path/to/obj_probe.npz> \
    --output-file <path/to/new_sim_data.npz> \
    --n-images 2000 \
    --gridsize 1

# Example with visualization
python scripts/simulation/simulate_and_save.py \
    --input-file datasets/fly/fly001_transposed.npz \
    --output-file sim_outputs/fly_simulation.npz \
    --n-images 1000 \
    --visualize

# High-level simulation workflow (recommended for complex scenarios)
python scripts/simulation/run_with_synthetic_lines.py \
    --output-dir simulation_outputs \
    --n-images 2000
```

### Running Tests

```bash
# Run all unit tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests.test_integration_workflow

# Run with verbose output
python -m unittest discover tests/ -v
```

## 5. Configuration Parameters

Parameters are controlled via YAML files (see `configs/`) or command-line arguments. The system uses modern `dataclasses` for configuration.

**For complete configuration details, see <doc-ref type="guide">docs/CONFIGURATION_GUIDE.md</doc-ref>**

### Quick Reference
- **Model Architecture**: `N` (diffraction pattern size), `model_type` (pinn/supervised), `object_big` (patch stitching)
- **Training**: `nepochs`, `batch_size`, `output_dir`  
- **Data**: `train_data_file`, `test_data_file`, `n_images`
- **Physics**: `nphotons`, `nll_weight`, `probe_trainable`

## 6. Critical: Data Format Requirements

**This is the most common source of errors.** A mismatch here will cause low-level TensorFlow errors that are hard to debug.

**Authoritative Source:** For all tasks involving the creation or modification of `.npz` datasets, you **MUST** consult and adhere to the specifications in the **<doc-ref type="contract">docs/data_contracts.md</doc-ref>**. This file defines the required key names, array shapes, and data types.

-   **`probeGuess`**: The scanning beam. A complex `(N, N)` array.
-   **`objectGuess`**: The full sample being scanned. A complex `(M, M)` array, where `M` is typically 3-5 times `N`.
-   **`diffraction`**: The stack of measured diffraction patterns. This must be a real `(n_images, N, N)` array representing **amplitude** (i.e., the square root of the measured intensity). The model's Poisson noise layer will square this value internally to simulate photon counts.

```python
# Example: Convert measured intensity to required amplitude format
measured_intensity = ... # Your (n_images, N, N) intensity data
diffraction_amplitude = np.sqrt(measured_intensity)
```
-   **`xcoords`, `ycoords`**: 1D arrays of scan positions, shape `(n_images,)`.

**Reference Example (Known-Good Data):**
File: `datasets/fly/fly001_transposed.npz`
- `probeGuess`: `(64, 64)`
- `objectGuess`: `(232, 232)`  *(Note: much larger than probe)*
- `diffraction`: `(10304, 64, 64)`

**Common Pitfall:** Creating a synthetic `objectGuess` that is the same size as the `probeGuess`. This leaves no room for the probe to scan across the object and will fail. Another common issue is storing intensity instead of amplitude in the `diffraction` array.

## 6.5. Critical: Understanding Normalization Conventions

**⚠️ CRITICAL:** PtychoPINN uses three distinct types of normalization that serve different purposes. Confusing these will lead to incorrect results.

**For complete normalization documentation, see <doc-ref type="guide">docs/DATA_NORMALIZATION_GUIDE.md</doc-ref>**

### The Three Types of Normalization

1. **Physics Normalization (`intensity_scale`)**
   - **Purpose:** Scales simulated data to realistic photon counts for Poisson noise modeling
   - **Where:** Applied ONLY in the physics loss layer during training
   - **Key:** Internal data remains normalized; scaling happens at physics boundary
   
2. **Statistical Normalization (`normalize_data`)**
   - **Purpose:** Preprocesses data for stable neural network training (zero mean, unit variance)
   - **Where:** Applied to training data before feeding to model
   - **Key:** Standard ML preprocessing, unrelated to physics
   
3. **Display/Comparison Scaling**
   - **Purpose:** Visual adjustments for plots and metrics
   - **Where:** Applied only for visualization and comparison
   - **Key:** Never affects training or physics calculations

### Critical Pipeline Convention

```python
# CORRECT: Internal pipeline keeps data normalized
X, Y_I, Y_phi, intensity_scale = illuminate_and_diffract(...)  # Returns normalized
norm_Y_I = scale_nphotons(X)  # Calculates factor but doesn't apply
# X remains normalized throughout pipeline

# WRONG: Applying scaling in data pipeline
X_scaled = X * norm_Y_I  # DON'T DO THIS in raw_data.py!
# This would break prepare.sh and other workflows
```

### Common Misunderstandings

- **Misunderstanding 1:** "nphotons should scale the diffraction patterns"
  - **Reality:** nphotons affects the Poisson noise model, not data values
  - **Fix:** Set nphotons parameter, but keep data normalized

- **Misunderstanding 2:** "Low photon data should have smaller values"
  - **Reality:** Data stays normalized; photon effects appear in noise statistics
  - **Fix:** Use physics loss to model photon statistics correctly

## 7. High-Level Architecture

-   **Configuration (`ptycho/config/`)**: Dataclass-based system (`ModelConfig`, `TrainingConfig`). This is the modern way to control the model. The source of truth is <code-ref type="config">ptycho/config/config.py</code-ref>. A legacy `params.cfg` dictionary is still used for backward compatibility. **Crucially, this is a one-way street:** at the start of a workflow, the modern `TrainingConfig` object is used to update the legacy `params.cfg` dictionary. This allows older modules that still use `params.get('key')` to receive the correct values from a single, modern source of truth. New code should always accept a configuration dataclass as an argument and avoid using the legacy `params.get()` function.
-   **Workflows (`ptycho/workflows/`)**: High-level functions that orchestrate common tasks (e.g., `run_cdi_example`). The `scripts/` call these functions.
-   **Data Loading (`ptycho/loader.py`, `ptycho/raw_data.py`)**: Defines `RawData` (for raw files) and `PtychoDataContainer` (for model-ready data).
-   **Model (`ptycho/model.py`)**: Defines the U-Net architecture and the custom Keras layers that incorporate the physics.
-   **Simulation (`ptycho/diffsim.py`, `ptycho/nongrid_simulation.py`)**: Contains the functions for generating simulated diffraction data from an object and probe.
-   **Image Processing (`ptycho/image/`)**: The modern, authoritative location for image processing tasks.
    -   `stitching.py`: Contains functions for grid-based patch reassembly.
    -   `cropping.py`: Contains the crucial `<code-ref type="function">align_for_evaluation</code-ref>` function for robustly aligning a reconstruction with its ground truth for metric calculation.

## 8. Tool Selection Guidance

Understanding which tool to use for different workflows is critical for efficient development.

**For complete tool selection guidance, see <doc-ref type="guide">docs/TOOL_SELECTION_GUIDE.md</doc-ref>**

### Quick Reference
- **Complete studies**: `run_complete_generalization_study.sh` 
- **Model comparison**: `compare_models.py`
- **Result visualization**: `aggregate_and_plot_results.py`
- **Dataset debugging**: `scripts/tools/visualize_dataset.py`
- **Data preparation**: `scripts/tools/split_dataset_tool.py`

## 9. Comparing Models: PtychoPINN vs Baseline

**For complete model comparison documentation, see <doc-ref type="guide">docs/MODEL_COMPARISON_GUIDE.md</doc-ref>**

### Quick Start

```bash
# Complete workflow: train both models + compare
./scripts/run_comparison.sh <train_data.npz> <test_data.npz> <output_dir>

# Compare existing trained models
python scripts/compare_models.py \
    --pinn_dir <pinn_model_dir> \
    --baseline_dir <baseline_model_dir> \
    --test_data <test_data.npz> \
    --output_dir <comparison_output>
```

### Key Features
- **Automatic image registration** for fair comparison
- **Advanced metrics** (SSIM, MS-SSIM, FRC)
- **Debug visualization** with `--save-debug-images`
- **Unified NPZ exports** for downstream analysis

## 10. Understanding the Output Directory

After a successful training run using `ptycho_train --output_dir <my_run>`, the output directory will contain several key files:

- **`logs/`**: Directory containing all log files for the run
  - **`debug.log`**: Complete log history (DEBUG level and above) for troubleshooting
- **`wts.h5.zip`**: This is the primary output. It's a zip archive containing the trained model weights and architecture for both the main autoencoder and the inference-only `diffraction_to_obj` model. Use `ModelManager.load_multiple_models()` to load it.
- **`history.dill`**: A Python pickle file (using dill) containing the training history dictionary. You can load it to plot loss curves:
  ```python
  import dill
  with open('<my_run>/history.dill', 'rb') as f:
      history = dill.load(f)
  plt.plot(history['loss'])
  ```
- **`reconstructed_amplitude.png` / `reconstructed_phase.png`**: Visualizations of the final reconstructed object from the test set, if stitching was performed.
- **`metrics.csv`**: If a ground truth object was available, this file contains quantitative image quality metrics (MAE, PSNR, FRC) comparing the reconstruction to the ground truth.
- **`params.dill`**: A snapshot of the full configuration used for the run, for reproducibility.

### Enhanced Logging System

The project uses an advanced centralized logging system with comprehensive output capture:

**Key Features:**
- **Complete Output Capture:** ALL stdout (including print statements from any module) is captured to log files
- **Tee-style Logging:** Simultaneous console and file output with flexible control
- **Command-line Options:** `--quiet`, `--verbose`, and `--console-level` for different use cases

**Common Usage:**
```bash
# Interactive development (default)
ptycho_train --train_data datasets/fly64.npz --output_dir my_run

# Automation-friendly (quiet console)  
ptycho_train --train_data datasets/fly64.npz --output_dir my_run --quiet

# Debugging (verbose console output)
ptycho_train --train_data datasets/fly64.npz --output_dir my_run --verbose
```

### Troubleshooting: Log File Locations

**Critical:** When a workflow fails, you **MUST** look for log files in the specified `<output_dir>/logs/` directory, not the project root. The centralized logging system ensures all logs are organized within each run's output directory, making it easier to debug specific runs and keeping the project root clean.

**Complete Record:** The `debug.log` file contains:
- All logging messages from the application
- All print() statements from any imported module
- Model architecture summaries and data shape information  
- Debug output from core modules
- **Everything that appeared on stdout during execution**

## 11. Advanced & Undocumented Features

### 11.1. Caching Decorators (`ptycho/misc.py`)

- **`@memoize_disk_and_memory`**: Caches the results of expensive functions to disk to speed up subsequent runs with the same parameters.
- **`@memoize_simulated_data`**: Specifically designed for caching simulated data generation, avoiding redundant computation.

### 11.2. Data Utility Tools (`scripts/tools/`)

- **`downsample_data_tool.py`**: For cropping k-space and binning real-space arrays to maintain physical consistency.
- **`prepare_data_tool.py`**: For apodizing, smoothing, or interpolating probes/objects before simulation.
- **`update_tool.py`**: For updating an NPZ file with a new reconstruction result.
- **`visualize_dataset.py`**: For generating a comprehensive visualization plot of an NPZ dataset.
- **`strip_code.py`**: For extracting module-level docstrings from Python files to create documentation-only views.

### 11.3. Documentation Generation Workflow

The project includes tools for creating documentation-only views of the codebase, useful for AI context priming and architectural analysis:

- **`/generate-doc-context` command**: Creates an isolated git worktree with Python files stripped to only their module-level docstrings
- **`strip_code.py` utility**: The underlying tool that extracts docstrings using Python's AST
- **`.maskset` files**: Define which files to include in the documentation view (see <doc-ref type="contract">docs/data_contracts.md</doc-ref> section 3)

**Example workflow:**
```bash
# Using a maskset file
/generate-doc-context architecture.maskset

# Using direct patterns
/generate-doc-context "ptycho/**/*.py" "scripts/workflows/*.py"
```

This creates a temporary worktree where selected Python files contain only their docstrings, preserving the interface documentation while removing implementation details.

### 11.4. Automated Testing Framework (`ptycho/autotest/`)

- This internal framework provides testing utilities for the project.
- The `@debug` decorator (imported from `ptycho.autotest.debug`) is used to serialize function inputs and outputs during development for creating regression tests.
- This is a developer-facing feature primarily used for debugging and test creation.

## 12. Legacy Code & Deprecation Warnings

- **Legacy Training Script (`ptycho/train.py`):** The file `ptycho/train.py` is a legacy script that uses an older configuration system. **Do not use it.** Always use the `ptycho_train` command-line tool (which points to `scripts/training/train.py`) for all training workflows, as it uses the modern, correct configuration system.

## 13. Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

### File and Directory Inclusion Syntax

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the gemini command:

**Examples:**

```bash
# Single file analysis
gemini -p "@src/main.py Explain this file's purpose and structure"

# Multiple files
gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

# Entire directory
gemini -p "@src/ Summarize the architecture of this codebase"

# Multiple directories
gemini -p "@src/ @tests/ Analyze test coverage for the source code"

# Current directory and subdirectories
gemini -p "@./ Give me an overview of this entire project"
# Or use --all_files flag:
gemini --all_files -p "Analyze the project structure and dependencies"
```

### Implementation Verification Examples

```bash
# Check if a feature is implemented
gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"

# Verify authentication implementation
gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"

# Check for specific patterns
gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"

# Verify error handling
gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"

# Check for rate limiting
gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"

# Verify caching strategy
gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"

# Check for specific security measures
gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"

# Verify test coverage for features
gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"
```

### When to Use Gemini CLI

Use `gemini -p` when:
- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

### Important Notes

- Paths in @ syntax are relative to your current working directory when invoking gemini
- The CLI will include file contents directly in the context
- No need for --yolo flag for read-only analysis
- Gemini's context window can handle entire codebases that would overflow Claude's context
- When checking implementations, be specific about what you're looking for to get accurate results

## 14. Project Organization

For detailed information on project file organization, initiative planning, and document structure conventions, see <doc-ref type="guide">docs/PROJECT_ORGANIZATION_GUIDE.md</doc-ref>.