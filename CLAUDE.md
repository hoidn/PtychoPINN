# CLAUDE.md

This file provides guidance to Claude when working with the PtychoPINN repository.

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
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 512 --output_dir verification_run
```

If the verification run completes and creates files in the `verification_run/` directory, the environment is correct.

## 2. Key Workflows & Scripts

This project provides several high-level scripts to automate common tasks. For detailed usage, see the documentation in each script's directory.

- **Training:** See <doc-ref type="workflow-guide">scripts/training/CLAUDE.md</doc-ref> and <doc-ref type="workflow-guide">scripts/training/README.md</doc-ref>
- **Inference:** See <doc-ref type="workflow-guide">scripts/inference/CLAUDE.md</doc-ref> and <doc-ref type="workflow-guide">scripts/inference/README.md</doc-ref>
- **Simulation:** See <doc-ref type="workflow-guide">scripts/simulation/CLAUDE.md</doc-ref> and <doc-ref type="workflow-guide">scripts/simulation/README.md</doc-ref>
- **Data Preprocessing Tools:** See <doc-ref type="workflow-guide">scripts/tools/CLAUDE.md</doc-ref> and <doc-ref type="workflow-guide">scripts/tools/README.md</doc-ref>
- **Model Comparison & Studies:** See <doc-ref type="workflow-guide">scripts/studies/CLAUDE.md</doc-ref>, <doc-ref type="workflow-guide">scripts/studies/README.md</doc-ref> and <doc-ref type="workflow-guide">scripts/studies/QUICK_REFERENCE.md</doc-ref>
- **Experimental Datasets:** See <doc-ref type="guide">docs/FLY64_DATASET_GUIDE.md</doc-ref>
- **Configuration Guide:** See <doc-ref type="guide">docs/CONFIGURATION_GUIDE.md</doc-ref>
- **Tool Selection Guide:** See <doc-ref type="guide">docs/TOOL_SELECTION_GUIDE.md</doc-ref>
- **Commands Reference:** See <doc-ref type="guide">docs/COMMANDS_REFERENCE.md</doc-ref>
- **Model Comparison Guide:** See <doc-ref type="guide">docs/MODEL_COMPARISON_GUIDE.md</doc-ref>
- **Project Organization Guide:** See <doc-ref type="guide">docs/PROJECT_ORGANIZATION_GUIDE.md</doc-ref>
- **Documentation Generation Guide:** See <doc-ref type="guide">docs/DOCUMENTATION_GENERATION_GUIDE.md</doc-ref>
- **Initiative Workflow Guide:** See <doc-ref type="workflow-guide">docs/INITIATIVE_WORKFLOW_GUIDE.md</doc-ref>
- **Gemini-Powered Initiative Commands:** For complex codebases (>5K lines), use `/customplan-gemini-full`, `/implementation-gemini-full`, `/phase-checklist-gemini-full`, and `/complete-phase-gemini-full` for comprehensive codebase analysis and exact code generation
- **Documentation Navigation:** See <doc-ref type="workflow-guide">docs/CLAUDE.md</doc-ref>
- **Core Library Development:** See <doc-ref type="workflow-guide">ptycho/CLAUDE.md</doc-ref>
- **GridSize Inference Troubleshooting:** See <doc-ref type="troubleshooting">docs/GRIDSIZE_INFERENCE_GOTCHAS.md</doc-ref>

## 2. Key Workflows & Commands

### Training a Model

```bash
# Train using a YAML configuration file (preferred method)
ptycho_train --config configs/fly_config.yaml

# Train by specifying files and parameters directly
ptycho_train --train_data_file <path/to/train.npz> --test_data_file <path/to/test.npz> --output_dir <output_path> --n_images 5000
```

### Running Inference

```bash
# Run inference on a test dataset using a trained model
ptycho_inference --model_path <path/to/model_dir> --test_data <path/to/test.npz> --output_dir <inference_output>
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
python -m unittest discover -s ptycho -p "test_*.py"
```

## 3. Configuration Parameters

Parameters are controlled via YAML files (see `configs/`) or command-line arguments. The system uses modern `dataclasses` for configuration.

**For complete configuration details, see <doc-ref type="guide">docs/CONFIGURATION_GUIDE.md</doc-ref>**

### Quick Reference
- **Model Architecture**: `N` (diffraction pattern size), `model_type` (pinn/supervised), `object_big` (patch stitching)
- **Training**: `nepochs`, `batch_size`, `output_dir`  
- **Data**: `train_data_file`, `test_data_file`, `n_images`
- **Physics**: `nphotons`, `nll_weight`, `probe_trainable`

## 4. Critical: Data Format Requirements

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

## 5. High-Level Architecture

-   **Configuration (`ptycho/config/`)**: Dataclass-based system (`ModelConfig`, `TrainingConfig`). This is the modern way to control the model. The source of truth is <code-ref type="config">ptycho/config/config.py</code-ref>. A legacy `params.cfg` dictionary is still used for backward compatibility. **Crucially, this is a one-way street:** at the start of a workflow, the modern `TrainingConfig` object is used to update the legacy `params.cfg` dictionary. This allows older modules that still use `params.get('key')` to receive the correct values from a single, modern source of truth. New code should always accept a configuration dataclass as an argument and avoid using the legacy `params.get()` function.
-   **Workflows (`ptycho/workflows/`)**: High-level functions that orchestrate common tasks (e.g., `run_cdi_example`). The `scripts/` call these functions.
-   **Data Loading (`ptycho/loader.py`, `ptycho/raw_data.py`)**: Defines `RawData` (for raw files) and `PtychoDataContainer` (for model-ready data).
-   **Model (`ptycho/model.py`)**: Defines the U-Net architecture and the custom Keras layers that incorporate the physics.
-   **Simulation (`ptycho/diffsim.py`, `ptycho/nongrid_simulation.py`)**: Contains the functions for generating simulated diffraction data from an object and probe.
-   **Image Processing (`ptycho/image/`)**: The modern, authoritative location for image processing tasks.
    -   `stitching.py`: Contains functions for grid-based patch reassembly.
    -   `cropping.py`: Contains the crucial `<code-ref type="function">align_for_evaluation</code-ref>` function for robustly aligning a reconstruction with its ground truth for metric calculation.

## 6. Tool Selection Guidance

Understanding which tool to use for different workflows is critical for efficient development.

**For complete tool selection guidance, see <doc-ref type="guide">docs/TOOL_SELECTION_GUIDE.md</doc-ref>**

### Quick Reference
- **Complete studies**: `run_complete_generalization_study.sh` 
- **Model comparison**: `compare_models.py`
- **Result visualization**: `aggregate_and_plot_results.py`
- **Dataset debugging**: `scripts/tools/visualize_dataset.py`
- **Data preparation**: `scripts/tools/split_dataset_tool.py`

## 7. Comparing Models: PtychoPINN vs Baseline

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

## 8. Understanding the Output Directory

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

## 9. Advanced & Undocumented Features

### 9.1. Caching Decorators (`ptycho/misc.py`)

- **`@memoize_disk_and_memory`**: Caches the results of expensive functions to disk to speed up subsequent runs with the same parameters.
- **`@memoize_simulated_data`**: Specifically designed for caching simulated data generation, avoiding redundant computation.

### 9.2. Data Utility Tools (`scripts/tools/`)

- **`downsample_data_tool.py`**: For cropping k-space and binning real-space arrays to maintain physical consistency.
- **`prepare_data_tool.py`**: For apodizing, smoothing, or interpolating probes/objects before simulation.
- **`update_tool.py`**: For updating an NPZ file with a new reconstruction result.
- **`visualize_dataset.py`**: For generating a comprehensive visualization plot of an NPZ dataset.
- **`strip_code.py`**: For extracting module-level docstrings from Python files to create documentation-only views.

### 9.3. Documentation Generation Workflow

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

### 9.4. Automated Testing Framework (`ptycho/autotest/`)

- This internal framework provides testing utilities for the project.
- The `@debug` decorator (imported from `ptycho.autotest.debug`) is used to serialize function inputs and outputs during development for creating regression tests.
- This is a developer-facing feature primarily used for debugging and test creation.

## 10. Legacy Code & Deprecation Warnings

- **Legacy Training Script (`ptycho/train.py`):** The file `ptycho/train.py` is a legacy script that uses an older configuration system. **Do not use it.** Always use the `ptycho_train` command-line tool (which points to `scripts/training/train.py`) for all training workflows, as it uses the modern, correct configuration system.

## 11. Using Gemini CLI for Large Codebase Analysis

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

## 12. Project Organization

For detailed information on project file organization, initiative planning, and document structure conventions, see <doc-ref type="guide">docs/PROJECT_ORGANIZATION_GUIDE.md</doc-ref>.
