# Unified Developer Guide & Architectural Learnings for the PtychoPINN Project

## Document Purpose

This document summarizes key architectural principles, data pipeline best practices, and non-obvious implementation details discovered during the development and debugging of the PtychoPINN project. It is intended as a **canonical guide for all developers** to ensure future work is robust, maintainable, and consistent with the project's design.

---

## 1. The Core Concept: A "Two-System" Architecture

**The Lesson:** The most critical realization for any developer is that the repository contains two distinct, semi-independent systems: a legacy, grid-based system and a modern, coordinate-based system. Many bugs arise from the friction between them.

| Feature            | Legacy "Grid-Based" System                                  | Modern "Coordinate-Based" System                                                                       |
| ------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Primary Workflow**   | `ptycho/train.py`                                           | `scripts/run_comparison.sh`, `scripts/run_baseline.py`, `scripts/inference/inference.py`               |
| **Configuration**  | Legacy `ptycho.params.cfg` global dictionary.               | Modern `ptycho.config.config` dataclasses.                                                             |
| **Patch Reassembly** | `ptycho/image/stitch_patches`                               | `ptycho/tf_helper.reassemble_position`                                                                 |
| **Characteristic** | Relies on global state and implicit configuration.          | Aims for explicit data flow and configuration via function arguments.                                  |

**The Rule:** Before starting any task, identify which system you are operating in. The long-term goal is to migrate all functionality to the modern, coordinate-based system and eliminate the legacy system's reliance on global state.

---

## 2. Critical Architectural Principles & Anti-Patterns

These are fundamental rules to follow to avoid introducing fragile, difficult-to-debug code.

### 2.1. Anti-Pattern: Side Effects on Import

**The Lesson:** A module must **never** perform complex, state-dependent operations (like loading or generating data) at the top level (i.e., when it is imported). This practice leads to unpredictable behavior and untraceable bugs.

**The Discovery:** A `KeyError` was traced back to `from ptycho.generate_data import YY_ground_truth`. This simple import was re-executing an entire data-loading pipeline in a different context, which caused a crash.

**The Correct Pattern:** Design functions to receive all data they need as explicit arguments.

**DON'T:** Trigger complex logic at the module level.
```python
# ptycho/export.py (Incorrect)
from ptycho.generate_data import YY_ground_truth # Hidden side effect!

def save_recons(model_type, stitched_obj):
    if YY_ground_truth is not None: # Fragile, hidden dependency
        ...
```

**DO:** Design functions to receive all data they need as explicit arguments.
```python
# ptycho/export.py (Correct)
def save_recons(model_type, stitched_obj, ground_truth_obj=None):
    if ground_truth_obj is not None: # Dependency is clear and safe
        plt.imsave(..., np.absolute(ground_truth_obj[:, :, 0]))
        ...
```

### 2.2. Anti-Pattern: Implicit Dependencies via Global State

**The Lesson:** Relying on the global configuration dictionary (`ptycho.params.cfg`) makes the codebase fragile and introduces unsafe initialization-order dependencies.

**The Safe Initialization Pattern (Current Workaround):**
Until the codebase is fully refactored, all modern scripts must follow this order:
1.  Set up configuration using the modern `TrainingConfig` dataclasses.
2.  Update the legacy global dictionary: `update_legacy_dict(params.cfg, config)`.
3.  Load any necessary data (e.g., a probe from a file).
4.  Use this data to populate any remaining required keys in `params.cfg` (e.g., `p.set('probe', ...)`).
5.  **Only then**, perform local imports of modules (`ptycho.model`, `ptycho.nbutils`) that depend on this global state.

---

## 3. The Data Pipeline: Contracts and Bookkeeping

A data pipeline's file formats and loading logic constitute a public API. Its behavior must be explicit and robust.

**The Canonical Data Format:** All tools that produce or consume ptychography datasets for training or evaluation **MUST** adhere to the format defined in the official **<doc-ref type="contract">Data Contracts Document (docs/data_contracts.md)</doc-ref>**. This document is the single source of truth for array shapes, key names, and data types.

### 3.1. Lesson: Implicit `dtype` is a Time Bomb (The Deepest Bug)

**The Symptom:** The supervised model received real-valued `Y` patches (`float64`) when it expected complex data, causing it to train on amplitude only.

**The Root Cause:** In `<code-ref type="module">ptycho/raw_data.py</code-ref>`, a "canvas" array for assembling complex patches was initialized with `np.zeros(...)` without an explicit `dtype`. NumPy defaults to `float64`, causing the imaginary part of every patch to be silently discarded upon assignment.

**File:** `ptycho/raw_data.py`
```python
# The Bug: np.zeros() defaults to float64, creating a real-valued canvas.
canvas = np.zeros((B, N, N, c))

for i in range(B * c):
    # translated_patch is a complex tensor
    translated_patch = hh.translate(gt_padded, offset)
    
    # THE ERROR: When the complex patch is assigned to the float canvas,
    # the imaginary part is silently discarded.
    canvas[i // c, :, :, i % c] = np.array(translated_patch)[0, :N, :N, 0]
```
**The Rule:** Always be explicit about the data type when initializing NumPy arrays that will hold non-default types. The fix was a one-line change: `np.zeros(..., dtype=np.complex64)`.

### 3.2. Lesson: The Data File Format is a Strict API

**The Symptom:** The `diffraction` and `Y` arrays in the same `.npz` file had their batch dimensions in different positions (`(H, W, N)` vs. `(N, H, W, C)`), requiring fragile, hard-coded fixes in the data loader.

**The Root Cause:** The `transpose_rename_convert.py` script, responsible for creating the final prepared dataset, was only aware of `diffraction` and did not handle `Y` at all, resulting in an inconsistent file.

**The Rule:** An inconsistent file format is a bug in the script that **generates** it, not a problem to be solved by the script that **loads** it. The long-term solution was to modify `transpose_rename_convert.py` to handle all per-image arrays consistently, ensuring the batch dimension is always in the same position.

### 3.3. Lesson: Prioritize Prepared Data; Fail on Ambiguity

**The Symptom:** The supervised model was ignoring the correctly prepared, downsampled `Y` patches and instead regenerating incorrect, high-resolution patches on-the-fly.

**The Root Cause:** The data loader in `ptycho/raw_data.py` checked for the presence of `objectGuess` *before* checking for the prepared `Y` array. Because `objectGuess` is kept for evaluation, this check always passed, and the incorrect logic path was taken.

**The Rule:** A data loader must not be "helpfully" ambiguous.
1.  **Prioritize the Final Product:** Always check for the most processed, prepared version of the data first (the `Y` array).
2.  **Fail Loudly:** Do not silently fall back to regenerating data. This masks errors. The corrected logic now raises a `ValueError` or `NotImplementedError` if the expected prepared `Y` array is not found, forcing the developer to use a correctly prepared dataset.

### 3.4. Core Tensor Formats for gridsize > 1

To handle overlapping patches, the codebase uses three primary tensor formats. Understanding the role of each is critical for avoiding shape mismatch errors.

* **Channel Format (`B, N, N, C`)**: This is the primary format for **neural network processing**. The `C = gridsize**2` neighboring patches are treated as channels. This is the format produced by `get_image_patches` and expected by the U-Net in `ptycho/model.py`.

* **Flat Format (`B*C, N, N, 1`)**: This format is used for **individual patch physics simulation**. Each of the `C` patches from a group is treated as a separate item in a larger batch. **This is the required input format for `ptycho.diffsim.illuminate_and_diffract`**.

* **Grid Format (`B, G, G, N, N, 1`)**: A transitional format that makes the physical 2D grid of patches explicit.

**CRITICAL RULE:** You must use `ptycho.tf_helper._channel_to_flat()` to convert data from Channel Format to Flat Format before passing it to the core physics simulation engine.

### 3.5. Normalization Architecture: Three Distinct Systems

**The Critical Lesson:** PtychoPINN uses three separate normalization systems that must never be confused. Mixing them is a common source of subtle bugs and incorrect results.

**The Discovery:** A critical misunderstanding about where photon scaling should be applied led to attempting to scale data in `raw_data.py`, which would have broken the `prepare.sh` workflow and caused double-scaling issues.

#### The Three Normalization Systems

1. **Physics Normalization (`intensity_scale`)**
   - **Purpose:** Scales simulated data to match realistic experimental photon counts
   - **Location:** Applied ONLY in the physics loss layer during training
   - **Key Module:** `ptycho/diffsim.py` calculates but does NOT apply the scale
   - **Critical Rule:** Internal pipeline data remains normalized; scaling happens at physics boundary

2. **Statistical Normalization (`normalize_data`)**
   - **Purpose:** Standard ML preprocessing for stable neural network training
   - **Location:** Applied in data loader before model input
   - **Key Module:** `ptycho/loader.py`
   - **Note:** Completely independent from physics normalization

3. **Display/Comparison Scaling**
   - **Purpose:** Visual adjustments for plots and metric calculations
   - **Location:** Applied only in visualization and comparison code
   - **Key Modules:** `ptycho/image/`, comparison scripts
   - **Rule:** Never affects training or physics calculations

#### The Correct Data Flow

```python
# In diffsim.py - Calculate but don't apply
intensity_scale = scale_nphotons(Y_I * probe_amplitude)
X = diffract_obj(Y_I * probe)  # Normalized diffraction patterns
return X, Y_I / intensity_scale, Y_phi, intensity_scale  # Return normalized

# In raw_data.py - Keep data normalized
norm_Y_I = scale_nphotons(X)  # Calculate normalization factor
return RawData(..., X, ...)  # Return NORMALIZED X, not X * norm_Y_I

# In model.py - Apply scaling only at physics boundary
simulated = self.physics_layer(reconstructed) * intensity_scale
loss = poisson_nll(measured, simulated)
```

#### Common Anti-Patterns to Avoid

**Anti-Pattern 1: Applying intensity_scale in data pipeline**
```python
# WRONG - This breaks prepare.sh and causes double-scaling
X_scaled = X * norm_Y_I
return RawData(..., X_scaled, ...)
```

**Anti-Pattern 2: Confusing nphotons effect**
```python
# WRONG - nphotons doesn't directly scale data values
if nphotons == 1e3:
    X = X * 0.001  # DON'T DO THIS
```

**Anti-Pattern 3: Mixing normalization types**
```python
# WRONG - Don't mix physics and statistical normalization
X = normalize_data(X * intensity_scale)  # Confuses two systems
```

**The Rule:** Always document which normalization you're using, keep them separate, and apply physics scaling only at the model's physics boundary. For complete details, see <doc-ref type="guide">docs/DATA_NORMALIZATION_GUIDE.md</doc-ref>.

---

## 4. Physical Consistency in Data Preprocessing

### 4.1. Downsampling: Binning vs. Cropping

**The Lesson:** When downsampling data, the method used must be physically consistent across all related arrays.

**The Discovery:** An initial plan proposed downsampling diffraction patterns and ground truth patches by cropping their centers. This is physically incorrect. A downsampled diffraction pixel represents an average over a detector area. Therefore, the corresponding real-space object patch must also be downsampled via **binning (averaging)** to maintain physical correspondence.

**The Rule:** All real-space arrays (`objectGuess`, `probeGuess`, `Y` patches) must be downsampled using `bin_complex_array`. All k-space arrays (`diffraction`) are downsampled via `crop_center`. The `downsample_data_tool.py` script now correctly implements this.

### 4.2. Simulation Consistency

**The Principle:** The `diffraction` array in a dataset is only physically valid for the specific `objectGuess` and `probeGuess` it was generated from.

**The Rule:** If you modify the object or probe in any way (e.g., upsampling, smoothing via `prepare_data_tool.py`), the original `diffraction` data is now invalid. You **must** run a new simulation to generate a new, valid `diffraction` array. The `prepare.sh` script correctly models this workflow.

### 4.5 Critical Data Flow: The Patch Extraction Pipeline

The process of extracting patches in `ptycho/raw_data.py` involves a critical coordinate system convention that must be respected. Failure to do so will result in incorrect patch extraction.

1. **Offset Creation:** The `offsets_c` tensor is created by stacking `ycoords` and then `xcoords`. This results in a coordinate order of **`[y_offset, x_offset]`**.
2. **Translation Function:** The `ptycho.tf_helper.translate` function expects its `translations` argument to be in **`[dx, dy]`** (i.e., `[x_offset, y_offset]`) order.
3. **The Required Swap:** To ensure correct translation, the coordinate vector **must be swapped** before being passed to the `translate` function.

**Correct Implementation Pattern:**
```python
# offsets_yx has shape (batch, 2) and order [y, x]
offsets_yx = tf.reshape(offsets_f, (-1, 2))

# Swap columns to get [x, y] order for the translate function
offsets_xy = tf.gather(offsets_yx, [1, 0], axis=1)

# Now pass the correctly ordered offsets to the translate function
translated_patches = hh.translate(images, -offsets_xy)
```

The legacy iterative implementation contained a latent bug where this swap was not performed, leading to visually correct but technically transposed translations on symmetric data. All new and refactored code must perform this explicit swap for correctness.

### 4.6. High-Performance Patch Extraction with Memory-Efficient Mini-Batching

The patch extraction process in `ptycho/raw_data.py` has been optimized to use memory-efficient mini-batched operations, solving both performance and memory issues.

**The Problem:** The original batched approach created massive tensors with `tf.repeat(gt_padded, B*c, axis=0)`, causing out-of-memory errors for large datasets (e.g., 4096 patches).

**The Solution:** Mini-batching strategy that processes patches in configurable chunks:
- Processes patches in chunks of `patch_extraction_batch_size` (default: 256)
- Avoids creating massive intermediate tensors
- Maintains good performance while preventing OOM errors

**Implementation Details:**
- **Function:** `<code-ref type="function">ptycho.raw_data._get_image_patches_batched</code-ref>`
- **Config Parameter:** `patch_extraction_batch_size` in ModelConfig controls chunk size
- **Memory Savings:** Peak memory reduced from O(B*c*H*W) to O(mini_batch_size*H*W)
- **Feature Flag:** Controlled by `use_batched_patch_extraction` in ModelConfig (default: True)

**Performance Characteristics:**
- **Memory:** Configurable memory usage via `patch_extraction_batch_size`
- **Speed:** 1.3-1.5x faster than iterative approach (less speedup than pure batched due to chunking)
- **Numerical Precision:** Small differences (<0.002) from iterative version due to TensorFlow's batched translation optimizations

**Usage Notes:**
- Default `patch_extraction_batch_size=256` works well for most GPUs
- Reduce batch size if encountering OOM errors
- Set `mini_batch_size=1` for exact numerical equivalence with iterative version
- See `<code-ref type="test">tests/test_patch_extraction_performance.py</code-ref>` for validation

---

## 5. Authoritative Methods for Evaluation

To ensure fair and consistent model comparison, the project must use single, authoritative functions for common evaluation tasks.

### 5.1. Patch Reassembly

**The Principle:** Both the PINN and baseline models, when operating on non-grid data, must use the same function for reassembly.

**The Correct Function:** `<code-ref type="function">ptycho.tf_helper.reassemble_position</code-ref>`
*   **What it does:** Places a small central region of each patch onto a large canvas according to its specific, real-valued scan coordinates (`global_offsets`). It correctly handles normalization of overlapping regions.
*   **When to use it:** This is the correct method for visualizing the final output of any model that predicts patches for non-grid scan positions.

### 5.2. Evaluation Alignment

**The Principle:** The logic for aligning a reconstruction with its ground truth for metric calculation must be centralized.

**The Correct Function:** `<code-ref type="function">ptycho.image.cropping.align_for_evaluation</code-ref>`
*   **Interface Documentation:**
    ```python
    def align_for_evaluation(
        reconstruction_image: np.ndarray,
        ground_truth_image: np.ndarray,
        scan_coords_yx: np.ndarray,
        stitch_patch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aligns a reconstructed image with a ground truth object for evaluation.
        Uses scan coordinates and the stitching patch size ('M' from reassemble_position)
        to calculate the precise bounding box for a physically correct comparison.
        """
    ```
*   **The Rule:** Any script that calculates metrics (e.g., `run_baseline.py`, `compare_models.py`) **must** use this function to prepare its inputs for `eval_reconstruction`. This guarantees that the comparison is fair and physically meaningful.

---

## 6. Enhanced Centralized Logging

**The Golden Rule:** All logs for a specific run must be stored in a `logs/` subdirectory within that run's main output directory.

### 6.1. The Enhanced Logging Architecture

**The Authoritative Module:** <code-ref type="module">ptycho/log_config.py</code-ref> provides the single source of truth for all logging configuration in the project, now with advanced tee-style logging, stdout capture, and flexible console control.

**Key Features:**
- **Tee-style logging**: Simultaneous console and file output
- **Print statement capture**: All stdout from any module captured to log files
- **Flexible console control**: `--quiet`, `--verbose`, and custom log levels
- **Complete records**: All output preserved in files regardless of console settings

**The Correct Pattern:** All user-facing scripts must call the centralized logging setup function with enhanced options:
```python
from ptycho.log_config import setup_logging
from ptycho.cli_args import add_logging_arguments, get_logging_config
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="My Script")
    # Add standard arguments...
    
    # Add enhanced logging arguments
    add_logging_arguments(parser)
    return parser.parse_args()

def main():
    # Parse arguments and set up configuration first
    args = parse_arguments()
    config = setup_configuration(args, args.config)
    
    # Set up enhanced centralized logging with user options
    logging_config = get_logging_config(args) if hasattr(args, 'quiet') else {}
    setup_logging(Path(config.output_dir), **logging_config)
    
    # Continue with workflow logic...
```

### 6.2. Enhanced Log File Organization & Features

**Directory Structure:** When a workflow executes with `--output_dir my_run`, the logging system creates:
```
my_run/
├── logs/
│   └── debug.log        # Complete record: ALL messages + captured stdout
├── wts.h5.zip          # Model outputs
├── history.dill        # Training history
└── ...                 # Other workflow outputs
```

**Enhanced Log Capabilities:**
- **File (`debug.log`)**: 
  - All logging messages (DEBUG level and above)
  - **All print() statements from any module**
  - Model architecture summaries
  - Data shape information
  - Debug output from core modules
  - **Complete execution record**
- **Console**: Flexible control via command-line flags
  - Default: INFO level and above
  - `--quiet`: Only external output (TensorFlow warnings, etc.)
  - `--verbose`: DEBUG level output to console
  - `--console-level WARNING`: Custom console filtering

**Print Statement Capture:** The enhanced logging system automatically captures ALL stdout output (including print statements from any imported module) and writes it to the log file, ensuring complete traceability of execution.

### 6.3. Command-Line Logging Options

**Standard Logging Arguments:** All scripts supporting enhanced logging provide these options:

```bash
# Quiet mode: suppress console output from logging system
ptycho_train --train_data datasets/fly64.npz --output_dir my_run --quiet

# Verbose mode: show DEBUG messages on console  
ptycho_train --train_data datasets/fly64.npz --output_dir my_run --verbose

# Custom console log level
ptycho_train --train_data datasets/fly64.npz --output_dir my_run --console-level WARNING
```

**Use Cases:**
- **Interactive development**: Default or `--verbose` mode for real-time feedback
- **Automation/CI**: `--quiet` mode to reduce noise in automated workflows
- **Custom filtering**: `--console-level` for specific debugging scenarios

**Important:** These flags only affect console output. All messages are ALWAYS captured in the `logs/debug.log` file regardless of the console settings.

### 6.4. Anti-Pattern: Local Logging Configuration

**DON'T:** Add local `logging.basicConfig()` or manual handler setup to new scripts:
```python
# WRONG - Creates inconsistent log file locations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('my_script_debug.log')  # Creates root-level log files
    ]
)
```

**DO:** Use the centralized system:
```python
# CORRECT - Consistent log organization
from ptycho.log_config import setup_logging
setup_logging(output_dir)
```

### 6.4. Migration from Legacy Logging

**Historical Context:** Prior to the centralized logging system, scripts created log files directly in the project root (e.g., `train_debug.log`, `inference.log`). This pattern has been deprecated in favor of organized, per-run logging within output directories.

**Rule for New Development:** Any new script that needs logging must use `<code-ref type="module">ptycho/log_config.py</code-ref>`. Adding local logging configuration is now an anti-pattern that violates the single source of truth principle.

---

## 7. Testing Conventions

**The Principle:** All tests for the PtychoPINN project must follow a standardized, conventional structure to ensure maintainability, discoverability, and consistency.

### 7.1. Test Directory Structure

**The Rule:** All tests for the `ptycho` library code must reside in the top-level `tests/` directory, with a structure that mirrors the `ptycho/` package organization.

**Directory Structure:**
```
tests/
├── __init__.py
├── test_model_manager.py      # Tests for ptycho/model_manager.py
├── test_misc.py               # Tests for ptycho/misc.py  
├── image/
│   ├── __init__.py
│   ├── test_cropping.py       # Tests for ptycho/image/cropping.py
│   └── test_registration.py   # Tests for ptycho/image/registration.py
└── workflows/
    ├── __init__.py
    └── test_components.py     # Tests for ptycho/workflows/components.py
```

**Example:** A test for `ptycho/image/cropping.py` must be located at `tests/image/test_cropping.py`.

### 7.2. Running Tests

**The Command:** To run all tests, use the standard unittest discovery from the project root:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

This command will discover and execute all test files following the `test_*.py` naming pattern within the `tests/` directory structure.

### 7.3. Script-Level Tests

**Exception:** Tests for standalone scripts in the `scripts/` directory can be co-located with the script itself (e.g., `scripts/tools/test_update_tool.py`). This exception applies only to command-line scripts, not library modules.

### 7.4. Test File Naming

**The Convention:** All test files must follow the naming pattern `test_<module_name>.py`, where `<module_name>` corresponds to the module being tested.

**Examples:**
- Tests for `ptycho/model.py` → `tests/test_model.py`
- Tests for `ptycho/evaluation.py` → `tests/test_evaluation.py`
- Tests for `ptycho/image/stitching.py` → `tests/image/test_stitching.py`

---

## 8. Data Handling for Overlap-Based Training

The gridsize parameter controls the use of overlapping scan positions in the physics model. The data loading pipeline now uses a unified sampling strategy for all gridsize values.

### 8.1. Unified Sampling Strategy

As of the latest update, the pipeline uses a consistent "sample-then-group" strategy for all gridsize values:

1. **Random Sampling of Anchor Points**: The system randomly samples N anchor points from the complete set of scan coordinates.
2. **Neighbor Grouping**: For gridsize > 1, it finds the K-nearest neighbors for each anchor point to form groups. For gridsize = 1, each anchor point becomes a single-element group.

This unified approach eliminates the previous special-casing for gridsize=1 and ensures consistent behavior across all configurations.

**Key Changes:**
- **No manual shuffling required for gridsize=1**: The system now handles random sampling internally for all gridsize values.
- **Sequential sampling option**: Use the `--sequential_sampling` flag to get the old sequential behavior (first N images) for any gridsize.
- **Pre-shuffled datasets still work**: Existing shuffled datasets will continue to function correctly.

### 8.2. Tensor Shape and Configuration

When gridsize=2, the input tensors to the model will have a channel dimension of 4 (gridsize²). The data loader (`ptycho/loader.py`) and the model (`ptycho/model.py`) are designed to handle this multi-channel format.

The training log will confirm the configuration with a message similar to the following:

```
INFO - Parameter interpretation: --n-images=500 refers to neighbor groups (gridsize=2, total patterns=2000)
```

This indicates the system is generating 500 training samples, where each sample consists of a 2x2 group of neighboring diffraction patterns.

---

## 9. Troubleshooting References

### 9.1. Common Gotchas and Solutions

For detailed documentation of critical issues and solutions discovered during development:

- **GridSize Inference Issues**: Comprehensive guide to configuration loading, initialization order, and multi-channel data handling issues (see section 10.1 below).

---

## 10. Architectural Learnings & Historical Context

This section captures key architectural decisions and lessons learned from past development initiatives. This context is preserved to explain the reasoning behind certain design patterns in the codebase.

### 10.1. The GridSize Inference Issue (Resolved)

A critical issue was discovered where models trained with `gridsize > 1` would fail during inference. The root cause was a combination of initialization order bugs and an implicit dependency on the global `ptycho.params.cfg` state.

#### The Problem
- **Model Creation**: Lambda layers were constructed with a default `gridsize=1` during import, before the correct configuration could be loaded from the model's saved parameters
- **Global State Dependency**: The `_flat_to_channel` function in `<code-ref type="module">ptycho/tf_helper.py</code-ref>` relied on global configuration that wasn't preserved during model serialization
- **Initialization Order**: Module imports triggered model construction before configuration was properly set

#### The Solution Pattern
Multiple interconnected fixes were required:

1. **Configuration Loading Order**: Ensure parameters are loaded and applied **before** TensorFlow model loading in `<code-ref type="module">ptycho/model_manager.py</code-ref>`
2. **Delayed Module Imports**: Avoid importing model-constructing modules at top level in inference scripts
3. **Multi-Channel Data Preservation**: Fix data loading logic in `<code-ref type="module">ptycho/loader.py</code-ref>` to preserve channel dimensions for `gridsize > 1`
4. **Explicit Configuration Flow**: Use modern dataclass-based configuration instead of global state where possible

#### Key Anti-Patterns Identified
- **Never use training config loaders for inference** - They have different parameter priorities
- **Never import model-constructing modules at top level** - Delays model construction until configuration is finalized  
- **Never load models before setting configuration** - Global state must be updated before TensorFlow operations
- **Never assume single-channel tensor shapes** - Always preserve and validate channel dimensions

#### Lessons Learned
- **Avoid side effects on module import** and minimize reliance on global state
- **Configuration should be explicitly passed** wherever possible rather than accessed globally
- **Initialization order matters** - parameters → configuration → model loading → inference
- **Debug logging is essential** for tracing parameter flow and identifying initialization issues

*This architectural learning was derived from extensive debugging of gridsize>1 inference failures and represents a fundamental understanding of the initialization dependencies in the TensorFlow model loading pipeline.*

