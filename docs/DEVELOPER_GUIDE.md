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

**The Canonical Data Format:** All tools that produce or consume ptychography datasets for training or evaluation **MUST** adhere to the format defined in the official **[Data Contracts Document](./data_contracts.md)**. This document is the single source of truth for array shapes, key names, and data types.

### 3.1. Lesson: Implicit `dtype` is a Time Bomb (The Deepest Bug)

**The Symptom:** The supervised model received real-valued `Y` patches (`float64`) when it expected complex data, causing it to train on amplitude only.

**The Root Cause:** In `ptycho/raw_data.py`, a "canvas" array for assembling complex patches was initialized with `np.zeros(...)` without an explicit `dtype`. NumPy defaults to `float64`, causing the imaginary part of every patch to be silently discarded upon assignment.

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

---

## 4. Physical Consistency in Data Preprocessing

### 4.1. Downsampling: Binning vs. Cropping

**The Lesson:** When downsampling data, the method used must be physically consistent across all related arrays.

**The Discovery:** An initial plan proposed downsampling diffraction patterns and ground truth patches by cropping their centers. This is physically incorrect. A downsampled diffraction pixel represents an average over a detector area. Therefore, the corresponding real-space object patch must also be downsampled via **binning (averaging)** to maintain physical correspondence.

**The Rule:** All real-space arrays (`objectGuess`, `probeGuess`, `Y` patches) must be downsampled using `bin_complex_array`. All k-space arrays (`diffraction`) are downsampled via `crop_center`. The `downsample_data_tool.py` script now correctly implements this.

### 4.2. Simulation Consistency

**The Principle:** The `diffraction` array in a dataset is only physically valid for the specific `objectGuess` and `probeGuess` it was generated from.

**The Rule:** If you modify the object or probe in any way (e.g., upsampling, smoothing via `prepare_data_tool.py`), the original `diffraction` data is now invalid. You **must** run a new simulation to generate a new, valid `diffraction` array. The `prepare.sh` script correctly models this workflow.

---

## 5. Authoritative Methods for Evaluation

To ensure fair and consistent model comparison, the project must use single, authoritative functions for common evaluation tasks.

### 5.1. Patch Reassembly

**The Principle:** Both the PINN and baseline models, when operating on non-grid data, must use the same function for reassembly.

**The Correct Function:** `ptycho.tf_helper.reassemble_position`
*   **What it does:** Places a small central region of each patch onto a large canvas according to its specific, real-valued scan coordinates (`global_offsets`). It correctly handles normalization of overlapping regions.
*   **When to use it:** This is the correct method for visualizing the final output of any model that predicts patches for non-grid scan positions.

### 5.2. Evaluation Alignment

**The Principle:** The logic for aligning a reconstruction with its ground truth for metric calculation must be centralized.

**The Correct Function:** `ptycho.image.cropping.align_for_evaluation`
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

## 6. Testing Conventions

**The Principle:** All tests for the PtychoPINN project must follow a standardized, conventional structure to ensure maintainability, discoverability, and consistency.

### 6.1. Test Directory Structure

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

### 6.2. Running Tests

**The Command:** To run all tests, use the standard unittest discovery from the project root:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

This command will discover and execute all test files following the `test_*.py` naming pattern within the `tests/` directory structure.

### 6.3. Script-Level Tests

**Exception:** Tests for standalone scripts in the `scripts/` directory can be co-located with the script itself (e.g., `scripts/tools/test_update_tool.py`). This exception applies only to command-line scripts, not library modules.

### 6.4. Test File Naming

**The Convention:** All test files must follow the naming pattern `test_<module_name>.py`, where `<module_name>` corresponds to the module being tested.

**Examples:**
- Tests for `ptycho/model.py` → `tests/test_model.py`
- Tests for `ptycho/evaluation.py` → `tests/test_evaluation.py`
- Tests for `ptycho/image/stitching.py` → `tests/image/test_stitching.py`