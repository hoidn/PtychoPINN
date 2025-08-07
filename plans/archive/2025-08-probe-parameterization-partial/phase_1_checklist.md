# Phase 1: Core Utilities and Hybrid Probe Generation Checklist

**Initiative:** Probe Parameterization Study
**Created:** 2025-08-01
**Phase Goal:** To build the foundational, standalone tools required for the study: the hybrid probe generator and the core helper functions for probe loading and validation. This phase is self-contained and fully testable.
**Deliverable:** A new script `scripts/tools/create_hybrid_probe.py`, a new module `ptycho/workflows/simulation_utils.py` with helper functions, a complete set of unit tests, and a Jupyter notebook for visual validation.

## ✅ Task List

### Instructions:
1. Work through tasks in order. Dependencies are noted in the guidance column.
2. The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3. Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID  | Task Description                                   | State | How/Why & API Guidance |
| :-- | :------------------------------------------------- | :---- | :--------------------- |
| **Section 0: Preparation & Context Priming** |
| 0.A | **Review Key Documents & APIs**                    | `[D]` | **Why:** To understand project conventions and existing utilities before implementation. <br> **Docs:** Read `docs/DEVELOPER_GUIDE.md` (Section 3 on data contracts), `docs/data_contracts.md` (probe specifications), `ptycho/CLAUDE.md` (core library conventions). <br> **APIs:** Review `numpy.load()`, `skimage.transform.resize`, `np.angle()`, `np.abs()`. |
| 0.B | **Identify Target Files & Dependencies**           | `[D]` | **Why:** Map out exactly what will be created/modified to avoid surprises. <br> **New Files:** `ptycho/workflows/simulation_utils.py`, `scripts/tools/create_hybrid_probe.py`, `tests/workflows/test_simulation_utils.py`, `tests/tools/test_create_hybrid_probe.py`, `notebooks/validate_hybrid_probe.ipynb`. <br> **Dependencies:** Verify scikit-image is in requirements.txt. |
| **Section 1: Core Helper Module Creation** |
| 1.A | **Create simulation_utils.py module structure**    | `[D]` | **Why:** Establish the foundation for reusable probe utilities. <br> **How:** Create `ptycho/workflows/simulation_utils.py` with imports: `import numpy as np`, `from pathlib import Path`, `from typing import Union`, logging setup using `from ptycho.log_config import setup_logging`. <br> **File:** `ptycho/workflows/simulation_utils.py` |
| 1.B | **Implement load_probe_from_source function**      | `[D]` | **Why:** Centralize probe loading logic for consistency across tools. <br> **How:** Handle 3 cases: 1) NumPy array pass-through, 2) .npy file via `np.load()`, 3) .npz file extracting 'probeGuess' key. Add validation: check 2D shape, complex dtype, convert to complex64. Raise descriptive errors for invalid inputs. <br> **Verify:** Function handles all three input types correctly. |
| 1.C | **Implement validate_probe_object_compatibility**  | `[D]` | **Why:** Ensure physical validity - probe must be smaller than object to scan across it. <br> **How:** Check `probe.shape[0] < obj.shape[0] and probe.shape[1] < obj.shape[1]`. Raise `ValueError` with clear message including actual sizes if validation fails. <br> **Example:** "Probe (128x128) is too large for object (64x64). Probe must be smaller than object in both dimensions." |
| 1.D | **Create unit tests for helper functions**         | `[D]` | **Why:** Ensure reliability before building dependent tools. <br> **How:** Create `tests/workflows/test_simulation_utils.py`. Test cases: valid array/file inputs, invalid shapes, non-complex data, missing NPZ keys, probe larger than object. Use `unittest.TestCase` or `pytest`. <br> **Verify:** `python -m pytest tests/workflows/test_simulation_utils.py` passes. |
| **Section 2: Hybrid Probe Tool Implementation** |
| 2.A | **Create create_hybrid_probe.py script structure** | `[D]` | **Why:** Establish the command-line tool for probe mixing. <br> **How:** Create `scripts/tools/create_hybrid_probe.py` with argparse setup. Arguments: `amplitude_source` (required), `phase_source` (required), `output` (default: 'hybrid_probe.npy'), `--visualize` flag, `--normalize` flag. Add logging setup from `ptycho.log_config`. |
| 2.B | **Implement probe loading and validation**         | `[D]` | **Why:** Reuse tested utilities for consistency. <br> **How:** Import and use `load_probe_from_source` from simulation_utils for both amplitude and phase sources. Validate both are 2D complex arrays. Log probe shapes and dtypes for debugging. |
| 2.C | **Implement dimension matching logic**             | `[D]` | **Why:** Probes may have different sizes; need consistent output. <br> **How:** If shapes differ, use `skimage.transform.resize` with `order=3` (cubic interpolation) and `anti_aliasing=True` to resize smaller probe to match larger. Preserve complex nature by resizing real and imaginary parts separately. Log any resizing operations. |
| 2.D | **Implement probe mixing algorithm**               | `[D]` | **Why:** Core functionality - combine amplitude from one probe with phase from another. <br> **How:** Extract amplitude: `amp = np.abs(probe1)`. Extract phase: `phase = np.angle(probe2)`. Combine: `hybrid = amp * np.exp(1j * phase)`. If `--normalize` flag set, scale amplitude to preserve total power: `hybrid *= np.sqrt(np.sum(np.abs(probe1)**2) / np.sum(np.abs(hybrid)**2))`. |
| 2.E | **Add output validation and saving**               | `[D]` | **Why:** Ensure output is physically valid before saving. <br> **How:** Check for NaN/Inf values using `np.isfinite()`. Ensure dtype is complex64. Save using `np.save()`. Log output path and basic statistics (mean amplitude, phase variance). |
| 2.F | **Implement visualization option**                 | `[D]` | **Why:** Allow visual inspection of the hybrid probe. <br> **How:** If `--visualize` flag set, create 2x3 subplot grid showing: amplitude/phase for source 1, amplitude/phase for source 2, amplitude/phase for hybrid. Use `matplotlib.pyplot`. Save as 'hybrid_probe_comparison.png'. |
| 2.G | **Create unit tests for create_hybrid_probe**      | `[D]` | **Why:** Ensure robustness of the command-line tool. <br> **How:** Create `tests/tools/test_create_hybrid_probe.py`. Test the main mixing function (extract it for testability). Test cases: matching dimensions, mismatched dimensions, normalization, invalid inputs. Mock file I/O for testing. |
| **Section 3: Validation Notebook Creation** |
| 3.A | **Create Jupyter notebook structure**              | `[D]` | **Why:** Provide interactive validation environment. <br> **How:** Create `notebooks/validate_hybrid_probe.ipynb`. Structure: 1) Introduction markdown, 2) Import cells, 3) Load test data section, 4) Run hybrid probe creation, 5) Visualization section, 6) Numerical validation section. |
| 3.B | **Implement probe loading and display**            | `[D]` | **Why:** Visualize source probes for comparison. <br> **How:** Load fly64 dataset probes. Create helper function to display probe with amplitude/phase subplots. Include colorbar and proper labels. Use consistent color scaling across comparisons. |
| 3.C | **Add hybrid probe generation example**            | `[D]` | **Why:** Demonstrate the tool usage in notebook. <br> **How:** Show command-line equivalent, then run the mixing algorithm directly in notebook. Display all three probes (two sources + hybrid) side by side for comparison. |
| 3.D | **Add numerical validation checks**                | `[D]` | **Why:** Verify the hybrid probe maintains expected properties. <br> **How:** Check: amplitude matches source 1 (within floating point tolerance), phase matches source 2, total power preservation (if normalized), no NaN/Inf values, maintains complex64 dtype. Display results as pass/fail table. |
| **Section 4: Documentation Updates** |
| 4.A | **Update scripts/tools/CLAUDE.md**                 | `[D]` | **Why:** Document the new tool for future developers. <br> **How:** Add section for `create_hybrid_probe.py`. Include: purpose, usage examples, command-line arguments, expected inputs/outputs, common use cases. Follow existing documentation style in the file. |
| 4.B | **Add docstrings to all new functions**            | `[D]` | **Why:** Maintain code documentation standards. <br> **How:** Add comprehensive docstrings to all functions in simulation_utils.py and create_hybrid_probe.py. Include: purpose, parameters with types, return values, raises section for exceptions, usage examples. Follow NumPy docstring style. |
| **Section 5: Integration Testing** |
| 5.A | **Run all unit tests**                             | `[D]` | **Why:** Verify all components work correctly. <br> **How:** Run `python -m pytest tests/workflows/test_simulation_utils.py tests/tools/test_create_hybrid_probe.py -v`. All tests should pass. Fix any failures before proceeding. |
| 5.B | **Test create_hybrid_probe.py end-to-end**         | `[D]` | **Why:** Verify the command-line tool works as expected. <br> **How:** Using fly64 dataset: `python scripts/tools/create_hybrid_probe.py datasets/fly/fly64_transposed.npz datasets/fly/fly64_transposed.npz --output test_hybrid.npy --visualize`. Verify output file created and visualization saved. |
| 5.C | **Execute validation notebook completely**         | `[D]` | **Why:** Ensure notebook runs without errors and produces expected results. <br> **How:** Run all cells in `notebooks/validate_hybrid_probe.ipynb`. Verify visualizations are clear and numerical checks pass. Save executed notebook with outputs. |

---

## 🎯 Success Criteria

**This phase is complete when:**
1. All tasks in the table above are marked `[D]` (Done).
2. The phase success test passes:
   - All unit tests pass: `python -m pytest tests/workflows/test_simulation_utils.py tests/tools/test_create_hybrid_probe.py`
   - The `create_hybrid_probe.py` script successfully generates a `hybrid_probe.npy` file from known sources
   - The validation notebook visually confirms the hybrid probe has the correct amplitude and phase characteristics
3. No regressions are introduced in the existing test suite.
4. All new code follows project conventions (logging, error handling, documentation).