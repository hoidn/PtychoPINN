# Session Summary: Pty-Chi Integration and Data Normalization Analysis
**Date:** January 22, 2025  
**Focus:** TIKE dataset normalization analysis, pty-chi library integration, and DM vs PIE algorithm comparison

## Overview
This session successfully integrated the pty-chi ptychographic reconstruction library into the PtychoPINN project and uncovered critical insights about data normalization differences between TIKE and FLY datasets. The work involved deep analysis of data scaling, API exploration, format conversions, and algorithm performance comparisons.

## 1. Data Normalization Discovery

### Key Finding: Massive Scaling Difference
Discovered an ~8000x scaling difference between TIKE and FLY datasets:

**TIKE Dataset Characteristics:**
- **Data type:** Amplitude (already square-rooted)
- **Scaling:** Heavily normalized (mean: 0.0135, max: 0.0334)
- **Distribution:** Very consistent per-pattern maxima (CV=7.7%)
- **Processing:** Appears to use global normalization with scale factor ~0.033

**FLY Dataset Characteristics:**
- **Data type:** Raw intensity (photon counts)
- **Scaling:** Natural photon-count scale (mean: 107.6, max: 1123)
- **Distribution:** High variance typical of experimental data
- **Processing:** Minimal preprocessing, integer-like values

### Normalization Scheme Analysis
TIKE appears to use the following normalization pipeline:
1. Convert intensity → amplitude: `sqrt(intensity_data)`
2. Global normalization: Divide by global maximum for [0,1] range
3. Apply scale factor: Multiply by ~0.033 (possibly photon-based)

### Created Scaled Versions
- **30x scaling:** Brought TIKE max to ~1.0 (typical amplitude range)
- **5000x scaling:** Better magnitude match to FLY (max: 167 vs 1123)

## 2. TIKE Input Format Clarification

### Critical Discovery: TIKE Expects Amplitude
Through source code analysis, confirmed that TIKE reconstruction expects **amplitude** as input, not intensity:

```python
# TIKE's Poisson noise model implementation
diff = cp.sqrt(intensity) - cp.sqrt(data)  # data is amplitude input
```

**Implications:**
- FLY dataset needs `sqrt()` conversion before TIKE processing
- TIKE dataset already in correct format (amplitude)
- Explains the smaller values in TIKE datasets

## 3. Pty-Chi Library Integration

### Successfully Integrated Alternative Reconstruction Method

**API Exploration Findings:**
- Main entry point: `PtychographyTask` class
- Multiple algorithms available: DM, PIE, LSQML, Autodiff
- GPU acceleration with automatic device detection
- Requires specific data formats and conventions

### Data Format Conversions Required
Successfully identified and implemented necessary conversions:

1. **Amplitude → Intensity:** `intensity = amplitude²`
2. **Position Centering:** Zero-mean positions required by pty-chi
3. **Probe Dimensionality:** 2D → 4D tensor (n_opr_modes, n_modes, h, w)
4. **Coordinate Convention:** PtychoPINN [X,Y] → pty-chi [Y,X]

### Implementation Details
Created `ptychi_reconstruct_tike.py` with:
- Automatic data format conversion
- Configurable algorithm selection (DM, LSQML, PIE)
- GPU acceleration support
- Proper convergence parameters (200+ epochs)
- Visualization output

## 4. Algorithm Performance Analysis

### DM vs PIE Comparison

**DM (Difference Map) Algorithm:**
- **Memory:** High (stores full exit wave)
- **Speed:** ~15 epochs/second on GPU
- **Convergence:** Fewer epochs needed (typically 8-20)
- **Stability:** High, fewer parameters to tune
- **Best for:** Small-medium datasets, stability priority

**PIE (Ptychographic Iterative Engine):**
- **Memory:** Moderate (batch processing)
- **Speed:** Theoretically faster with batching
- **Convergence:** More epochs needed (typically 32-100)
- **Parameters:** Requires careful tuning (alpha, step_size)
- **Best for:** Large datasets, memory constraints

### Convergence Insights
**Critical Learning:** 15-20 iterations insufficient for quality reconstruction
- Minimum recommended: 200 epochs
- 200 epochs takes only ~24 seconds on GPU
- Significant quality improvement with proper convergence

## 5. Project Integration Best Practices

### Documentation Updates
Successfully integrated the new script following project conventions:

1. **Script Location:** Moved to `scripts/reconstruction/` following project structure
2. **Documentation Updated:**
   - `CLAUDE.md`: Added to Key Workflows section with XML doc-ref tags
   - `docs/COMMANDS_REFERENCE.md`: Added CLI reference section
   - `scripts/reconstruction/README.md`: Comprehensive usage documentation

3. **Followed Project Patterns:**
   - Used XML tagging: `<doc-ref type="workflow-guide">`
   - Maintained documentation hierarchy
   - Ensured discoverability per project directives

## 6. Technical Challenges and Solutions

### Challenge 1: Systemic PIE Implementation Bug in Pty-Chi
- **Problem:** All PIE algorithm variants (PIE, ePIE, rPIE, mPIE) failed with tensor dimension issues
- **Root Cause:** Fundamental bug in base PIE class `apply_updates()` method at line 278
- **Specific Issue:** Gradient tensor shape mismatch when assigning averaged probe updates
  - Expected: `[1, n_modes, H, W, 2]` (with OPR mode dimension)
  - Actual: `[n_modes, H, W, 2]` (after batch averaging removes dimension)
- **Scope:** Affects ALL PIE variants due to inheritance from buggy base class
- **Solution:** Used DM algorithm which has separate implementation without this bug

#### Detailed PIE Bug Analysis:
The bug occurs in the probe gradient update mechanism:
```python
# pty-chi/src/ptychi/reconstructors/pie.py, line 278
self.parameter_group.probe.set_grad(-delta_p_i.mean(0), slicer=(0, mode_slicer))
```

**Why all PIE variants fail:**
- All variants (ePIE, rPIE, mPIE) inherit from base PIE class
- None override the buggy `apply_updates()` method
- The bug manifests regardless of batch size (tested: 10, 32, 96, 200)
- This is a library-level bug, not a configuration issue

**Inheritance structure showing bug propagation:**
```
BaseIterativeReconstructor
    └── PIE (contains bug in apply_updates)
        ├── EPIE (inherits bug)
        ├── RPIE (inherits bug)
        └── MPIE (inherits bug)
```

### Challenge 2: Path Resolution
- **Problem:** Script needed to work from new location
- **Solution:** Updated pty-chi path to use relative imports from script location

### Challenge 3: Position Convention
- **Problem:** Pty-chi requires zero-centered positions
- **Solution:** Implemented automatic centering in data loader

## 7. Key Learnings

1. **Data Normalization is Critical**
   - Different tools expect different formats (amplitude vs intensity)
   - Scaling differences can be orders of magnitude
   - Always verify data format expectations before processing

2. **Convergence Requires Patience**
   - Initial tests with 20 epochs were insufficient
   - 200+ epochs needed for quality results
   - GPU acceleration makes longer runs practical

3. **Algorithm Selection Involves Trade-offs**
   - No single "best" algorithm - depends on constraints
   - Memory vs speed vs stability considerations
   - Parameter tuning complexity varies significantly

4. **Integration Requires Attention to Detail**
   - Data format conversions often necessary
   - Coordinate conventions may differ
   - Path resolution needs careful handling

5. **Documentation is Essential**
   - Proper integration requires updating multiple docs
   - Following project conventions ensures discoverability
   - Clear usage examples prevent confusion

## 8. Outputs and Artifacts

### Created Files:
- `scripts/reconstruction/ptychi_reconstruct_tike.py` - Main reconstruction script
- `tike_outputs/fly001_reconstructed_final_downsampled/fly001_reconstructed_final_downsampled_data_30x.npz` - 30x scaled dataset
- `tike_outputs/fly001_reconstructed_final_downsampled/fly001_reconstructed_final_downsampled_data_5000x.npz` - 5000x scaled dataset
- `ptychi_tike_reconstruction_converged/` - Reconstruction results

### Documentation Updated:
- `CLAUDE.md` - Added pty-chi workflow reference
- `docs/COMMANDS_REFERENCE.md` - Added pty-chi CLI section
- `scripts/reconstruction/README.md` - Added comprehensive pty-chi documentation

## 9. Future Recommendations

1. **Add CLI Arguments to Script**
   - Currently uses hardcoded parameters
   - Should accept dataset path, algorithm, epochs as arguments
   - Consider YAML configuration support

2. **Report or Fix PIE Bug in Pty-Chi**
   - **Bug Location:** `pty-chi/src/ptychi/reconstructors/pie.py:278`
   - **Issue:** Gradient tensor dimension mismatch in `apply_updates()`
   - **Options:**
     - Submit bug report to pty-chi repository with reproduction steps
     - Fork and fix: Add dimension handling in gradient assignment
     - Patch locally: Override `apply_updates()` with corrected implementation
   - **Impact:** Would enable all PIE variants (ePIE, rPIE, mPIE) for comparison

3. **Standardize Data Preprocessing**
   - Create unified preprocessing pipeline
   - Handle amplitude/intensity conversion automatically
   - Document standard normalization approaches

4. **Benchmark Multiple Datasets**
   - Test reconstruction quality across different normalizations
   - Compare reconstruction times for various dataset sizes
   - Establish best practices for different scenarios

5. **Algorithm Recommendation System**
   - Based on dataset characteristics (size, memory constraints)
   - Automate selection between DM (stability) vs PIE variants (speed)
   - Include fallback to DM when PIE fails

## 10. Three-Way Comparison Integration Analysis

### Feasibility Assessment for Replacing Tike with Pty-Chi

**Finding: SIMPLE DROP-IN REPLACEMENT**

The existing 3-way comparison framework can easily accommodate pty-chi as a replacement for Tike with minimal modifications:

### Required Changes

1. **Primary Integration Point:**
   - `scripts/studies/run_complete_generalization_study.sh` (lines 481-490)
   - Change flag from `--add-tike-arm` to `--add-ptychi-arm`
   - Update command construction to use pty-chi script

2. **Script Enhancement Needed:**
   - `ptychi_reconstruct_tike.py` needs CLI argument support
   - Add argparse similar to `run_tike_reconstruction.py`
   - Parameters: `--input`, `--output-dir`, `--n-images`, `--epochs`, `--algorithm`

3. **Path Reference Updates:**
   - Change from `tike_run/tike_reconstruction.npz` to `ptychi_run/ptychi_reconstruction.npz`
   - Update result aggregation paths

### Output Compatibility

**Key Discovery:** Output formats are compatible!
- Both use `reconstructed_object` as primary key
- Both save as NPZ with complex arrays
- Existing `load_tike_reconstruction()` function works with pty-chi output
- All comparison metrics (SSIM, PSNR, FRC) work identically

### Parameter Mapping

| Tike Parameter | Pty-Chi Equivalent | Notes |
|---|---|---|
| `--iterations` | `--epochs` | Different terminology, same concept |
| `--n-images` | `--n-images` | Direct mapping |
| `--extra-padding` | N/A | Pty-chi handles automatically |
| N/A | `--algorithm` | New option: choose DM/LSQML/PIE |

### Implementation Effort

**Estimated: 3-4 hours total**
- 2-3 hours: Add CLI arguments to pty-chi script
- 1 hour: Update generalization study integration
- 30 minutes: Update paths and parameter names

### Advantages of Pty-Chi Integration

1. **Algorithm Flexibility:** Choose between DM, PIE, LSQML algorithms
2. **GPU Optimization:** Better GPU utilization with PyTorch backend
3. **Active Development:** Pty-chi is actively maintained
4. **Advanced Features:** Position correction, multi-modal support

### Recommendation

The integration would be straightforward and beneficial. Consider:
1. **Keep both options:** Add `--reconstruction-method {tike,ptychi}` flag
2. **Default to pty-chi:** For new studies due to better performance
3. **Maintain backwards compatibility:** Keep Tike option for reproducibility

## Conclusion

This session successfully integrated pty-chi as an alternative reconstruction method for the PtychoPINN project, while uncovering critical insights about data normalization that explain previous discrepancies between datasets. The work established a foundation for comparing different reconstruction algorithms and highlighted the importance of understanding data format expectations when integrating external libraries. 

The analysis shows that pty-chi can be easily integrated into the existing 3-way comparison framework with minimal changes, offering improved performance and flexibility. The proper documentation and project integration ensures this work is discoverable and usable by future users.