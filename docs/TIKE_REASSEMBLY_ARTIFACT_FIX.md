# Tike Reassembly Artifact Fix Implementation Guide

**Created:** 2025-07-25  
**Status:** Planned - To be implemented after Tike Integration Phase 2  
**Related Initiative:** [Tike Comparison Integration](../plans/active/tike-comparison-integration/)

## Overview

This document outlines the implementation plan for fixing visual reassembly artifacts that may appear in Tike reconstructions when processed through PtychoPINN's comparison pipeline. The fix addresses boundary handling issues in TensorFlow Addons' `translate` function during patch reassembly operations.

## Problem Context

### Root Cause
The issue stems from the default boundary handling behavior in `tensorflow_addons.image.translate()`:
- **Default behavior**: Uses reflection or other fill modes that create artificial edges
- **Manifestation**: "Blocky, dislocated grid patterns" in reconstructed images
- **Location**: `ptycho/tf_helper.py` in `shift_and_sum()` functions during patch translation

### When This Affects Tike
- **Phase 1 (Current)**: ❌ No impact - Tike handles reconstruction internally via `tike.ptycho.reconstruct()`
- **Phase 2 (Planned)**: ⚠️ Potential impact - If Tike outputs require PtychoPINN reassembly/processing
- **Registration**: ✅ No impact - Uses Fourier-domain shifting, not TensorFlow Addons translate

### Investigation Results (2025-07-25)
**Zero-Padding Experiment**: Attempted to implement larger canvas approach for better reconstruction quality:
- **Canvas Strategy**: Create 1.5x-2x minimum size canvas with zero padding
- **Coordinate Adjustment**: Center scan positions on larger canvas
- **Result**: ❌ **NaN convergence failure** - algorithm produces infinite/NaN values
- **Root Cause**: Manual coordinate offsetting breaks Tike's internal coordinate system relationships

## Implementation Strategy

### Timing: After Phase 2 Completion

**Wait for Phase 2 implementation** of the Tike comparison integration before applying this fix. The fix should only be implemented if/when:

1. Phase 2 introduces Tike reconstruction processing through PtychoPINN's reassembly functions
2. Visual artifacts are actually observed in Tike comparison outputs
3. The artifacts can be traced to the `reassemble_position` → `shift_and_sum` → `translate` call chain

### Non-Invasive Fix Approach

Follow the established project principle of **not modifying core `ptycho/tf_helper.py`**. Instead, implement targeted fixes only where Tike processing occurs.

#### Option A: Wrapper Functions in compare_models.py (Preferred)

If Phase 2 integrates Tike processing into `scripts/compare_models.py`:

```python
# Add to scripts/compare_models.py after Phase 2 integration
# --- START: TIKE-SPECIFIC REASSEMBLY ARTIFACT FIX ---

from tensorflow_addons.image import translate as _translate
from ptycho.tf_helper import _tf_pad_sym

def tike_fixed_translate(imgs: tf.Tensor, offsets: tf.Tensor, **kwargs: Any) -> tf.Tensor:
    """Tike-specific corrected translate with constant zero padding."""
    return _translate(imgs, offsets, interpolation='bilinear', 
                     fill_mode='constant', fill_value=0.)

@tf.function(reduce_retracing=True)
def tike_fixed_shift_and_sum(obj_tensor, global_offsets, M=20):
    """Tike-specific shift-and-sum using corrected translate."""
    # [Implementation similar to fixed_shift_and_sum from analysis]
    # ... [cropping, padding, offset adjustment logic] ...
    translated = tike_fixed_translate(imgs_padded, offsets_flat)
    return tf.reduce_sum(translated, axis=0)

def tike_fixed_reassemble_position(obj_tensor, global_offsets, M=20):
    """Tike-specific reassembly function."""
    ones = tf.ones_like(obj_tensor)
    numerator = tike_fixed_shift_and_sum(obj_tensor, global_offsets, M=M)
    denominator = tike_fixed_shift_and_sum(ones, global_offsets, M=M)
    return numerator / (1e-9 + denominator)

# --- END: TIKE-SPECIFIC FIX ---
```

**Usage Pattern:**
```python
# Only when processing Tike reconstruction that requires reassembly
if tike_recon_path and requires_reassembly:
    tike_processed = tike_fixed_reassemble_position(tike_patches, offsets)
else:
    # Use standard processing for PINN/baseline (unchanged)
    pinn_recon = reassemble_position(pinn_patches, offsets)
```

#### Option B: Dedicated Tike Processing Module

If Tike processing becomes complex, create a separate module:

```python
# New file: ptycho/tike_integration.py
def process_tike_reconstruction_for_comparison(tike_recon_data, comparison_params):
    """Process Tike reconstruction data for fair comparison with ML models."""
    # Include artifact-free reassembly if needed
    # Apply same registration/alignment as ML models
    # Return processed data ready for metrics calculation
```

### Implementation Checklist

**Prerequisites:**
- [ ] Phase 2 of Tike integration is complete
- [ ] Tike reconstruction artifacts are observed and confirmed
- [ ] Artifacts traced to `translate` boundary handling

**Implementation Steps:**
1. [ ] **Identify scope**: Determine exactly where Tike processing uses PtychoPINN reassembly
2. [ ] **Create wrapper functions**: Implement Tike-specific versions with correct boundary handling
3. [ ] **Selective application**: Apply fix only to Tike processing pathways
4. [ ] **Preserve existing workflows**: Ensure PINN/baseline comparisons remain unchanged
5. [ ] **Test verification**: Confirm artifacts are eliminated in Tike outputs
6. [ ] **Documentation**: Update comparison guide with three-way workflow including fix

### Testing Strategy

**Before/After Validation:**
1. Generate comparison plots with current (potentially artifacted) Tike processing
2. Apply the fix to Tike-specific pathways only
3. Regenerate comparison plots and verify:
   - ✅ Tike reconstructions appear smooth without grid artifacts
   - ✅ PINN and baseline reconstructions unchanged
   - ✅ Quantitative metrics (SSIM, MAE) improve for Tike comparisons
   - ✅ No regressions in existing two-way comparisons

**Test Cases:**
- Three-way comparison with artifact-prone dataset
- Backward compatibility test (two-way comparison without `--tike_recon_path`)
- Edge cases with extreme scan position offsets

## Risk Mitigation

### Avoiding Unintended Changes

**Critical Rules:**
1. **Never modify core `ptycho/tf_helper.py`** - maintains stability of physics implementation
2. **Isolate Tike-specific fixes** - prevent impact on PINN/baseline workflows  
3. **Test backward compatibility** - ensure existing comparisons work identically
4. **Document changes clearly** - make Tike-specific nature explicit in code comments

### Rollback Plan

If the fix causes issues:
1. **Git revert**: Each fix should be a separate commit for easy rollback
2. **Feature flag**: Consider making the fix optional via command-line flag
3. **Fallback**: Always maintain ability to process Tike without the fix

## Coordinate System Analysis

### Tike Coordinate Convention Investigation
**Analysis Date**: 2025-07-25

**Key Findings**:
1. **Tike uses [Y, X] convention**: `scan[:, 0]` = Y coordinates (row/vertical), `scan[:, 1]` = X coordinates (col/horizontal)
2. **PtychoPINN uses [X, Y] convention**: Different from Tike's convention
3. **Current implementation handles this correctly**: 
   ```python
   scan = np.stack([
       data_dict['ycoords'].astype(tike.precision.floating),  # Y coords in column 0
       data_dict['xcoords'].astype(tike.precision.floating)   # X coords in column 1
   ], axis=1)
   ```
   The NPZ files store xcoords and ycoords separately, so we stack them in Tike's expected [Y, X] order
3. **Coordinate system is tightly coupled**: Tike's `get_padded_object()` creates specific coordinate relationships that internal algorithms expect
4. **Manual offsetting breaks algorithms**: When we offset coordinates to center on larger canvas, we disrupt internal coordinate assumptions

### Zero-Padding Implementation Failure Analysis
**Attempted Implementation**:
```python
# Calculate minimum canvas size
min_psi_2d, min_scan = tike.ptycho.object.get_padded_object(scan=scan, probe=probe)
min_size = min_psi_2d.shape[0]

# Create larger canvas (1.5x or 2x)
canvas_size = int(min_size * 1.5)
psi_2d = np.zeros((canvas_size, canvas_size), dtype=tike.precision.cfloating)

# Center scan positions on larger canvas  
scan_offset = (canvas_size - min_size) // 2
scan = min_scan + scan_offset  # ❌ THIS BREAKS TIKE'S COORDINATE SYSTEM
```

**Failure Symptoms**:
- NaN values in object updates from first iteration
- Infinite probe rescaling factors  
- Poisson cost becomes NaN
- Algorithm cannot converge

**Root Cause**: Tike's internal algorithms assume specific relationships between:
- Scan position ranges and object boundaries
- Coordinate normalization (minimum position becomes [1,1])
- Probe/object overlap calculations
- Boundary condition handling

## Alternative Approaches

### Recommended: Use Tike's Standard Padding

**Current Working Approach**:
```python
# Use Tike's automatic coordinate-consistent padding
psi_2d, scan = tike.ptycho.object.get_padded_object(scan=scan, probe=probe)
```
- ✅ **Maintains coordinate relationships** that algorithms expect
- ✅ **Proven stable convergence** with proper error metrics
- ✅ **No manual coordinate transformation** required

### Future Zero-Padding Investigation

For future attempts at canvas optimization:
1. **Study Tike's coordinate system deeper**: Understand all coordinate transformations in `get_padded_object()`
2. **Investigate object initialization**: Try larger initial object support instead of canvas resizing
3. **Post-processing approach**: Apply padding after reconstruction rather than before
4. **Parameter exploration**: Use Tike's `extra` parameter in `get_padded_object(scan, probe, extra=N)`

### If Phase 2 Doesn't Require Reassembly

If Phase 2 implementation loads Tike reconstructions as final objects (not patches requiring reassembly), then:
- **No fix needed** - artifacts won't occur
- **This document becomes obsolete** - archive for reference
- **Monitor registration step** - ensure Fourier-domain alignment doesn't introduce artifacts

### If Artifacts Appear Elsewhere

If artifacts appear in other parts of the Tike processing pipeline:
- **Trace the source** - identify specific `translate` calls causing issues
- **Apply targeted fixes** - create minimal wrapper functions
- **Follow same principles** - non-invasive, Tike-specific, well-documented

## Success Criteria

**Fix is successful when:**
- ✅ Tike reconstructions in comparison plots are visually artifact-free
- ✅ Three-way comparison metrics show improvement in Tike evaluation
- ✅ Existing PINN/baseline workflows remain unchanged
- ✅ Performance impact is minimal (fix uses same batched operations)
- ✅ Code maintainability is preserved (clear separation of concerns)

## References

- **Analysis Document**: `DEBUGGING_SESSION_TENSOR_SHAPE_MISMATCH_2025-07-22.md` (contains detailed technical analysis)
- **Tike Integration Plan**: `plans/active/tike-comparison-integration/plan.md`
- **Model Comparison Guide**: `docs/MODEL_COMPARISON_GUIDE.md`
- **Core TF Helper**: `ptycho/tf_helper.py` (functions: `shift_and_sum`, `translate`, `reassemble_position`)
- **Tike Source Code**: `tike/src/tike/ptycho/object.py` (coordinate system analysis)
- **Zero-Padding Experiments**: `scripts/reconstruction/run_tike_reconstruction.py` (coordinate investigation)

---

**Investigation Summary (2025-07-25)**:
1. ✅ **Coordinate convention verified**: [Y, X] format is correct for Tike
2. ❌ **Zero-padding approach failed**: Manual coordinate offsetting breaks internal algorithms
3. ✅ **Standard Tike padding works**: Maintains coordinate relationships for stable convergence
4. 📋 **Future work**: Investigate Tike's `extra` parameter or post-processing approaches

**Next Steps:**
1. Complete Phase 2 of Tike integration using standard padding
2. Test for artifacts in Tike comparison outputs  
3. If artifacts confirmed, implement fix according to this guide
4. Consider alternative canvas optimization approaches if needed