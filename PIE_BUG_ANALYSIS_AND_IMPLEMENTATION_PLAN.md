# PIE Bug Analysis and Implementation Plan

## Executive Summary

The PIE algorithm bug documented in January 2025 **still exists** in the current pty-chi installation. However, the situation is more nuanced than initially reported - some PIE variants work while others fail.

## Test Results

### Working Algorithms ✅
- **ePIE** - Works with all batch sizes
- **rPIE** - Works with all batch sizes  
- **PIE with batch_size=1** - Works but impractical for large datasets
- **DM** - Already known to work (different implementation)
- **LSQML** - Already known to work (different implementation)

### Failing Algorithms ❌
- **Base PIE** - Fails with batch_size > 1
- **mPIE** - Not available in API

## Bug Analysis

### Root Cause
The bug occurs at line 278 in `pty-chi/src/ptychi/reconstructors/pie.py`:
```python
self.parameter_group.probe.set_grad(-delta_p_i.mean(0), slicer=(0, mode_slicer))
```

**Issue**: Tensor dimension mismatch when averaging probe gradients
- `delta_p_i.mean(0)` produces shape `[1, H, W, 2]` after batch averaging
- But gradient assignment expects shape compatible with actual batch size `[batch_size, H, W, 2]`

### Why Some Variants Work

1. **ePIE and rPIE override `apply_updates()`** - They implement their own gradient application logic that doesn't have this bug
2. **Base PIE inherits the buggy method** - Direct instantiation of PIE class uses the buggy implementation
3. **Batch size dependency** - Bug only manifests with batch_size > 1

## Implementation Strategy

### Phase 1: Immediate Integration (2-3 hours)
Use **ePIE as the default PIE variant** since it works correctly:

1. **Update `ptychi_reconstruct_tike.py`**
   - Change default algorithm from 'PIE' to 'ePIE'
   - Add CLI arguments for algorithm selection
   - Document ePIE as recommended PIE variant

2. **Add algorithm selection logic**
   ```python
   if algorithm == 'PIE':
       # Warn about bug and suggest ePIE
       print("WARNING: Base PIE has a bug with batch_size > 1. Using ePIE instead.")
       options = api.EPIEOptions()
   ```

3. **Update documentation**
   - Note PIE bug in README
   - Recommend ePIE or rPIE for PIE-based reconstruction
   - Document batch_size=1 workaround for base PIE if needed

### Phase 2: Three-Way Comparison Integration (3-4 hours)

1. **Enhance CLI interface**
   ```bash
   python ptychi_reconstruct_tike.py \
       --input <data.npz> \
       --output-dir <output> \
       --algorithm ePIE \
       --n-images 1000 \
       --epochs 200
   ```

2. **Integrate with generalization study**
   - Update `run_complete_generalization_study.sh`
   - Add `--add-ptychi-arm` flag
   - Use ePIE as default algorithm

3. **Ensure output compatibility**
   - Verify NPZ format matches expected structure
   - Test with existing comparison scripts

### Phase 3: Optional Bug Fix (If Needed)

If base PIE is specifically required, create a local patch:

```python
# Override apply_updates in a custom PIE class
class FixedPIE(api.PIEReconstructor):
    def apply_updates(self, delta_o, delta_p_i, delta_pos):
        # Fix: Don't average if batch_size == 1
        if delta_p_i is not None:
            mode_slicer = self.parameter_group.probe._get_probe_mode_slicer(None)
            if delta_p_i.shape[0] == 1:
                grad = -delta_p_i[0]  # No averaging needed
            else:
                grad = -delta_p_i.mean(0, keepdim=True)  # Keep batch dimension
            self.parameter_group.probe.set_grad(grad, slicer=(0, mode_slicer))
            self.parameter_group.probe.optimizer.step()
        # ... rest of method
```

## Recommendations

### Immediate Actions
1. **Use ePIE or rPIE** for all PIE-based reconstructions
2. **Document the bug** in project documentation
3. **Add warning** when users select base PIE algorithm

### Long-term Actions
1. **Report bug upstream** to pty-chi maintainers with reproduction script
2. **Consider contributing fix** if pty-chi accepts PRs
3. **Monitor pty-chi updates** for official fix

## Testing Checklist

- [x] Verify ePIE works with various batch sizes
- [x] Verify rPIE works with various batch sizes
- [x] Confirm base PIE fails with batch_size > 1
- [x] Confirm base PIE works with batch_size = 1
- [ ] Test ePIE with actual TIKE dataset
- [ ] Verify output format compatibility
- [ ] Test integration with comparison pipeline

## Conclusion

The PIE bug persists but is **not a blocker** for integration. Using ePIE or rPIE provides full PIE functionality without the bug. The integration can proceed immediately with these alternative algorithms.