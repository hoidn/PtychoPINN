# FNO/Hybrid Full Pipeline Integration

**Created:** 2026-01-27
**Status:** Open (revalidated 2026-01-27)
**Priority:** High
**Related:** `ptycho_torch/generators/fno.py`, `ptycho_torch/model.py`, `ptycho_torch/workflows/components.py`
**Depends on:** FNO_HYBRID_TESTING_GAPS.md (completed 2026-01-27)

## Summary

The FNO and Hybrid U-NO generators have unit tests and basic integration tests, but are **not yet integrated** with the full `PtychoPINN_Lightning` training pipeline. This blocks performance comparison against PINN/CNN baseline and production use for reconstruction.

## Revalidation Notes (2026-01-27)

- `PtychoPINN_Lightning` still builds the legacy CNN path directly and does not select generators via the registry.
- There is still no Lightning-side adapter for FNO/Hybrid input/output layout differences.
- The standalone `grid_lines_torch_runner.py` path remains separate from the Lightning/physics pipeline.

## Current State (after FNO_HYBRID_TESTING_GAPS work)

| Component | Status |
|-----------|--------|
| FNO/Hybrid generators (`fno.py`) | ✅ Unit tested, forward/backward pass verified |
| `HAS_NEURALOPERATOR` flag | ✅ Added, tests skip gracefully |
| `_LossHistoryCallback` | ✅ Added, collects per-epoch loss |
| Synthetic fixture (`synthetic_ptycho_npz`) | ✅ Added to conftest.py |
| Lightning training with CNN | ✅ Working (fixed dataloader bugs) |
| Lightning training with FNO/Hybrid | ❌ **Not integrated** |

## Integration Gap

### Root Cause

The FNO/Hybrid generators have a different input/output contract than `PtychoPINN_Lightning`:

| Aspect | CNN (`PtychoPINN_Lightning`) | FNO/Hybrid Generators |
|--------|------------------------------|------------------------|
| Input | `(B, 1, N, N)` diffraction | `(B, C, N, N)` where C=gridsize² |
| Output | Complex tensor via `compute_loss()` | `(B, N, N, C, 2)` real/imag split |
| Architecture selection | Hardcoded U-Net | Via registry |
| Coordinates | Concatenated to input | Not used (coordinate-free design) |

### What's Needed for Full Integration

1. **Architecture Selection in `PtychoPINN_Lightning`**
   - Modify `__init__` to accept `architecture` parameter
   - Use generator registry to build model instead of hardcoded U-Net
   - Adapt forward pass to handle FNO/Hybrid output format

2. **Input Preprocessing Alignment**
   - FNO/Hybrid expect `(B, C, N, N)` input (gridsize² channels)
   - Current dataloader provides `(B, 1, N, N)` with separate coordinate handling
   - Options: modify dataloader OR modify generators to accept coordinates

3. **Output Format Conversion**
   - FNO/Hybrid output `(B, N, N, C, 2)` for real/imag
   - `compute_loss()` expects complex tensor format
   - Add conversion layer or modify loss computation

4. **Loss Function Compatibility**
   - Current Poisson NLL loss assumes specific tensor layout
   - Verify compatibility with FNO/Hybrid output

5. **Consistency Layer Integration**
   - The spatial consistency layer (stitch-and-extract) is part of PtychoPINN
   - FNO/Hybrid should integrate with this for overlap constraints

## Proposed Implementation Path

### Phase 1: Minimal Integration (MVP)
- Add `architecture` parameter to `PtychoPINN_Lightning`
- For FNO/Hybrid, wrap generator output to match expected format
- Keep physics decoder unchanged

### Phase 2: Native Integration
- Refactor `compute_loss()` to be architecture-agnostic
- Move coordinate handling to preprocessing (not network input)
- Add FNO-specific batch handling for gridsize>1

### Phase 3: Performance Comparison
- Train FNO/Hybrid on same datasets as CNN
- Compare SSIM/MSE metrics
- Document reconstruction quality differences

## Acceptance Criteria

- [ ] `TrainingConfig(model=ModelConfig(architecture='fno'))` trains via Lightning
- [ ] `TrainingConfig(model=ModelConfig(architecture='hybrid'))` trains via Lightning
- [ ] Loss history collected for all architectures
- [ ] Reconstruction quality metrics (SSIM, MSE) computed
- [ ] Performance comparison documented (FNO vs Hybrid vs CNN)

## Bug Fixes from Testing Gap Work

The following bugs were fixed while working on FNO_HYBRID_TESTING_GAPS and are prerequisites for any Lightning training:

| Finding ID | Issue | Resolution |
|------------|-------|------------|
| DATALOADER-EXPID-001 | `experiment_id` tensor shape `(1,)` vs `(batch,)` | Use scalar, collates to `(batch,)` |
| DATALOADER-SCALE-001 | Scaling constants don't broadcast with `(B,C,H,W)` | Use shape `(1,1,1)` → `(B,1,1,1)` |
| LIGHTNING-STRATEGY-001 | Wrong strategy check in `_train_with_lightning` | Use `execution_config.strategy` |

## Related Files

- `ptycho_torch/model.py` - `PtychoPINN_Lightning` class (needs modification)
- `ptycho_torch/generators/fno.py` - FNO/Hybrid generator implementations
- `ptycho_torch/generators/registry.py` - Generator registry
- `ptycho_torch/workflows/components.py` - Training orchestration
- `scripts/studies/grid_lines_torch_runner.py` - CLI runner for FNO/Hybrid

## Estimated Effort

- Phase 1 (MVP): 2-3 days
- Phase 2 (Native): 3-5 days
- Phase 3 (Comparison): 1-2 days

## Notes

The `grid_lines_torch_runner.py` provides a standalone training path that bypasses `PtychoPINN_Lightning`. This could be used for initial FNO/Hybrid evaluation while full integration is pending. However, it lacks the physics-informed loss and consistency layer that are central to PtychoPINN's approach.
