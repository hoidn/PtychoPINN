# Independent Sampling and Grouping Control Initiative

## Overview

This initiative adds independent control over data subsampling and neighbor grouping operations in PtychoPINN, which are currently coupled through a single `n_images` parameter.

## Status

- **Started**: 2025-08-28
- **Phase**: Phase 4 of 6 Complete (67%)
- **Branch**: `feature/fix-sampling`
- **Estimated Effort**: 3-4 days, ~35 lines of code
- **Actual Progress**: ~120 lines implemented, 4 phases complete
- **Additional improvements**: Made InferenceConfig consistent with TrainingConfig

## Problem Statement

Currently, the `n_images` parameter controls both:
1. How many images to subsample from the dataset
2. How many neighbor groups to create for training

This coupling prevents important use cases:
- **Dense grouping**: Using 2000 images to create 1800 groups
- **Sparse grouping**: Using 10000 images to create 500 groups  
- **Memory management**: Subsampling large datasets before grouping

## Solution

Add a new `--n-subsample` parameter that controls data subsampling independently from `--n-images` (which will control grouping only).

### Key Design Decisions

1. **Minimal changes**: Implement subsampling at data loading stage, not in `raw_data.py`
2. **Backward compatible**: Optional parameter maintains existing behavior
3. **Clear semantics**: Separate concerns of data selection vs. group creation

### Implementation Strategy

```
Current: Dataset → RawData → generate_grouped_data(n_images) → Groups
                              ↑
                        Controls both operations

New: Dataset → [Subsample] → RawData → generate_grouped_data(n_images) → Groups
               ↑                         ↑
          n_subsample              n_images (groups only)
```

## Test Coverage

### Current Test Suite Baseline
- **Total Tests**: 172
- **Passing**: ~140 tests
- **Critical Tests**: ✅ All integration and grouping tests passing

### Regression Testing Strategy
- Continuous monitoring after each phase
- No new test failures allowed
- Key tests tracked throughout implementation:
  - `test_train_save_load_infer_cycle` 
  - `test_backward_compatibility`
  - `test_efficient_grouping_*`

## Files Modified

| File | Changes | Lines | Status |
|------|---------|-------|--------|
| `ptycho/workflows/components.py` | Add subsampling logic | ~15 | ✅ Done |
| `ptycho/config/config.py` | Add n_subsample fields + n_images to InferenceConfig | ~5 | ✅ Done |
| `scripts/training/train.py` | Add CLI argument + interpretation | ~30 | ✅ Done |
| `scripts/inference/inference.py` | Add CLI argument + interpretation | ~35 | ✅ Done |
| `scripts/compare_models.py` | Add CLI argument + interpretation | ~25 | ✅ Done |
| `tests/test_subsampling.py` | New test file | ~100 | ✅ Done |
| `scripts/studies/run_complete_generalization_study.sh` | No changes needed | 0 | ✅ Done |

**Actual Progress**: ~120 lines of production code + 100 lines of tests

## Usage Examples

```bash
# Current (coupled) - unchanged for backward compatibility
python train.py --n-images 1000 --gridsize 2

# New (independent control)
python train.py --n-subsample 4000 --n-images 1000 --gridsize 2

# Dense grouping
python train.py --n-subsample 1200 --n-images 1000 --gridsize 2

# Sparse grouping
python train.py --n-subsample 10000 --n-images 500 --gridsize 2
```

## Success Criteria

- [ ] All existing workflows continue unchanged
- [ ] Can control subsampling and grouping independently
- [ ] No regression in test suite (same pass/fail counts)
- [ ] Performance overhead < 1%
- [ ] Clear documentation and examples

## Related Documents

- [Implementation Plan](implementation.md) - Detailed 6-phase implementation with checklists
- [Test Tracking](test_tracking.md) - Continuous test monitoring (to be created)

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing workflows | Low | High | Optional parameter with None default |
| Confusing parameter interaction | Medium | Medium | Clear documentation and logging |
| Performance regression | Low | Medium | Efficient numpy operations |
| Test failures | Low | High | Comprehensive regression testing |

## Progress Summary

### Completed
1. ✅ Create implementation plan with regression testing
2. ✅ Establish baseline test results
3. ✅ Implement Phase 1 (Core Infrastructure)
   - Added `n_subsample` and `subsample_seed` to configs
   - Updated `load_data()` with subsampling logic
   - Created 11 comprehensive unit tests
4. ✅ Implement Phase 2 (Training Integration)
   - Added `interpret_sampling_parameters()` function
   - Fixed Optional[int] argument parsing
   - All integration tests passing
5. ✅ Implement Phase 3 (Inference Integration)
   - Added `--n-subsample` and `--subsample-seed` arguments
   - Implemented parameter interpretation for inference
   - All backward compatibility maintained
   - Edge cases handled with appropriate warnings
   - **Bonus**: Added `n_images` to `InferenceConfig` for consistency with `TrainingConfig`
   - **Bonus**: Unified function signatures between training and inference scripts
6. ✅ Implement Phase 4 (Comparison Script Updates)
   - Added `--n-subsample` and `--subsample-seed` to compare_models.py
   - Implemented parameter interpretation logging
   - Determined study scripts don't need changes (use --n-test-images appropriately)
   - All tests pass with backward compatibility maintained

### Next Steps
7. ⏳ Phase 5: Documentation and Examples
8. ⬜ Phase 6: Advanced Testing and Validation