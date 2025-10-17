# Independent Sampling Control Initiative - Phase 5 Context

## Current State Summary (End of Phase 4)

### What Has Been Implemented

#### Core Functionality (Phases 1-4)
1. **Core Infrastructure** (Phase 1)
   - Added `n_subsample` and `subsample_seed` to `TrainingConfig` and `InferenceConfig`
   - Modified `load_data()` in `ptycho/workflows/components.py` to handle independent sampling
   - Created comprehensive test suite in `tests/test_subsampling.py`

2. **Training Integration** (Phase 2)
   - Added `interpret_sampling_parameters()` function to `scripts/training/train.py`
   - Handles three modes: legacy, independent control, and default
   - Proper warning system for problematic configurations

3. **Inference Integration** (Phase 3)
   - Added similar functionality to `scripts/inference/inference.py`
   - **Bonus**: Made `InferenceConfig` consistent with `TrainingConfig` by adding `n_images` field
   - Unified function signatures between training and inference

4. **Comparison Script Updates** (Phase 4)
   - Updated `scripts/compare_models.py` with n_subsample support
   - Determined study scripts don't need changes (they appropriately use `--n-test-images`)

### Key Implementation Details Not in Initiative Docs

#### Parameter Interpretation Logic
The system now has three distinct modes across all scripts:

1. **Legacy Mode**: When only `n_images` is specified
   - For gridsize=1: n_images = individual images
   - For gridsize>1: n_images = number of groups (total patterns = n_images × gridsize²)

2. **Independent Control Mode**: When `n_subsample` is specified
   - `n_subsample` controls initial data selection from dataset
   - `n_images` controls how many images/groups to use for training/inference
   - If `n_images` not specified with `n_subsample`, defaults to n_subsample value

3. **Default Mode**: When neither parameter specified
   - Training: Uses default n_images=512
   - Inference: Uses full dataset
   - Comparison: Uses full dataset

#### Warning Conditions
The system warns when:
- `n_subsample < n_images × gridsize²` (may not have enough data for requested groups)
- `n_subsample > dataset_size` (will use full dataset)

#### Function Signatures
All three scripts now have consistent `interpret_sampling_parameters()` functions:
- Training: Takes `TrainingConfig`, returns `(n_subsample, n_images, message)`
- Inference: Takes `InferenceConfig`, returns `(n_subsample, n_images, message)`
- Comparison: Inline logic, logs interpretation message

### Documentation Already Updated

Before Phase 5, we've already updated:
1. **CONFIGURATION.md**: Added all new parameters with descriptions and examples
2. **COMMANDS_REFERENCE.md**: Added examples and new "Independent Sampling Control" section
3. **Initiative docs**: Updated implementation.md, test_tracking.md, and README.md

### What Phase 5 Needs to Address

According to the implementation plan, Phase 5 should:
1. Create example scripts showing different use cases
2. Update any remaining user documentation
3. Add migration guide from old to new syntax
4. Ensure all CLI help messages are clear

### Specific Gaps for Phase 5

#### Example Scripts Needed
1. **Dense grouping example**: Show how to use most of subsampled data
2. **Sparse grouping example**: Show large subsample with fewer groups
3. **Memory-constrained example**: Show limiting data loading
4. **Migration example**: Show old vs new syntax

#### Documentation Gaps
1. **No dedicated user guide** for the new feature (just scattered in existing docs)
2. **No migration guide** for users updating their scripts
3. **No troubleshooting section** for common issues

#### Help Text Review Needed
All scripts have help text, but should verify:
- Consistency across scripts
- Clarity about parameter interaction
- Examples in help text itself

### Technical Details for Phase 5

#### Current Help Text
- Training: "Number of images to subsample from training data (independent control)"
- Inference: "Number of images to subsample from test data (independent control)"
- Comparison: "Number of images to subsample from test data (independent control)"

#### Log Messages Being Generated
- "Legacy mode: using X individual images (gridsize=1)"
- "Legacy mode: --n-images=X refers to neighbor groups (gridsize=Y, approx Z patterns)"
- "Independent sampling control: subsampling X images, using Y for [training/inference]"
- "Using all test data (no subsampling or grouping limit)"

#### Edge Cases Handled
- n_subsample > dataset size → uses full dataset
- n_subsample < required for groups → warning but continues
- n_images not specified with n_subsample → defaults to n_subsample
- Both parameters None → uses defaults (512 for training, all for inference)

### Testing Status

All tests passing:
- 11 unit tests in `tests/test_subsampling.py`
- Integration tests for all scripts
- Backward compatibility verified
- Edge cases tested

### File Changes Summary

Total changes: ~120 lines of production code + 100 lines of tests

Files modified:
- `ptycho/workflows/components.py`: ~15 lines
- `ptycho/config/config.py`: ~5 lines
- `scripts/training/train.py`: ~30 lines
- `scripts/inference/inference.py`: ~35 lines (including consistency fix)
- `scripts/compare_models.py`: ~25 lines
- `tests/test_subsampling.py`: ~100 lines (new file)

### Notes for Phase 5 Implementation

1. **Example scripts location**: Should probably go in `examples/` or `scripts/examples/`
2. **User guide location**: Could be `docs/SAMPLING_GUIDE.md` or add to existing guides
3. **Consider creating a notebook**: Jupyter notebook showing parameter effects visually
4. **Help text improvement**: Could add example usage directly in argparse help
5. **Migration guide**: Should show common workflow updates

### Key Success Criteria for Phase 5

From implementation.md:
- Users can understand when and how to use n_subsample
- Documentation clearly explains parameter interaction
- Examples demonstrate practical use cases
- All CLI help messages are clear
- Migration path from old to new syntax is documented

### Additional Context

The feature is fully functional and tested. Phase 5 is purely about documentation and user experience. No code changes are expected unless help text needs clarification.

The main value proposition of the feature:
1. **Memory management**: Load less data while still training on diverse samples
2. **Flexible experimentation**: Test different sampling strategies without reloading
3. **Reproducibility**: Seed-based sampling ensures consistent results
4. **Backward compatibility**: All existing workflows continue unchanged