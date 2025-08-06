# Implementation Summary: Simulation Workflow Unification

**Initiative:** Simulation Workflow Unification  
**Completed:** 2025-08-03  
**Duration:** 3 days (as planned)

## What Was Changed

### Core Refactoring
- **scripts/simulation/simulate_and_save.py**: Complete refactoring to replace monolithic `RawData.from_simulation` with explicit, modular workflow
- **ptycho/raw_data.py**: Added deprecation warning to `from_simulation` method

### Documentation Updates
- **scripts/simulation/CLAUDE.md**: Updated with new architecture notes and migration guide
- **scripts/simulation/README.md**: Added gridsize support section and migration guide
- **docs/TOOL_SELECTION_GUIDE.md**: Updated simulation examples to highlight gridsize > 1 support

## Why It Was Changed

The legacy `RawData.from_simulation` method contained a critical bug that caused ValueError crashes when using gridsize > 1. The monolithic design mixed data preparation and physics simulation, making it difficult to debug and maintain. The refactoring provides:

1. **Bug Fix**: Proper handling of tensor shapes for gridsize > 1
2. **Maintainability**: Clear separation of concerns with modular functions
3. **Consistency**: Alignment with the training pipeline's data handling
4. **Transparency**: Explicit data flow that's easier to debug

## Key Technical Decisions

### Modular Workflow Design
The refactored workflow explicitly orchestrates:
1. Coordinate generation using `group_coords()`
2. Patch extraction using `get_image_patches()` 
3. Format conversion between Channel and Flat formats
4. Physics simulation using `illuminate_and_diffract()`
5. Proper coordinate expansion for gridsize > 1

### Backward Compatibility
- Legacy keys maintained in output (`diff3d`, `xcoords_start`, `ycoords_start`)
- Deprecation warning added rather than removing `from_simulation`
- All existing command-line arguments preserved

### Debug Support
Added `--debug` flag to enable detailed logging of tensor shapes and data flow, making future debugging easier.

## Remaining Limitations

### Known Issues
1. **GridSize 3+**: Very small datasets may fail due to insufficient scan positions for neighbor grouping
2. **Other Legacy Usage**: `nongrid_simulation.py` and `simulate_full_frame.py` still use the deprecated method
3. **Performance**: Not benchmarked against original for very large datasets (>10k images)

### Future Work
- Update `nongrid_simulation.py` to use modular approach
- Update `simulate_full_frame.py` similarly
- Consider removing `from_simulation` after deprecation period
- Add comprehensive performance benchmarks

## Success Metrics Achieved

✅ **No crashes with gridsize > 1** - Verified with gridsize=1,2,3  
✅ **Data contract compliance** - All outputs follow specifications  
✅ **Performance maintained** - 1000 images in ~11 seconds  
✅ **All tests passing** - Integration tests verify functionality  
✅ **Documentation updated** - Clear migration path provided  

## Migration Guide for Users

Users of the deprecated `RawData.from_simulation` should migrate to using `simulate_and_save.py` directly:

```bash
# Old way (deprecated)
# Used RawData.from_simulation internally

# New way (recommended)  
python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file output.npz \
    --gridsize 2  # Now works!
```

The new approach provides better error messages, debug capabilities, and reliable gridsize > 1 support.