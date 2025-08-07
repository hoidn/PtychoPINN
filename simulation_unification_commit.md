# feat: Complete simulation workflow unification with gridsize > 1 fix

## Summary
Fix gridsize > 1 crashes in simulation pipeline by refactoring to modular workflow and add comprehensive documentation improvements across the codebase.

## Key Changes

### Simulation Pipeline Fix (Core Initiative)
- **Refactored `scripts/simulation/simulate_and_save.py`**: Replaced monolithic `RawData.from_simulation` with explicit modular workflow
- **Added deprecation warning** to `ptycho/raw_data.py:from_simulation()` method
- **Updated documentation**: Added migration guides in simulation README and CLAUDE.md
- **Fixed tensor shape handling**: Proper Channel↔Flat format conversion for gridsize > 1
- **Added comprehensive test suite**: `tests/simulation/test_simulate_and_save.py`

### Documentation Enhancements
- **Added module-level docstrings** to 30+ core library modules with architectural context
- **Updated `DEVELOPER_GUIDE.md`**: Added section 3.4 on tensor formats for gridsize > 1
- **Updated `PROJECT_STATUS.md`**: Marked initiative as complete, moved to completed section
- **Updated `TOOL_SELECTION_GUIDE.md`**: Added gridsize > 1 support examples

### Code Quality Improvements
- **Removed broken tests**: Deleted `tests/test_simulate_and_save*.py` with incorrect API usage
- **Added `.gitignore` entries**: phase_diff.txt, tmp*
- **Enhanced documentation style**: More detailed architectural roles and workflow examples

## Technical Details

The core fix addresses the ValueError crash when using gridsize > 1 by:
1. Properly handling coordinate expansion for neighbor groups
2. Explicit Channel Format (B,N,N,C) to Flat Format (B*C,N,N,1) conversion
3. Correct tensor shape management throughout the pipeline

## Verification
```bash
# Core bug fix test passes:
python -m unittest tests.simulation.test_simulate_and_save.TestSimulateAndSave.test_gridsize2_no_crash -v
# Result: ok (20.287s)
```

## Migration Guide
Users of the deprecated `RawData.from_simulation` should use:
```bash
python scripts/simulation/simulate_and_save.py \
    --input-file input.npz \
    --output-file output.npz \
    --gridsize 2  # Now works correctly!
```

## Breaking Changes
None - deprecation warning added but legacy method still functional for gridsize=1

## Related Documentation
- Implementation summary: `plans/active/simulation-workflow-unification/implementation_summary.md`
- R&D plan: `plans/active/simulation-workflow-unification/plan.md`
- Phase checklists: Complete