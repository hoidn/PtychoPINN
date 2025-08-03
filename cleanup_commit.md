# chore: Remove broken and redundant test files

## Summary
Remove outdated/broken test files that have been superseded by comprehensive test suite.

## Changes
- Remove `tests/test_simulate_and_save.py` - Uses incorrect parameter names (`nimages` instead of `n_images`)
- Remove `tests/test_simulate_and_save_simple.py` - Redundant with comprehensive test suite

## Rationale
These test files were using outdated API calls and failing with `TypeError: simulate_and_save() got an unexpected keyword argument 'nimages'`. The comprehensive test suite at `tests/simulation/test_simulate_and_save.py` provides full coverage including:
- Gridsize > 1 validation (core bug fix test)
- Data contract compliance
- Performance benchmarking
- Integration testing

The removed files were creating confusion and technical debt. The single, well-maintained test suite is sufficient for validating the simulation workflow.

## Verification
```bash
# Confirm the good test still works
python -m unittest tests.simulation.test_simulate_and_save.TestSimulateAndSave.test_gridsize2_no_crash -v
# Result: ok
```

## Files removed:
- tests/test_simulate_and_save.py
- tests/test_simulate_and_save_simple.py

## Files kept:
- tests/simulation/test_simulate_and_save.py (comprehensive test suite)