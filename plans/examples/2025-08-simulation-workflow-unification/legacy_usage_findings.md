# RawData.from_simulation Usage Findings

**Date:** 2025-08-03  
**Search conducted:** `grep -r "from_simulation"`

## Active Code Usage

### 1. ptycho/nongrid_simulation.py (line 176)
- **Function:** `_generate_simulated_data_legacy_params()`
- **Context:** Used by `generate_simulated_data()` function
- **Status:** Active usage - this is the main function that scripts/simulation/simulate_and_save.py was using before refactoring
- **Note:** Already marked as "legacy" function with TODO for refactoring

### 2. scripts/simulation/simulate_full_frame.py (lines 293, 304)
- **Context:** Alternative simulation script for full-frame data generation
- **Status:** Active usage - uses from_simulation in two places
- **Note:** Has a workaround for gridsize>1 by temporarily setting gridsize=1
- **Action Required:** This script should also be updated to use the new modular approach

## Log/Error References

### 3. Various simulation.log files
Found in probe study directories showing the gridsize>1 crash:
- probe_study_phase3_test_QUICK_TEST/gs2_default/simulation.log
- probe_study_phase3_test_QUICK_TEST/gs2_hybrid/simulation.log
- probe_study_phase3_fix_test_QUICK_TEST/gs2_default/simulation.log
- probe_study_phase3_fix_test_QUICK_TEST/gs2_hybrid/simulation.log
- test_2x2_study_with_lines_QUICK_TEST/gs2_idealized/simulation.log
- probe_study_FULL_gs2/gs2_default/simulation.log

These logs show the ValueError crash when using from_simulation with gridsize>1.

## Documentation References

### 4. Planning documents
- plans/active/simulation-workflow-unification/* - Multiple references in initiative documents
- docs/PROJECT_STATUS.md - Mentions the from_simulation issue

## Summary

**Active code dependencies:**
1. `ptycho/nongrid_simulation.py` - Primary usage point
2. `scripts/simulation/simulate_full_frame.py` - Secondary script that needs updating

**Recommendation:** 
- The deprecation warning has been added to `RawData.from_simulation()`
- `nongrid_simulation.py` could be updated in a future initiative to use the modular approach
- `simulate_full_frame.py` should be documented as having the same gridsize>1 limitation