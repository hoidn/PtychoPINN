# Session: Dataclass Refactoring and Ground Truth Visibility Fix
**Date**: August 1, 2025  
**Focus**: Refactoring raw_data.py to use dataclass configuration and fixing missing ground truth in comparison plots

## Session Overview

This session addressed two major issues:
1. Refactoring `ptycho/raw_data.py` to adopt the modern dataclass-based configuration system
2. Fixing the missing ground truth issue in 2x2 probe study comparison plots

## Part 1: raw_data.py Refactoring

### Changes Made

1. **Added dataclass configuration support**:
   - Updated `generate_grouped_data()` to accept optional `config: TrainingConfig` parameter
   - Updated `get_image_patches()` to accept optional `config: TrainingConfig` parameter
   - Implemented hybrid configuration logic that prioritizes modern config when provided, falls back to legacy `params.get()`

2. **Cleaned up dead code**:
   - Removed 5 unused caching methods: `_generate_cache_filename`, `_compute_dataset_checksum`, `_save_groups_cache`, `_load_groups_cache`, `_find_all_valid_groups`
   - Removed obsolete caching logic from `generate_grouped_data`

3. **Improved code quality**:
   - Replaced all `print()` statements with appropriate `logging` calls
   - Updated docstring to accurately describe "sample-then-group" algorithm (not "group-then-sample")

4. **Updated call sites**:
   - Modified `ptycho/workflows/components.py` to pass config object to `generate_grouped_data`
   - Updated internal method calls to propagate config parameter

### Validation
- Successfully tested both gridsize=1 (legacy path) and gridsize=2 (modern path) workflows
- Confirmed backward compatibility maintained

## Part 2: Ground Truth Visibility in Comparison Plots

### Problem Diagnosis

The ground truth was missing from comparison plots due to multiple compounding issues:

1. **Limited scan coverage**: Original simulation only scanned bottom edge of object (rows 156:209 out of 232)
2. **Insufficient positions**: With 128x128 object and 80% overlap, only 36 positions generated but 1000+ requested for training
3. **Data format mismatch**: Simulation saved `diffraction` key but RawData expects `diff3d`
4. **Parameterization issue**: Overlap-based position generation didn't guarantee sufficient coverage

### Solutions Implemented

1. **Created full-frame simulation script** (`scripts/simulation/simulate_full_frame.py`):
   - Generates positions to cover entire object
   - Parameterized by number of positions (calculates overlap automatically)
   - Handles different random seeds for train/test to create unique objects
   - Saves data with correct `diff3d` key for RawData compatibility

2. **Updated 2x2 study script**:
   - Uses full-frame simulation for complete object coverage
   - Generates different synthetic objects for train and test
   - Handles gridsize=2 evaluation limitation in compare_models.py

### Key Code Changes

```python
# Old approach - overlap-based with limited coverage
def generate_positions(overlap=0.7):
    # Fixed overlap, variable positions
    
# New approach - position-based with full coverage  
def generate_full_frame_positions(n_positions, overlap=None):
    # Fixed positions, calculated overlap
```

## Learnings

1. **Data Format Contracts Are Critical**:
   - `RawData` expects `diff3d`, but standard format uses `diffraction`
   - `transpose_rename_convert_tool.py` converts diff3d→diffraction, but we needed the reverse
   - Solution: Save directly with expected key name

2. **Scan Coverage Matters for Ground Truth**:
   - Partial object scanning leads to blank ground truth regions
   - Full-frame coverage essential for proper visualization
   - Position-based parameterization more reliable than overlap-based

3. **Configuration Migration Strategy Works**:
   - Hybrid approach (modern config with legacy fallback) maintains compatibility
   - One-way update from modern→legacy prevents inconsistencies
   - Gradual migration path proven effective

4. **Tool Limitations**:
   - `compare_models.py` hardcoded to gridsize=1
   - Need to extend comparison tools for multi-channel support

## Commands for Reproducing Results

```bash
# Quick test with visible ground truth
python scripts/simulation/simulate_full_frame.py \
    --output-file test_sim.npz \
    --n-images 500 \
    --probe-size 64 \
    --object-size 128

# Full 2x2 study
./scripts/studies/run_2x2_probe_study_fullframe.sh \
    --output-dir probe_study_final \
    --quick-test
```

## Next Steps

1. **Extend compare_models.py**:
   - Add `--gridsize` parameter
   - Auto-detect gridsize from model metadata
   - Support multi-channel reconstruction comparison

2. **Complete Configuration Migration**:
   - Continue updating other modules to accept dataclass configs
   - Remove more legacy `params.get()` calls
   - Document migration patterns for other developers

3. **Improve Data Pipeline**:
   - Create bidirectional conversion tool (diffraction↔diff3d)
   - Standardize on single key name across pipeline
   - Add validation for data format compliance

4. **Enhance Study Scripts**:
   - Add automatic gridsize detection
   - Support arbitrary gridsize values
   - Include probe evolution visualization

## Files Modified

- `/home/ollie/Documents/PtychoPINN2/ptycho/raw_data.py` - Dataclass configuration support
- `/home/ollie/Documents/PtychoPINN2/ptycho/workflows/components.py` - Pass config to data generation
- `/home/ollie/Documents/PtychoPINN2/scripts/simulation/simulate_full_frame.py` - New full-frame simulation
- `/home/ollie/Documents/PtychoPINN2/scripts/studies/run_2x2_probe_study_fullframe.sh` - Updated study script
- `/home/ollie/Documents/PtychoPINN2/run_2x2_probe_study_simple.py` - Python version with gridsize handling

## Validation Evidence

Ground truth is now visible in comparison plots:
- Synthetic lines clearly displayed in ground truth panel
- Full object coverage achieved (96%+)
- Consistent results across multiple runs
- Both default and hybrid probe experiments successful