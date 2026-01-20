# GridSize and n_groups Interaction Guide

## Critical Concept: Unified n_groups Parameter

The `n_groups` parameter provides **consistent behavior** regardless of the `gridsize` value. Unlike the deprecated `n_images` parameter, `n_groups` always refers to the number of groups, eliminating previous confusion. This guide explains the unified behavior.

## The Two Modes

### GridSize = 1 (Single Image Groups)
- **Meaning of n_groups**: Number of groups, each containing 1 image
- **Example**: `--n_groups 512` → Creates 512 groups of 1 image each = 512 total patterns
- **Total patterns used**: `n_groups × 1 = n_groups`
- **Common use case**: Standard ptychography reconstruction

### GridSize > 1 (Multi-Image Groups)
- **Meaning of n_groups**: Number of groups, each containing gridsize² neighboring patterns
- **Example**: `--n_groups 512` with `--gridsize 2` → Creates 512 groups of 4 patterns each = 2048 total patterns
- **Total patterns needed**: `n_groups × gridsize²`
- **Data selection**: Finds spatially adjacent patterns to form groups
- **Common use case**: Multi-shot reconstruction, improved noise handling

### Key Advantage
With `n_groups`, the parameter **always means the same thing**: number of groups to create. The confusion about whether it means "individual images" vs "groups" is eliminated.

## Real Example: The Run1084 Bug

### What Happened (Old n_images Behavior)
```yaml
# Config file specified:
gridsize: 1
n_images: 512  # Old parameter - meaning changed based on gridsize!
```

```bash
# Command requested:
--n_train_images 512  # Old parameter with confusing interpretation
```

But due to configuration precedence issues, the system used `gridsize=2` from legacy defaults.

### The Result
- **Expected**: 512 individual patterns from Run1084 dataset (1,087 available)
- **Actual**: System tried to create 512 groups of 4 patterns = 2,048 patterns needed
- **Outcome**: Only ~64 valid groups could be formed from 1,087 patterns
- **Final dataset**: 64 groups × 4 = 256 patterns used (not 512!)

### How n_groups Fixes This
With the new `n_groups` parameter:
```yaml
gridsize: 1
n_groups: 512  # Always means 512 groups, regardless of gridsize
```
- **gridsize=1**: 512 groups × 1 image = 512 patterns (as expected)
- **gridsize=2**: 512 groups × 4 images = 2048 patterns (clearly specified)

## Configuration Precedence (Highest to Lowest)

1. **Command-line arguments**: `--gridsize 1` (highest priority)
2. **YAML config file**: `gridsize: 1` 
3. **Legacy params.cfg**: Default gridsize=2 (lowest priority)

⚠️ **Warning**: If command-line argument is not provided, legacy defaults may override YAML!

## How to Avoid This Issue

### Use n_groups for Clarity
```bash
# Best - modern n_groups parameter
./scripts/run_comparison.sh data.npz test.npz output --n-train-groups 512 --gridsize 1

# Acceptable - deprecated n_images with explicit gridsize
./scripts/run_comparison.sh data.npz test.npz output --n-train-images 512 --gridsize 1

# Problematic - relies on config precedence
./scripts/run_comparison.sh data.npz test.npz output --n-train-images 512
```

### Check Your Logs
Look for these log messages to verify correct interpretation:

**Modern n_groups behavior (gridsize=1)**:
```
INFO - Using n_groups=512 with gridsize=1: creating 512 groups of 1 image each
📊 Total patterns: 512
```

**Modern n_groups behavior (gridsize>1)**:
```
INFO - Using n_groups=512 with gridsize=2: creating 512 groups of 4 images each
📊 Total patterns: 2048
Using grouping-aware subsampling for gridsize=2
```

**Legacy n_images deprecation warning**:
```
WARNING - Parameter n_images is deprecated. Please use n_groups instead.
```

## Quick Reference Table

| gridsize | n_groups | Actual Patterns Used | Minimum Dataset Size |
|----------|----------|---------------------|---------------------|
| 1        | 512      | 512                 | 512                 |
| 2        | 512      | 2,048               | 2,048               |
| 3        | 512      | 4,608               | 4,608               |
| 4        | 512      | 8,192               | 8,192               |

**Key Difference**: With `n_groups`, the calculation is always `n_groups × gridsize²`. No ambiguity.

## Common Scenarios

### Scenario 1: Quick Test Run
```bash
# Use 100 groups for quick test
--gridsize 1 --n_groups 100  # 100 groups × 1 = 100 patterns
```

### Scenario 2: Full Dataset Training
```bash
# Omit n_groups to use all available data
--gridsize 1  # No --n_groups parameter
```

### Scenario 3: Grouped Training
```bash
# Create 100 groups of 4 patterns (400 patterns total)
--gridsize 2 --n_groups 100  # 100 groups × 4 = 400 patterns
```

## Debugging Checklist

If you're getting unexpected dataset sizes:

1. ✅ **Migrate to n_groups**: Use `--n_groups` instead of deprecated `--n_images`
2. ✅ Check the log output for gridsize value being used
3. ✅ Verify YAML config has correct gridsize
4. ✅ Add explicit `--gridsize` to command line
5. ✅ Check if dataset has enough patterns for requested groups
6. ✅ Look for deprecation warnings about n_images usage

## Inter‑Group Overlap Control (Updated)

Grouping guarantees tight intra‑group neighborhoods (C = gridsize² from each seed’s K‑NN). For this study, inter‑group behavior is controlled explicitly by sampling parameters and measured via overlap metrics, not by spacing gates:

- Control: use image subsampling fraction (`s_img`) and number of groups (`n_groups`).
- Measure: compute 2D disc‑overlap metrics as defined in `specs/overlap_metrics.md` (Metric 2 and Metric 3); for `gridsize=2`, Metric 1 (group‑based) is also defined.

Note: Phase D now writes filtered NPZs per `s_img` and `rng_seed_subsample` and records per‑split metrics JSON and a bundle; see `specs/overlap_metrics.md` for the normative API and outputs. This ensures Phase E training consumes exactly the subsampled datasets that the metrics were computed on (no additional training‑side sampling required).
- Behavior: no geometry/packing acceptance gates; the pipeline proceeds and records the measured overlaps.

Dense/sparse labels are deprecated for this study; prefer explicit `s_img`/`n_groups` and report the measured overlaps per the spec.

Tips:
- Use independent subsampling to bound memory, then request `n_groups` according to desired throughput; oversampling (duplicated membership across groups) remains allowed for `gridsize>1` as in current grouping.
- Log the three overlap metrics (where applicable) to validate sampling settings.

## Related Documentation

- [Configuration Guide](CONFIGURATION_GUIDE.md) - Full configuration parameter reference
- [Data Contracts](data_contracts.md) - NPZ dataset format requirements
- [Tool Selection Guide](TOOL_SELECTION_GUIDE.md) - Choosing the right scripts

## Summary

**Remember**: `n_groups` always means "number of groups" regardless of gridsize. For this study, use explicit `s_img` and `n_groups` to control sampling and report measured overlaps as defined in `specs/overlap_metrics.md` (and written into Phase D NPZ `_metadata`). Avoid legacy dense/sparse labels and spacing‑based gates.

**Migration**: Replace `n_images` with `n_groups` and adopt the overlap metrics. Add `s_img` and `n_groups` explicitly in commands/config for clarity.
