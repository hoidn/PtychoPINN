# Sampling User Guide

## Overview

PtychoPINN provides flexible sampling control for managing memory usage and training efficiency. This guide explains how to use the sampling parameters effectively.

> Note: K‑choose‑C oversampling is implemented in the loader for `gridsize>1` when `n_groups > n_subsample` and `neighbor_count≥C`. Phase 6 focuses on hardening, examples, and documentation — see [phase6_plan.md](initiatives/independent_sampling_control/phase6_plan.md).

## Quick Start

### Basic Usage

```bash
# Simple training with default sampling
ptycho_train --train_data_file data.npz --output_dir my_run

# Explicit sampling control
ptycho_train --train_data_file data.npz \
    --n_subsample 2000 \  # Load 2000 images from dataset
    --n_groups 500 \      # Use 500 groups for training
    --output_dir my_run
```

## Understanding Sampling Parameters

### The Two Key Parameters

1. **`n_subsample`**: How many images to load from the dataset file
2. **`n_groups`**: How many groups to actually use for training (replaces deprecated `n_images`)

### Parameter Interaction

The system operates in three modes:

#### 1. Legacy Mode (Backward Compatible)
When only the deprecated `n_images` is specified:
- For `gridsize=1`: `n_images` = number of groups with 1 image each
- For `gridsize>1`: `n_images` = number of groups with gridsize² images each

**Note**: New code should use `n_groups` instead of `n_images`.

```bash
# Modern: Load and use 1000 groups
ptycho_train --train_data_file data.npz --n_groups 1000

# Modern with grouping: Create 100 groups (uses 400 images for gridsize=2)
ptycho_train --train_data_file data.npz --n_groups 100 --gridsize 2

# Legacy (deprecated but still works): 
ptycho_train --train_data_file data.npz --n_images 1000  # Shows deprecation warning
```

#### 2. Independent Control Mode
When `n_subsample` is specified:
- `n_subsample` controls data loading
- `n_groups` controls training usage
- Allows loading more data than you train on

```bash
# Load 5000 images, but only train on 1000 groups
ptycho_train --train_data_file data.npz \
    --n_subsample 5000 \
    --n_groups 1000
```

#### 3. Default Mode
When neither parameter is specified:
- Training: Uses default `n_groups=512`
- Inference: Uses full dataset

## Common Use Cases

### Memory-Constrained Training

When GPU/CPU memory is limited:

```bash
# Minimal memory usage
ptycho_train --train_data_file data.npz \
    --n_subsample 256 \
    --n_groups 256 \
    --batch_size 8 \
    --output_dir memory_limited_run
```

### Diverse Sampling

Load more data for diversity, train on subset for speed:

```bash
# Load 5000 diverse images, train on 500 groups
ptycho_train --train_data_file data.npz \
    --n_subsample 5000 \
    --n_groups 500 \
    --subsample_seed 42 \  # For reproducibility
    --output_dir diverse_sampling_run
```

### Dense Grouping (gridsize > 1)

Maximize data utilization with neighbor grouping:

```bash
# Load 2048 images, create 512 groups of 4 (gridsize=2)
ptycho_train --train_data_file data.npz \
    --n_subsample 2048 \
    --n_groups 512 \
    --gridsize 2 \
    --output_dir dense_grouping_run
```

## Oversampling (K choose C)

**NEW in Phase 6**: Oversampling now requires **explicit opt-in** and validation to prevent accidental misuse.

The loader supports K‑choose‑C oversampling for creating more training groups than available seed points by sampling multiple combinations from each seed's K nearest neighbors.

### Prerequisites (OVERSAMPLING-001)
1. **`gridsize > 1`** so that `C = gridsize² > 1`
2. **`enable_oversampling=True`** - Explicit opt-in flag (required as of Phase 6)
3. **`neighbor_pool_size ≥ C`** - Pool size must be at least C (default: uses `neighbor_count` value)

### How to Enable Oversampling

**Basic Example:**
```bash
ptycho_train --train_data_file data.npz \
    --n_subsample 500 \           # 500 seed points
    --n_groups 2000 \             # ask for 2000 groups
    --gridsize 2 \                # C = 4
    --neighbor_count 7 \          # K value for grouping
    --enable_oversampling \       # NEW: Explicit opt-in
    --neighbor_pool_size 7 \      # NEW: Pool size (must be >= C=4)
    --output_dir oversampled_run
```

**Why the guards?** Oversampling reuses local neighborhoods and can lead to overfitting if used incorrectly. The explicit flags ensure you're aware of this behavior.

### Error Messages

If you request `n_groups > n_subsample` without proper flags, you'll see:
```
ValueError: Requesting 2000 groups but only 500 points available (gridsize=2, C=4).
K choose C oversampling is required but not enabled.
Set enable_oversampling=True and ensure neighbor_pool_size >= 4 to proceed.
See OVERSAMPLING-001 in docs/findings.md for details.
```

If `neighbor_pool_size < C`:
```
ValueError: K choose C oversampling requires neighbor_pool_size >= C (gridsize²).
Got neighbor_pool_size=3, but C=4.
Increase neighbor_pool_size to at least 4.
See OVERSAMPLING-001 in docs/findings.md for details.
```

### Debug Logging

The loader emits `[OVERSAMPLING DEBUG]` log lines showing:
- Whether oversampling is enabled/disabled
- Effective pool size K
- Which branch (oversampling vs. standard) was taken

### Notes & Tips
- **Larger K** increases the combination pool; keep K reasonable for memory
- **Overfitting risk**: Oversampling reuses local neighborhoods; monitor using held‑out regions (e.g., spatial half splits)
- **Memory friendly**: Works with independent sampling - keep `n_subsample` small while training on many groups
- **Default behavior**: If `neighbor_pool_size` is `None`, it defaults to the `neighbor_count` value

## Reproducibility

Always use `--subsample_seed` for reproducible sampling:

```bash
# Reproducible sampling
ptycho_train --train_data_file data.npz \
    --n_subsample 1000 \
    --n_groups 500 \
    --subsample_seed 42 \  # Same seed = same data selection
    --output_dir reproducible_run
```

## Migration from Old Scripts

### Old Script (Deprecated)
```bash
ptycho_train --train_data_file data.npz --n_images 1000  # Will show deprecation warning
```

### New Equivalent (Recommended)
```bash
ptycho_train --train_data_file data.npz \
    --n_subsample 1000 \
    --n_groups 1000
```

### Modern with Memory Control
```bash
ptycho_train --train_data_file data.npz \
    --n_subsample 5000 \  # Load more for diversity
    --n_groups 1000       # Train on subset
```

## Best Practices

1. **Always specify `--subsample_seed`** for reproducible experiments
2. **Use `--n_subsample`** when you need explicit memory control
3. **Check log messages** for parameter interpretation
4. **For gridsize > 1**, ensure `n_subsample ≥ n_groups × gridsize²`
5. **Use `n_groups` instead of deprecated `n_images`** in new scripts

## Troubleshooting

### Warning: "n_subsample may be too small"

This appears when `n_subsample < n_groups × gridsize²`:

```
WARNING: n_subsample (100) may be too small to create 50 groups of size 4. 
Consider increasing n_subsample to at least 200
```

**Solution**: Increase `n_subsample` or decrease `n_groups`.

### Deprecation Warning: "n_images is deprecated"

```
WARNING: Parameter n_images is deprecated. Please use n_groups instead for consistent behavior.
```

**Solution**: Replace `--n_images` with `--n_groups` in your commands.

### No Progress in Training

If training seems stuck:
1. Check if `n_groups` is too small
2. Verify data is loading correctly in logs
3. Try increasing `n_subsample` for more diversity

## Examples

See the `examples/sampling/` directory for complete examples:
- `dense_grouping_example.sh` - Maximize data utilization with n_groups
- `sparse_grouping_example.sh` - Fast training with subset using n_groups
- `memory_constrained_example.sh` - Memory strategies with n_groups
- `migration_from_legacy.sh` - Converting from n_images to n_groups

## Further Reading

- [Configuration Guide](CONFIGURATION.md) - Detailed parameter documentation
- [Commands Reference](COMMANDS_REFERENCE.md) - Complete CLI reference
- [Phase 6 Plan](initiatives/independent_sampling_control/phase6_plan.md) - Upcoming K choose C oversampling
