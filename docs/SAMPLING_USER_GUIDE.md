# Sampling User Guide

## Overview

PtychoPINN provides flexible sampling control for managing memory usage and training efficiency. This guide explains how to use the sampling parameters effectively.

> **Note**: This guide documents the current sampling system (Phase 5). An advanced K choose C oversampling feature is planned for Phase 6 - see [phase6_plan.md](initiatives/independent_sampling_control/phase6_plan.md).

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

## Current Limitations

> **Important**: The current system has a fundamental limitation that will be addressed in Phase 6.

### One Group Per Seed Point

Currently, the system can only create **one group per seed point**. This means:
- Maximum groups = number of seed points available
- Cannot create overlapping groups from same data
- Cannot exploit K choose C combinations for augmentation

#### Example of Current Limitation

```bash
# This WILL NOT create 2000 groups as expected:
ptycho_train --train_data_file data.npz \
    --n_subsample 500 \   # Only 500 seed points available
    --n_groups 2000 \     # Wants 2000 groups
    --gridsize 2
# Result: Only creates 500 groups (capped at seed points)
```

### Planned Enhancement (Phase 6)

The upcoming K choose C oversampling feature will enable:
- Multiple groups per seed point
- Combinatorial augmentation (e.g., K=7, C=4 → 35 combinations per seed)
- True oversampling where `n_images >> n_subsample`

```bash
# Future capability (Phase 6):
ptycho_train --train_data_file data.npz \
    --n_subsample 500 \
    --n_groups 2000 \
    --gridsize 2 \
    --enable_oversampling true \  # Coming in Phase 6
    --neighbor_pool_size 7         # K=7 for K choose C
# Will create 2000 groups from 500 images via combinations
```

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