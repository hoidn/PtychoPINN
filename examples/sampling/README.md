# Sampling Examples

This directory contains example scripts demonstrating PtychoPINN's sampling capabilities, including the new automatic K choose C oversampling feature.

## Available Examples

### 1. `oversampling_comparison.sh` - K Choose C Oversampling
Demonstrates the power of automatic oversampling when requesting more groups than available points. Shows how to:
- Create standard 1:1 mapping (traditional approach)
- Use K=7 to enable 2x oversampling 
- Push to 4x oversampling for maximum augmentation

**Key Feature**: Automatic oversampling triggers when `n_groups > n_points` and `gridsize > 1`

### 2. `dense_grouping_example.sh` - Maximum Data Utilization
Shows how to use all of your subsampled data efficiently:
- Dense packing: `n_groups = n_subsample / gridsize²`
- Maximizes data usage within memory constraints

### 3. `sparse_grouping_example.sh` - Fast Training with Diversity
Demonstrates loading diverse data but training on fewer groups:
- Large `n_subsample` for diversity
- Smaller `n_groups` for faster training

### 4. `memory_constrained_example.sh` - Limited Memory Strategies
Shows techniques for memory-limited environments:
- Minimal data loading
- Small batch sizes
- Progressive training with different seeds

### 5. `migration_from_legacy.sh` - Converting Old Scripts
Guide for updating scripts to use the new parameters:
- Shows old vs new syntax
- Explains parameter mapping
- Demonstrates backward compatibility

## Quick Start

Run the oversampling comparison to see the new feature in action:

```bash
./oversampling_comparison.sh
```

This will train three models using the same 512 images but creating different numbers of groups:
- Traditional: 128 groups (K=4)
- Oversampled 2x: 256 groups (K=7)
- Oversampled 4x: 512 groups (K=7)

## Understanding the Parameters

### Core Parameters
- **`--n_groups`**: Number of groups to generate (replaces deprecated `--n_images`)
- **`--n_subsample`**: Number of images to load from dataset
- **`--neighbor_count`**: K value for nearest neighbors (use 7+ for oversampling)
- **`--gridsize`**: Size of neighbor groups (gridsize² images per group)

### Automatic Oversampling
When `n_groups > available_points` and `gridsize > 1`:
1. System automatically detects oversampling is needed
2. Switches to K choose C combination generation
3. Creates requested groups using combinations from K neighbors

### K Value Guidelines
- **K=4**: Traditional, no oversampling possible
- **K=5-6**: Modest oversampling (up to 2x)
- **K=7-8**: Good oversampling (up to 4x)
- **K>8**: Maximum oversampling (limited by C(K,C) combinations)

## Memory vs Augmentation Trade-off

The key insight: You can now trade compute for data augmentation:
- Same memory usage (controlled by `n_subsample`)
- More training groups (controlled by `n_groups` and `K`)
- Better model generalization through combinatorial augmentation

## Tips

1. **Start with K=7** for good balance of diversity and augmentation
2. **Monitor memory usage** - more groups means more compute, not more memory
3. **Use seeds** (`--subsample_seed`) for reproducible experiments
4. **Gradual increase** - start with 2x oversampling before pushing to 4x

## Backward Compatibility

Old scripts using `--n_images` still work but will show a deprecation warning. Update to `--n_groups` for clarity:

```bash
# Old (deprecated but works)
ptycho_train --n_images 500

# New (recommended)
ptycho_train --n_groups 500
```