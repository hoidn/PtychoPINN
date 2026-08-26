# Sampling Examples

This directory contains example scripts demonstrating PtychoPINN's centered-nearest
grouping (exact unique centers, no oversampling).

## Available Examples

### 1. `dense_grouping_example.sh` - Maximum Data Utilization
Shows how to use all of your candidate pool efficiently:
- Every candidate row becomes a group center: `training_groups = train_raw_selection`
- Maximizes data usage within memory constraints

### 2. `sparse_grouping_example.sh` - Fast Training with Diversity
Demonstrates loading a diverse candidate pool but training on fewer groups:
- Large `train_raw_selection` for diversity
- Smaller `training_groups` (exact group count = unique centers) for faster training

### 3. `memory_constrained_example.sh` - Limited Memory Strategies
Shows techniques for memory-limited environments:
- Minimal data loading
- Small batch sizes
- Progressive training with different seeds

### 4. `migration_from_legacy.sh` - Converting Old Scripts
Guide for updating scripts to the current parameters:
- Shows old vs new syntax
- Explains parameter mapping
- Demonstrates backward compatibility

## Quick Start

```bash
./dense_grouping_example.sh
```

This shows centered-nearest grouping on 2000 candidate rows: creating 2000 groups
(every row a center) versus 500 groups (a subset of centers).

## Understanding the Parameters

### Core Parameters
- **`--training_groups`**: Exact number of groups to create (= number of unique centers; replaces deprecated `--n_images` / `--n_groups`)
- **`--train_raw_selection`**: Number of rows loaded as the candidate pool (replaces deprecated `--n_subsample`)
- **`--neighbor_count`**: K, the nearest non-center candidate pool per group (must be ≥ `gridsize² - 1`)
- **`--gridsize`**: Square group side; each group has `gridsize²` rows: its designated center plus `gridsize² - 1` rows chosen without replacement from that K-candidate pool
- **`--subsample_seed`**: Reproducible random selection

### Centered-Nearest Grouping (No Oversampling)
- `training_groups` can never exceed the candidate pool size — more groups require a larger pool, not oversampling
- Every group's first member (column zero) is its designated center, on both the RAM and mmap rails
- Groups never cross an `object_index` boundary
- `group_padding_step` sizes only the Torch canvas — it does not change grouping membership

## Candidate Pool versus Group Count

The pool bounds how many distinct centers can be drawn; the group count selects
how many pool rows become centers:

```bash
# 2000 candidate rows -> up to 2000 groups
ptycho_train --train_raw_selection 2000 --training_groups 500 --gridsize 2 ...
```

Requesting `--training_groups 2500` with a 2000-row pool fails with a clear error.

## Tips

1. **Start with `neighbor_count = gridsize²`** for a good balance of locality and diversity
2. **Monitor memory usage** — the candidate pool size (`train_raw_selection`) bounds loaded data
3. **Use seeds** (`--subsample_seed`) for reproducible experiments
4. **Dense vs sparse** — set `training_groups` equal to the pool size for dense coverage, or smaller for faster iteration

## Backward Compatibility

Old scripts using `--n_images` / `--n_groups` (→ `training_groups`) and
`--n_subsample` (→ `train_raw_selection`) still parse as deprecated aliases:

```bash
# Old (deprecated but works)
ptycho_train --n_images 500

# New (recommended)
ptycho_train --training_groups 500
```

The retired K-choose-C oversampling flags (`--enable_oversampling`,
`--neighbor_pool_size`) are removed; migration diagnostics name the retired fields.
