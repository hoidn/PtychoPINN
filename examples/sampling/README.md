# Sampling Examples

This directory contains runnable YAML-backed examples for the unified
`ptycho_train` sampling configuration, including explicit K-choose-C
oversampling. Run the scripts from any directory; each training script
resolves its adjacent YAML file. The migration script is print-only.

Numeric, Boolean, model, and sampling values live in YAML because the current
generated CLI does not decode those values before strict Pydantic validation.
The scripts use CLI overrides only for paths and literal strings.

## Available Examples

### 1. `oversampling_comparison.sh` - K Choose C Oversampling
Demonstrates the exact oversampling boundary with a 512-row selected pool:
- Standard grouping: 512 groups, one grouping anchor per selected row
- K=7 oversampling: 1024 groups
- K=7 oversampling: 2048 groups

Oversampling is entered only when `training_groups` is greater than the selected
raw-row count. It also requires `sampling.enable_oversampling=true`, grid size
greater than one, and a candidate pool at least as large as the channel count.

### 2. `dense_grouping_example.sh` - Maximum Data Utilization
Uses every selected row once as a grouping anchor. Neighbor membership can
overlap between groups, so this is anchor coverage rather than disjoint
packing.

### 3. `sparse_grouping_example.sh` - Fast Training with Diversity
Selects a large raw-row pool but materializes groups for only a small subset of
grouping anchors. Neighbor membership can still overlap.

### 4. `memory_constrained_example.sh` - Limited Memory Strategies
Uses bounded raw-row selection, group count, and batch size. Raw storage grows
with `train_raw_selection`; grouped arrays grow approximately with
`training_groups * gridsize²`.

### 5. `migration_from_legacy.sh` - Converting Old Scripts
Prints the flat-to-nested configuration mapping and a runnable YAML shape.

## Quick Start

From the repository root, run:

```bash
./examples/sampling/oversampling_comparison.sh
```

This trains three models from the same 512-row selected pool:
- Standard: 512 groups (K=4)
- Oversampled 2x: 1024 groups (K=7)
- Oversampled 4x: 2048 groups (K=7)

## Understanding the Parameters

### Core Parameters
- **`sampling.training_groups`**: Number of groups to generate
- **`sampling.train_raw_selection`**: Number of raw rows to select
- **`sampling.neighbor_count`**: Neighbor-query count
- **`sampling.enable_oversampling`**: Explicit oversampling opt-in
- **`sampling.neighbor_pool_size`**: K value used for combinations
- **`model.gridsize`**: Neighbor-grid width; each group has `gridsize²` channels

### Explicit oversampling

When `sampling.training_groups > sampling.train_raw_selection` and `gridsize > 1`, explicit
oversampling constructs multiple `C`-member combinations from K-neighbor
candidate pools. The per-anchor combination capacity is `binomial(K, C)`,
where `C = gridsize²`. Invalid pool sizes and requests without explicit opt-in
fail instead of silently changing the policy.

## Memory vs Augmentation Trade-off

`train_raw_selection` bounds the selected raw pool. `training_groups` controls
both training compute and the materialized grouped arrays, whose leading payload
is roughly `training_groups * gridsize²`. Oversampling can increase diversity from a fixed raw
pool, but it does not have constant memory cost as group count grows.

## Tips

1. More groups increase both training compute and grouped-array memory;
   `train_raw_selection` separately controls the raw selected pool.
2. Set `sampling.subsample_seed` in YAML for reproducible selection.
3. Increase group count gradually and measure the quality/compute trade-off.

## Backward Compatibility

The nested YAML alias `sampling.n_images` remains loadable but emits a
deprecation warning. Update it to `sampling.training_groups`:

```yaml
# Old
sampling:
  n_images: 500

# New
sampling:
  training_groups: 500
```
