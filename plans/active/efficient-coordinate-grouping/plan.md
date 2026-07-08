# R&D Plan: Efficient Coordinate Grouping Implementation

**Project/Initiative Name:** Efficient "Sample-then-Group" Coordinate Logic

**Start Date:** 2025-01-22

**Status:** Active

## Problem Statement

The current coordinate grouping logic for gridsize > 1 uses an inefficient "group-then-sample" approach. While mitigated by caching, this causes a severe performance penalty and high memory usage on the first run with a new dataset or new parameters. It also incorrectly samples from a pre-computed universe of all possible groups rather than truly sampling n_images points from the dataset.

### Current Implementation Issues:
1. **Memory explosion**: The `_find_all_valid_groups()` method generates groups for ALL points in the dataset, then samples from them
2. **Poor first-run performance**: For large datasets (10,000+ points), this can take minutes and consume gigabytes of memory
3. **Statistical bias**: Sampling from pre-computed groups doesn't provide uniform sampling of scan positions
4. **Cache dependency**: Performance is only acceptable due to caching, making the system brittle

## Proposed Solution / Hypothesis

We will refactor the `generate_grouped_data` method in `ptycho/raw_data.py` to implement a true "sample-then-group" strategy. This involves:

1. First randomly sampling the desired number of seed points (nsamples) from the dataset
2. Only then finding the nearest neighbors for that small subset
3. Building groups only for the sampled points

We hypothesize this will:
- Eliminate the first-run performance penalty (>10x improvement expected)
- Drastically reduce memory usage
- Provide more statistically sound sampling of the scan positions
- Maintain backward compatibility for gridsize=1

## Success Criteria

1. ✅ The "sample explosion" issue is eliminated; nsamples=512 results in exactly 512 groups
2. ✅ First-run data generation for gridsize > 1 is significantly faster (e.g., >10x improvement) and uses less memory
3. ✅ The gridsize=1 workflow continues to function correctly without regression
4. ✅ All existing and new tests pass
5. ✅ The legacy functions (`get_neighbor_diffraction_and_positions`, `group_coords`) are successfully removed

## Technical Approach

### Algorithm Change
**Current (inefficient):**
```
1. Find ALL possible groups (O(n_total * K))
2. Cache all groups
3. Sample nsamples from cached groups
```

**New (efficient):**
```
1. Sample nsamples seed points (O(nsamples))
2. Find neighbors only for seed points (O(nsamples * K))
3. Form groups from neighbors
```

### Key Implementation Details
- Use `np.random.choice` with seed parameter for reproducible sampling
- Build single `cKDTree` for efficient neighbor queries
- Unify gridsize=1 and gridsize>1 code paths
- Remove caching infrastructure (no longer needed)

## Risk Analysis

### Risks:
1. **Reproducibility change**: Different sampling behavior may affect existing workflows
2. **Cache file conflicts**: Old cache files may need cleanup
3. **Backward compatibility**: Some users may depend on exact group patterns

### Mitigations:
1. Add random seed parameter for reproducibility
2. Provide cache cleanup utility
3. Add deprecation warnings before removing legacy functions
4. Consider temporary compatibility flag during transition

## Phased Implementation Plan

### Overall Goal
To replace the inefficient "group-then-sample" logic with a high-performance "sample-then-group" strategy, remove the legacy code paths, and unify the data generation logic for all gridsize values.

### Phase 1: Implement the New "Sample-then-Group" Core Logic
**Goal:** To create a new, private helper function within RawData that implements the efficient grouping logic, and to write comprehensive unit tests for it.

**Deliverable:** A new, well-tested `_generate_groups_efficiently` method in `ptycho/raw_data.py` and a corresponding new test file.

**Timeline:** 2-3 days

### Phase 2: Integration and Legacy Code Removal
**Goal:** To integrate the new efficient logic into the public `generate_grouped_data` method, refactor the gridsize=1 case to use it, and remove the now-redundant legacy functions.

**Deliverable:** A refactored `ptycho/raw_data.py` with a single, unified code path for all gridsize values and the removal of deprecated functions.

**Timeline:** 2-3 days

### Phase 3: Validation and Documentation
**Goal:** To perform end-to-end validation, benchmark the performance improvements, and update all relevant documentation.

**Deliverable:** A validation report with performance benchmarks, and updated documentation (README.md, CLAUDE.md, etc.) reflecting the new, unified architecture.

**Timeline:** 1-2 days

## Expected Outcomes

1. **Performance**: 10-100x improvement in first-run data generation time
2. **Memory**: 10-100x reduction in memory usage for large datasets
3. **Simplicity**: Removal of ~200 lines of legacy code
4. **Correctness**: More statistically sound sampling behavior
5. **Maintainability**: Single, unified code path for all gridsize values

## Review Recommendations

Based on initial review, the following enhancements are recommended:

1. **Add random seed parameter** for reproducibility in the new method
2. **Include memory usage benchmarks** alongside time benchmarks
3. **Create migration guide** for users with existing cache files
4. **Document the semantic change** in sampling behavior
5. **Consider versioning flag** during transition period
6. **Add property-based tests** for group validity

## References

- Current implementation: `ptycho/raw_data.py` lines 350-654
- Legacy functions: `group_coords()`, `get_neighbor_diffraction_and_positions()`
- Cache mechanism: `_load_groups_cache()`, `_save_groups_cache()`