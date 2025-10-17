# Implementation Plan: Efficient Coordinate Grouping

## Phase 1: Implement the New "Sample-then-Group" Core Logic

### Section 1: Core Logic Implementation

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| 1.A | Create New Private Method | [ ] | **Why:** To build the new logic in isolation. <br> **How:** In `ptycho/raw_data.py`, create `def _generate_groups_efficiently(self, nsamples: int, K: int, C: int, seed: Optional[int] = None) -> np.ndarray:`. <br> **File:** `ptycho/raw_data.py` |
| 1.B | Implement "Sample-then-Group" | [ ] | **Why:** This is the core of the performance improvement. <br> **How:** <br>1. Set random seed if provided<br>2. Randomly sample nsamples indices from `self.xcoords` (seed points)<br>3. Build a single `cKDTree` for all scan positions<br>4. For only the nsamples seed points, query the tree to find their K nearest neighbors<br>5. For each of the nsamples neighbor sets, randomly select C indices to form the final group <br> **Return:** An array of shape `(nsamples, C)` |
| 1.C | Handle Edge Cases | [ ] | **Why:** Ensure robustness for all dataset sizes. <br> **How:** <br>1. If `nsamples > len(self.xcoords)`, use all points as seeds<br>2. If `K < C`, raise informative `ValueError`<br>3. Add logging for sampling strategy used |

### Section 2: Unit Testing

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| 2.A | Create New Test File | [ ] | **Why:** To house the new unit tests for the efficient logic. <br> **How:** Create `tests/test_raw_data_grouping.py`. Set up a `RawData` instance with known coordinates in the `setUp` method. |
| 2.B | Test Output Shape | [ ] | **Why:** To verify the function returns the correct number of groups. <br> **How:** Call `_generate_groups_efficiently(nsamples=100, K=7, C=4)`. Assert the output shape is exactly `(100, 4)`. |
| 2.C | Test Content Validity | [ ] | **Why:** To ensure the groups contain valid neighbor indices. <br> **How:** For a small, deterministic coordinate set, assert that the generated groups contain indices that are spatially close to the seed point. |
| 2.D | Test Edge Cases | [ ] | **Why:** To ensure robustness. <br> **How:** <br>1. Test with `nsamples` larger than total points (should return all possible points as seeds)<br>2. Test with `K < C` (should raise a `ValueError`)<br>3. Test with very small datasets (< 10 points) |
| 2.E | Test Reproducibility | [ ] | **Why:** To ensure deterministic behavior. <br> **How:** Call the function twice with the same seed parameter, assert identical outputs. |
| 2.F | Test Memory Usage | [ ] | **Why:** To verify memory efficiency improvement. <br> **How:** Use `memory_profiler` or `tracemalloc` to compare memory usage between old and new approaches for a 10,000 point dataset. |

## Phase 2: Integration and Legacy Code Removal

### Section 1: Integration

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| 1.A | Integrate New Logic | [ ] | **Why:** To replace the old workflow. <br> **How:** In `generate_grouped_data`, replace the entire `if gridsize > 1:` block (including the caching logic) with a direct call to `self._generate_groups_efficiently()`. |
| 1.B | Unify gridsize=1 Path | [ ] | **Why:** To have a single, clean code path. <br> **How:** Modify the `if gridsize == 1:` block. Instead of calling the legacy `get_neighbor_diffraction_and_positions`, make it also call `self._generate_groups_efficiently(..., C=1)`. This unifies the logic. |
| 1.C | Add Deprecation Warnings | [ ] | **Why:** To notify users of upcoming changes. <br> **How:** Add deprecation warnings to legacy functions before removal, pointing users to the new approach. |
| 1.D | Update Method Signature | [ ] | **Why:** To support reproducibility. <br> **How:** Add optional `seed` parameter to `generate_grouped_data` method and pass through to `_generate_groups_efficiently`. |

### Section 2: Legacy Code Removal

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| 2.A | Remove `_find_all_valid_groups` | [ ] | **Why:** This is the core inefficient function. <br> **How:** Delete the method and all associated caching methods (`_load_groups_cache`, `_save_groups_cache`, `_compute_dataset_checksum`, `_generate_cache_filename`). |
| 2.B | Remove `get_neighbor_diffraction_and_positions` | [ ] | **Why:** This legacy wrapper is no longer needed. <br> **How:** Delete the function from `ptycho/raw_data.py`. Update any references in other modules. |
| 2.C | Remove `group_coords` and `calculate_relative_coords` | [ ] | **Why:** These are the final pieces of the old logic. <br> **How:** Delete these functions. The new logic generates the final groups directly. |
| 2.D | Clean Up Imports | [ ] | **Why:** Remove unused imports. <br> **How:** Remove any imports that were only used by deleted functions (e.g., `hashlib` if only used for caching). |

### Section 3: Validation

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| 3.A | Run Full Test Suite | [ ] | **Why:** To check for regressions. <br> **How:** Run `python -m pytest tests/`. All existing tests, especially the integration test, must still pass. |
| 3.B | Test Backward Compatibility | [ ] | **Why:** To ensure existing workflows aren't broken. <br> **How:** Test with existing training scripts and configs to ensure they still work correctly. |
| 3.C | Create Cache Cleanup Script | [ ] | **Why:** To help users clean up old cache files. <br> **How:** Create `scripts/cleanup_old_cache.py` that finds and removes `*.groups_cache.npz` files. |

## Phase 3: Validation and Documentation

### Section 1: Performance Benchmarking

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| 1.A | Create Benchmark Script | [ ] | **Why:** To measure improvements. <br> **How:** Create `scripts/benchmark_grouping.py` that times both old and new implementations on various dataset sizes. |
| 1.B | Benchmark First-Run Performance | [ ] | **Why:** To quantify the improvement. <br> **How:** <br>1. Clear any old cache files<br>2. Time execution of `generate_grouped_data` on datasets of 1K, 10K, 100K points<br>3. Compare with gridsize=2 using both old and new code |
| 1.C | Benchmark Memory Usage | [ ] | **Why:** To quantify memory improvements. <br> **How:** Use `memory_profiler` to measure peak memory usage for both implementations on large datasets. |
| 1.D | Document Performance Gains | [ ] | **Why:** To record the success of the initiative. <br> **How:** Add benchmark results to validation report (e.g., "First-run time reduced from 120s to 3s, a 40x improvement; Memory usage reduced from 8GB to 200MB"). |

### Section 2: Documentation

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| 2.A | Update `raw_data.py` Docstrings | [ ] | **Why:** To reflect the new, unified logic. <br> **How:** Update the docstring for `generate_grouped_data` to explain the "sample-then-group" strategy and remove any mention of caching or the old logic. |
| 2.B | Update Module Documentation | [ ] | **Why:** To reflect architectural changes. <br> **How:** Update the module-level docstring in `raw_data.py` to remove references to caching and legacy functions. |
| 2.C | Update README.md | [ ] | **Why:** To inform users of improvements. <br> **How:** Add a note about the performance improvements in the data loading section. |
| 2.D | Update CLAUDE.md | [ ] | **Why:** To inform other agents of the change. <br> **How:** Update the data loading section to reflect the new efficient grouping strategy. Remove references to cache files. |
| 2.E | Create Migration Guide | [ ] | **Why:** To help users transition. <br> **How:** Create `docs/migration/coordinate_grouping.md` explaining:<br>1. What changed<br>2. How to clean up old cache files<br>3. How to ensure reproducibility with the seed parameter |

### Section 3: Finalization

| ID | Task Description | State | How/Why & API Guidance |
|----|------------------|-------|------------------------|
| 3.A | Final Code Review | [ ] | **Why:** To ensure code quality. <br> **How:** Review all changes for clarity, style, and correctness. Ensure all legacy code has been successfully removed. |
| 3.B | Create Validation Report | [ ] | **Why:** To document the initiative success. <br> **How:** Create `plans/active/efficient-coordinate-grouping/validation_report.md` with:<br>1. Performance benchmarks<br>2. Memory usage improvements<br>3. Test results<br>4. Code reduction metrics |
| 3.C | Update Project Status | [ ] | **Why:** To mark initiative complete. <br> **How:** Update `PROJECT_STATUS.md` to mark this initiative as completed and document the improvements achieved. |
| 3.D | Commit and Merge | [ ] | **Why:** To finalize the initiative. <br> **How:** Create PR with clear description of changes and improvements. Merge the feature branch into the main development branch. |

## Implementation Notes

### Key Considerations

1. **Random Seed Management**: Ensure the seed parameter is properly threaded through all function calls for reproducibility
2. **Error Messages**: Provide clear, actionable error messages for edge cases
3. **Logging**: Add appropriate logging at INFO level for the sampling process
4. **Performance**: Use NumPy vectorized operations wherever possible
5. **Testing**: Ensure tests cover both small and large datasets

### Dependencies

- NumPy for array operations
- SciPy for cKDTree
- Standard library random or NumPy random for sampling

### Potential Challenges

1. **Neighbor Finding**: Ensure cKDTree queries are efficient for large datasets
2. **Memory Management**: Verify that intermediate arrays don't cause memory spikes
3. **Backward Compatibility**: Carefully test that gridsize=1 behavior is preserved
4. **Random State**: Properly manage random state for reproducibility without affecting global state

### Success Metrics

- [ ] All tests pass
- [ ] Performance improvement > 10x for large datasets
- [ ] Memory usage reduction > 10x for large datasets
- [ ] Code reduction of ~200 lines
- [ ] No regressions in existing functionality

## Status Tracking
Last Phase Commit Hash: 6959fd0a4248f0958f83c106856c3f3f7754159e
