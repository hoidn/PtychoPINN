<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** Grouping-Aware Subsampling for Overlap-Based Training

**Core Technologies:** Python, NumPy, caching, spatial data structures

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

*   **`docs/initiatives/smart-subsampling/plan.md`** (The high-level R&D Plan)
    *   **`docs/initiatives/smart-subsampling/implementation.md`** (This file - The Phased Implementation Plan)
        *   `docs/initiatives/smart-subsampling/phase_1_checklist.md` (Detailed checklist for Phase 1)
        *   `docs/initiatives/smart-subsampling/phase_2_checklist.md` (Detailed checklist for Phase 2)
        *   `docs/initiatives/smart-subsampling/final_phase_checklist.md` (Detailed checklist for Final Phase)

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** To replace the spatially biased sequential subsampling with a "group-then-sample" strategy that ensures both physical coherence and spatial representativeness for overlap-based training (`gridsize > 1`).

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core Data Structure Refactoring**

**Goal:** To refactor the `RawData.generate_grouped_data` method to implement the "group-first" strategy with automated caching, while maintaining full backward compatibility for `gridsize=1`.

**Deliverable:** A modified `ptycho/raw_data.py` with enhanced `generate_grouped_data` method that can find all valid neighbor groups across the entire dataset and cache them for subsequent runs.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] docs/initiatives/smart-subsampling/phase_1_checklist.md`

**Key Tasks Summary:**
*   Implement `_find_all_valid_groups()` method to exhaustively find neighbor groups across the full coordinate set
*   Add automated cache creation and loading logic with `<dataset_name>.g{C}k{K}.groups_cache.npz` naming convention
*   Modify `generate_grouped_data` to use cached groups when available and create cache when missing
*   Ensure the `gridsize=1` path remains completely unchanged for backward compatibility
*   Add comprehensive logging to inform users about cache creation and group sampling

**Success Test:** All tasks in the Phase 1 checklist are marked as done. The data loader can successfully create and use cache files, and training with `gridsize=1` continues to work exactly as before.

---

### **Phase 2: Parameter Interface Enhancement**

**Goal:** To enhance the training script interface to intelligently interpret the `--n-images` parameter based on `gridsize`, simplifying the user experience while maintaining clear documentation of the behavior.

**Deliverable:** A modified `scripts/training/train.py` that accepts a single `--n-images` parameter and automatically adapts its behavior based on `gridsize`, with clear logging and documentation.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] docs/initiatives/smart-subsampling/phase_2_checklist.md`

**Key Tasks Summary:**
*   Implement intelligent `--n-images` interpretation logic in the training script
*   Add clear logging that explains whether N refers to individual images or groups
*   Update the `load_data` function in `ptycho/workflows/components.py` to remove legacy sequential slicing
*   Ensure parameter validation and error handling for edge cases
*   Maintain consistency with existing command-line argument patterns

**Success Test:** All tasks in the Phase 2 checklist are marked as done. Training with `--n-images=512 --gridsize=2` successfully samples 512 groups (2048 total patterns) and logs this behavior clearly to the user.

---

### **Final Phase: Integration Testing and Documentation**

**Goal:** To thoroughly validate the complete system through integration tests, performance validation, and comprehensive documentation updates.

**Deliverable:** A fully validated grouping-aware subsampling system with updated documentation, comprehensive test coverage, and performance benchmarks demonstrating the improvement in spatial representativeness.

**Implementation Checklist:**
*   The detailed, step-by-step implementation for this phase is tracked in: `[ ] docs/initiatives/smart-subsampling/final_phase_checklist.md`

**Key Tasks Summary:**
*   Create comprehensive integration tests for both `gridsize=1` and `gridsize>1` scenarios
*   Perform regression testing to ensure no breaking changes to existing workflows
*   Update documentation in Developer Guide and training script READMEs
*   Conduct performance benchmarks comparing old vs new spatial distribution
*   Validate cache performance improvements on large datasets
*   Update `docs/PROJECT_STATUS.md` to mark initiative as complete

**Success Test:** Execute comprehensive validation with train-on-full, test-on-subset approach:

**Training (Full Dataset):**
1. `ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 10304 --gridsize 1 --nepochs 50 --output_dir final_validation_gs1` (train on all 10,304 individual images)
2. `ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_images 2576 --gridsize 2 --nepochs 50 --output_dir final_validation_gs2` (train on 2576 groups ≈ 10,304 total patterns)

**Testing (1024 patterns):**
1. `ptycho_inference --model_path final_validation_gs1 --test_data datasets/fly/fly001_transposed.npz --n_images 1024 --output_dir final_test_gs1` (test on 1024 individual images)
2. `ptycho_inference --model_path final_validation_gs2 --test_data datasets/fly/fly001_transposed.npz --n_images 256 --gridsize 2 --output_dir final_test_gs2` (test on 256 groups = 1024 total patterns)

Both training and testing workflows complete successfully, demonstrating the system works at production scale with improved spatial representativeness and transparent caching behavior.

---

## 📝 **PHASE TRACKING**

- [x] **Phase 1:** Core Data Structure Refactoring (see `docs/initiatives/smart-subsampling/phase_1_checklist.md`)
- [x] **Phase 2:** Parameter Interface Enhancement (see `docs/initiatives/smart-subsampling/phase_2_checklist.md`)
- [x] **Final Phase:** Integration Testing and Documentation (see `docs/initiatives/smart-subsampling/final_phase_checklist.md`) - ✅ COMPLETE

**Current Phase:** ✅ COMPLETE - All phases successfully implemented and validated
**Next Milestone:** Initiative complete - ready for archive

---

## 🔬 **TECHNICAL SPECIFICATIONS**

**Cache File Format:**
```python
# File: <dataset_name>.g{gridsize}k{overlap_factor}.groups_cache.npz
{
    'all_groups': np.array(shape=(total_num_groups, gridsize**2), dtype=int),
    'dataset_checksum': str,  # For cache invalidation
    'gridsize': int,
    'overlap_factor': float
}
```

**Group-First Algorithm:**
1. Load full coordinate arrays (xcoords, ycoords)
2. Use `get_neighbor_indices()` on complete dataset
3. Extract all valid groups using `sample_rows()` with `n_samples=max_possible`
4. Cache the complete group list
5. Randomly sample from cached groups when `n_groups` is specified

**Parameter Interpretation Logic:**
```python
if gridsize == 1:
    # Traditional behavior: n_images refers to individual images
    selected_indices = range(min(n_images, total_available))
else:
    # Grouping-aware subsampling: n_images refers to number of groups
    n_groups = min(n_images, total_cached_groups)
    selected_groups = random.choice(cached_groups, size=n_groups)
    selected_indices = selected_groups.flatten()
```

---

## 🎯 **VALIDATION CRITERIA**

**Unit Test Requirements:**
1. Cache creation: Verify `.groups_cache.npz` is created on first run
2. Cache loading: Verify cache is loaded on subsequent runs (check via logging)
3. Group sampling: Verify correct number of groups/patterns are selected
4. Backward compatibility: Verify `gridsize=1` behavior is unchanged

**Integration Test Requirements:**
1. Full training workflow with `gridsize=2` and `--n-images=512` completes successfully
2. Cache files are reused across multiple training runs with same parameters
3. Performance improvement: Group finding is fast on subsequent runs with cache
4. Spatial distribution: Random sampling produces more spatially diverse training sets

**Performance Requirements:**
1. First run (cache creation): Acceptable performance degradation vs current implementation
2. Subsequent runs (cache loading): Significant speedup vs repeated group finding
3. Memory usage: Cache files should be reasonably sized (< 100MB for typical datasets)
4. Disk I/O: Cache loading should be faster than recomputing groups

---

## 🔍 **RISK MITIGATION**

**Risk:** Cache invalidation issues if dataset changes
**Mitigation:** Include dataset checksum in cache file and validate on load

**Risk:** Large cache files for massive datasets
**Mitigation:** Implement cache size warnings and optional compression

**Risk:** Breaking existing user workflows
**Mitigation:** Extensive backward compatibility testing and clear migration documentation

**Risk:** Performance degradation on first run
**Mitigation:** Clear user communication about one-time cache creation cost

---

## 📊 **SUCCESS METRICS**

**Functionality Metrics:**
- Zero breaking changes to existing `gridsize=1` workflows
- Successful cache creation and reuse for `gridsize>1` scenarios
- Correct interpretation of `--n-images` parameter in all contexts

**Performance Metrics:**
- Cache creation time: < 2x current data loading time
- Cache loading time: < 10% of group computation time
- Spatial diversity: Measurable improvement in training set spatial coverage

**User Experience Metrics:**
- Single `--n-images` parameter handles all scenarios
- Clear logging explains behavior to users
- Transparent caching with no user intervention required