# Implementation Plan: Independent Sampling and Grouping Control

## Executive Summary

Enable independent control of data subsampling and neighbor grouping in PtychoPINN by adding a `--n-subsample` parameter to CLI scripts, while keeping the core `raw_data.py` module unchanged.

**Key Innovation**: Perform subsampling at the data loading stage (in `components.py`) rather than modifying the grouping logic, achieving full flexibility with minimal code changes (~35 lines total).

## Motivation

Currently, the `n_images` parameter controls both subsampling and grouping simultaneously:
- **GridSize = 1**: `n_images` = individual patterns to use
- **GridSize > 1**: `n_images` = number of groups to create (with implicit subsampling)

This coupling prevents use cases like:
- Dense grouping: "Use 2000 images to create 1800 groups"
- Sparse grouping: "Use 10000 images to create 500 groups"
- Memory management: "Subsample large datasets before grouping"

## Solution Architecture

```
Current Flow:
Full Dataset → RawData → generate_grouped_data(nsamples) → Groups
                          ↑
                    Controls both sampling & grouping

Proposed Flow:
Full Dataset → [Subsample] → RawData → generate_grouped_data(n_images) → Groups
                ↑                        ↑
           n_subsample               n_images (groups only)
```

---

## Phase 1: Core Infrastructure (Day 1) ✅ COMPLETED
**Goal**: Add subsampling capability to data loading pipeline

### Pre-Implementation: Baseline Test Status

**Current Test Suite Status** (172 total tests):
- **Core Integration**: `test_integration_workflow` ✅ PASSING
- **Coordinate Grouping**: 14/16 tests passing (2 edge case failures)
- **Model Manager**: ❌ Multiple persistence tests failing
- **Image Registration**: ❌ 9 failures in registration tests
- **Critical for this work**:
  - `test_train_save_load_infer_cycle` ✅ PASSING
  - `test_backward_compatibility` ✅ PASSING
  - `test_efficient_grouping_*` ✅ PASSING

### Regression Testing Strategy

- ✅ **1.0 Establish baseline**
  - ✅ Run and document: `python -m pytest tests/test_integration_workflow.py -v`
  - ✅ Run and document: `python -m pytest tests/test_coordinate_grouping.py -v`
  - ✅ Save baseline results to `phase1_baseline.txt`
  - ✅ Identify any pre-existing failures to ignore

### Checklist

- ✅ **1.1 Update `components.py` data loading**
  - ✅ Add `n_subsample` parameter to `load_data()` function signature
  - ✅ Implement subsampling logic before RawData creation
  - ✅ Add random seed support for reproducible subsampling
  - ✅ Handle edge cases (n_subsample > dataset size)
  - ✅ Add logging for subsampling operations
  - ✅ Update docstring with parameter description
  - ✅ **Regression check**: Run `pytest tests/test_coordinate_grouping.py`

- ✅ **1.2 Update configuration dataclasses**
  - ✅ Add `n_subsample: Optional[int] = None` to TrainingConfig
  - ✅ Add `n_subsample: Optional[int] = None` to InferenceConfig
  - ✅ Document the parameter in class docstrings
  - ✅ Ensure serialization/deserialization works correctly
  - ✅ **Regression check**: Run `pytest tests/test_integration_workflow.py`

- ✅ **1.3 Create unit tests**
  - ✅ Create `tests/test_subsampling.py` with new tests
  - ✅ Test subsampling with various dataset sizes
  - ✅ Test reproducibility with fixed seed
  - ✅ Test edge cases (n_subsample = None, > dataset size, = 0)
  - ✅ Test interaction with existing n_images parameter
  - ✅ Test that Y array is subsampled consistently with diffraction
  - ✅ **Regression check**: Full test suite shouldn't have new failures

### Success Criteria
- ✅ Can call `load_data(..., n_subsample=1000)` and get subsampled RawData
- ✅ All previously passing tests still pass (no regression)
- ✅ New unit tests pass (11/11 tests passing)
- ✅ Test results documented in `test_tracking.md`

---

## Phase 2: Training Script Integration (Day 1-2) ✅ COMPLETED
**Goal**: Enable independent control in training workflow

### Checklist

- ✅ **2.1 Update `scripts/training/train.py`**
  - ✅ Add `--n-subsample` argument to parser (auto-generated from config)
  - ✅ Pass n_subsample to `load_data()` function
  - ✅ Update parameter interpretation logging
  - ✅ Test with various gridsize values (1, 2, 3)

- ✅ **2.2 Update interpretation logic**
  - ✅ Create `interpret_sampling_parameters()` function
  - ✅ Add clear logging messages explaining the interpretation
  - ✅ Document the three modes: legacy, hybrid, independent
  - ✅ Add warnings for potentially problematic configurations

- ✅ **2.3 Integration testing**
  - ✅ Test training with n_subsample < n_images
  - ✅ Test training with n_subsample > n_images  
  - ✅ Test training with n_subsample only (legacy n_images)
  - ✅ Test training with both parameters specified
  - ✅ Verify memory usage reduction with subsampling
  - ✅ **Regression check**: Run integration test `test_train_save_load_infer_cycle`

### Success Criteria
- ✅ Can run: `python train.py --n-subsample 2000 --n-images 500 --gridsize 2`
- ✅ Clear log messages explain what's happening
- ✅ Training proceeds normally with subsampled data
- ✅ No regression in existing training workflows

---

## Phase 3: Inference Script Integration (Day 2) ✅ COMPLETED
**Goal**: Add subsampling control to inference workflow

### Checklist

- ✅ **3.1 Update `scripts/inference/inference.py`**
  - ✅ Add `--n-subsample` argument to parser
  - ✅ Update data loading calls with n_subsample
  - ✅ Ensure consistent interpretation with training script
  - ✅ Update help text and documentation

- ✅ **3.2 Test inference workflow**
  - ✅ Test inference with subsampled test data
  - ✅ Verify metrics calculation works correctly
  - ✅ Test with saved models from Phase 2
  - ✅ Check visualization outputs
  - ✅ **Regression check**: Ensure inference results match baseline for same data

- ✅ **3.3 Additional consistency improvements**
  - ✅ Add `n_images` field to `InferenceConfig` for consistency with `TrainingConfig`
  - ✅ Update `interpret_sampling_parameters()` to match training script signature
  - ✅ Test all changes to ensure no breaking changes
  - ✅ Verify function signatures now consistent between training and inference

### Success Criteria
- ✅ Can run inference with independent sampling control
- ✅ Results are consistent with training configuration
- ✅ All existing inference features work correctly
- ✅ No regression in inference accuracy or performance
- ✅ InferenceConfig and TrainingConfig have consistent field structure

---

## Phase 4: Comparison Script Updates (Day 2-3) ✅ COMPLETED
**Goal**: Enable fair comparisons with independent sampling

### Checklist

- ✅ **4.1 Update `scripts/compare_models.py`**
  - ✅ Add `--n-subsample` argument
  - ✅ Ensure all models use same subsampled data
  - ✅ Update logging to show sampling configuration
  - ✅ Test with PtychoPINN and Baseline models

- ✅ **4.2 Update study script**
  - ✅ Analyzed `run_complete_generalization_study.sh`
  - ✅ Determined no changes needed (uses --n-test-images appropriately)
  - ✅ Documented rationale for keeping as-is
  - ✅ Verified study configurations work correctly

- ✅ **4.3 Validation**
  - ✅ Run comparison with legacy mode (--n-test-images only)
  - ✅ Run comparison with independent control (--n-subsample + --n-test-images)
  - ✅ Verify fair comparison across all model types
  - ✅ Check that metrics are computed correctly
  - ✅ **Regression check**: Compare baseline metrics before/after changes

### Success Criteria
- ✅ Comparison scripts work with new parameter
- ✅ All models receive identical subsampled data
- ✅ Study scripts continue to work unchanged (appropriate use of existing params)
- ✅ No regression in comparison fairness or metrics

---

## Phase 5: Documentation and Examples (Day 3)
**Goal**: Document the new capability for users

### Checklist

- [ ] **5.1 Update user documentation**
  - [ ] Add section to `docs/COMMANDS_REFERENCE.md`
  - [ ] Update parameter tables in relevant guides
  - [ ] Add examples showing different use cases
  - [ ] Document the interpretation modes

- [ ] **5.2 Create example scripts**
  - [ ] Dense grouping example
  - [ ] Sparse grouping example  
  - [ ] Memory-constrained example
  - [ ] Migration example (old → new syntax)

- [ ] **5.3 Update help text**
  - [ ] Ensure all CLI help messages are clear
  - [ ] Add examples to help text
  - [ ] Update any auto-generated documentation

### Success Criteria
- [ ] Users can understand when and how to use n_subsample
- [ ] Documentation clearly explains the parameter interaction
- [ ] Examples demonstrate practical use cases

---

## Phase 6: Advanced Testing and Validation (Day 3-4)
**Goal**: Ensure robustness and performance

### Checklist

- [ ] **6.1 Performance testing**
  - [ ] Benchmark subsampling overhead
  - [ ] Test with very large datasets (>100k images)
  - [ ] Measure memory savings with subsampling
  - [ ] Compare training convergence with different sampling strategies

- [ ] **6.2 Edge case testing**
  - [ ] Test with n_subsample = n_groups = n_dataset
  - [ ] Test with n_subsample = 1, gridsize = 3
  - [ ] Test with incompatible parameter combinations
  - [ ] Test error messages and warnings

- [ ] **6.3 Comprehensive regression testing**
  - [ ] Run full test suite: `python -m pytest tests/ -v`
  - [ ] Document all test results in `test_results_final.txt`
  - [ ] Compare with baseline from Phase 1
  - [ ] Verify backward compatibility with existing scripts
  - [ ] Test saved model loading/inference
  - [ ] Check reproducibility with fixed seeds
  - [ ] Run critical integration tests:
    - `test_train_save_load_infer_cycle`
    - `test_backward_compatibility`
    - `test_efficient_grouping_*`

### Success Criteria
- [ ] No new test failures (same pass/fail count as baseline)
- [ ] No performance regression for existing workflows
- [ ] Clear error messages for invalid configurations
- [ ] All edge cases handled gracefully
- [ ] Test report shows improvement or stability

---

## Implementation Notes

### Key Design Decisions

1. **Modify loading, not grouping**: Keep `raw_data.py` unchanged by subsampling during data load
2. **Optional parameter**: `n_subsample=None` maintains backward compatibility
3. **Clear semantics**: `n_subsample` controls data selection, `n_images` controls group creation
4. **Fail gracefully**: Warning messages for suboptimal configurations, not hard failures

### Risk Mitigation

- **Risk**: Breaking existing workflows
  - **Mitigation**: Optional parameter with None default preserves all current behavior

- **Risk**: Confusing parameter interaction
  - **Mitigation**: Clear logging messages and comprehensive documentation

- **Risk**: Performance impact
  - **Mitigation**: Subsampling uses efficient numpy operations, minimal overhead

### Testing Strategy

1. **Unit tests**: Test individual functions in isolation
2. **Integration tests**: Test full workflows end-to-end
3. **Regression tests**: Ensure existing functionality unchanged
4. **Performance tests**: Verify no significant overhead

### Continuous Test Monitoring

Throughout implementation, maintain a test tracking document (`test_tracking.md`) with:

```markdown
## Test Status Tracking

### Baseline (Before Changes)
- Integration tests: 1/1 passing
- Coordinate grouping: 14/16 passing
- Total suite: ~140/172 passing

### After Phase 1
- [ ] No new failures
- [ ] Integration tests: 1/1 passing
- [ ] Coordinate grouping: 14/16 passing

### After Phase 2
- [ ] Training integration test passing
- [ ] New subsampling tests: X/Y passing

### After Phase 6
- [ ] Full regression suite matches baseline
```

**Key Tests to Monitor**:
- `test_train_save_load_infer_cycle` - Critical integration test
- `test_backward_compatibility` - Ensures no breaking changes
- `test_efficient_grouping_*` - Core grouping logic
- `test_coordinate_grouping.py` - Data pipeline integrity

---

## Rollout Plan

1. **Internal Testing** (Day 1-4): Complete all phases with developer testing
2. **Beta Testing** (Day 5-6): Test with real workflows and datasets
3. **Documentation Review** (Day 7): Ensure documentation is clear and complete
4. **Release** (Day 8): Merge to main branch with announcement

## Success Metrics

- [ ] Zero existing tests broken
- [ ] Can control subsampling and grouping independently
- [ ] Performance overhead < 1% for existing workflows
- [ ] Clear documentation and examples available
- [ ] Positive user feedback on flexibility

## Future Extensions

- Add support for stratified sampling (maintaining spatial distribution)
- Enable progressive loading for very large datasets
- Add density-based grouping as alternative to KNN
- Support for saving/loading sampling configurations

---

## Commands Quick Reference

```bash
# Legacy mode (unchanged)
python train.py --n-images 1000 --gridsize 2

# Independent control
python train.py --n-subsample 4000 --n-images 1000 --gridsize 2

# Dense grouping
python train.py --n-subsample 1200 --n-images 1000 --gridsize 2

# Sparse grouping  
python train.py --n-subsample 10000 --n-images 500 --gridsize 2

# Memory-constrained
python train.py --n-subsample 5000 --n-images 2000 --gridsize 1
```