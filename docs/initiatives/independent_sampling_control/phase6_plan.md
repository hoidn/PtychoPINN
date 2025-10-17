# Phase 6 Plan: K Choose C Oversampling Implementation

## Executive Summary

Phase 6 introduces K choose C oversampling capability to PtychoPINN, enabling creation of more training groups than seed points through combinatorial augmentation. This addresses the current limitation where `nsamples` is capped at the number of available points.

## Current State (End of Phase 5)

### Completed
- ✅ Independent sampling control (`n_subsample` parameter)
- ✅ Fixed help text inconsistencies
- ✅ Basic example scripts created
- ✅ Migration guide drafted

### Discovered Limitation
- System caps groups at number of seed points
- No K choose C combinatorial generation
- Parameter K is mentioned but not fully utilized
- Cannot create overlapping/augmented groups

## Phase 6 Implementation Plan

### Phase 6A: Core Implementation (3-4 hours)

#### Objective
Implement array-based K choose C oversampling in the data pipeline.

#### Tasks
1. **Modify `ptycho/raw_data.py`** (~2 hours)
   ```python
   def _generate_groups_with_oversampling_array(self, nsamples, K, C, seed=None):
       """Generate groups using K choose C combinations for augmentation."""
   ```
   - Pre-compute combination pools
   - Implement smart chunking for memory efficiency
   - Handle edge cases (K < C, insufficient points)

2. **Update `generate_grouped_data` method** (~30 min)
   - Add `enable_oversampling` parameter (default=False for backward compatibility)
   - Route to new or existing implementation based on flag
   - Maintain existing return format

3. **Add configuration support** (~30 min)
   - Add to `TrainingConfig`: `enable_oversampling: bool = False`
   - Add to `TrainingConfig`: `neighbor_pool_size: Optional[int] = None` (for K override)
   - Update argument parser in `components.py`

#### Success Criteria
- [ ] Can generate more groups than seed points
- [ ] Backward compatible (existing workflows unchanged)
- [ ] Memory usage stays reasonable (< 100MB for typical datasets)

### Phase 6B: Integration & Testing (2-3 hours)

#### Objective
Ensure robust integration with existing pipeline.

#### Tasks
1. **Create unit tests** (~1.5 hours)
   ```python
   # tests/test_oversampling.py
   - test_k_choose_c_generation
   - test_oversampling_reproducibility
   - test_memory_efficiency
   - test_backward_compatibility
   ```

2. **Integration tests** (~1 hour)
   - Test with actual training pipeline
   - Verify TensorFlow compatibility
   - Check batch generation consistency

3. **Performance benchmarks** (~30 min)
   - Measure memory usage for various K, C, nsamples
   - Compare training speed with/without oversampling
   - Document performance characteristics

#### Success Criteria
- [ ] All tests pass
- [ ] No performance regression for existing workflows
- [ ] Memory usage documented and acceptable

### Phase 6C: Examples & Documentation (2 hours)

#### Objective
Create compelling examples demonstrating the power of K choose C oversampling.

#### Tasks

1. **Create comparison examples** (~1 hour)

   **Example 1: Traditional 1:1 Mapping**
   ```bash
   # examples/sampling/traditional_vs_oversampled.sh
   
   # Traditional approach: 128 images → 128 groups
   ptycho_train \
       --train_data_file prepare_1e4_photons_5k/dataset/train.npz \
       --n_subsample 512 \
       --n_images 128 \
       --gridsize 2 \
       --enable_oversampling false \
       --output_dir traditional_128groups \
       --config configs/gridsize2_minimal.yaml \
       --nepochs 50
   ```

   **Example 2: K Choose C Oversampling**
   ```bash
   # Same 512 images → 512 groups (4x augmentation)
   ptycho_train \
       --train_data_file prepare_1e4_photons_5k/dataset/train.npz \
       --n_subsample 512 \
       --n_images 512 \
       --gridsize 2 \
       --neighbor_pool_size 7 \
       --enable_oversampling true \
       --output_dir oversampled_512groups \
       --config configs/gridsize2_minimal.yaml \
       --nepochs 50
   ```

   **Example 3: Extreme Oversampling**
   ```bash
   # 512 images → 2048 groups (16x augmentation with K=7, C=4)
   ptycho_train \
       --train_data_file prepare_1e4_photons_5k/dataset/train.npz \
       --n_subsample 512 \
       --n_images 2048 \
       --gridsize 2 \
       --neighbor_pool_size 7 \
       --enable_oversampling true \
       --output_dir extreme_oversampled_2048groups \
       --config configs/gridsize2_minimal.yaml \
       --nepochs 50
   ```

2. **Create comparison study script** (~30 min)
   ```bash
   # examples/sampling/oversampling_study.sh
   # Automated study comparing different oversampling factors
   ```

3. **Update documentation** (~30 min)
   - Add oversampling section to SAMPLING_USER_GUIDE.md
   - Update CONFIGURATION.md with new parameters
   - Add technical details to DEVELOPER_GUIDE.md

#### Success Criteria
- [ ] Examples clearly demonstrate performance difference
- [ ] Documentation explains when to use oversampling
- [ ] Migration path for users wanting this feature

### Phase 6D: Validation & Metrics (1-2 hours)

#### Objective
Quantify the benefits of K choose C oversampling.

#### Tasks
1. **Run comparative study**
   - Traditional: 128, 256, 512 groups (no oversampling)
   - Oversampled: Same base images, 2x, 4x, 8x groups
   - Measure: convergence speed, final quality, memory usage

2. **Generate plots**
   - Training loss curves comparison
   - Final reconstruction quality (PSNR, SSIM)
   - Memory usage over time

3. **Create summary report**
   - Performance improvements
   - Recommended use cases
   - Best practices

#### Success Criteria
- [ ] Clear quantitative benefits demonstrated
- [ ] Guidelines for when to use oversampling
- [ ] Performance/memory trade-offs documented

## Dependencies & Risks

### Dependencies
1. Phase 5 completion (documentation of current system)
2. Access to test datasets (prepare_1e4_photons_5k)
3. GPU availability for performance testing

### Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory explosion with large K | High | Implement chunking, add memory monitoring |
| Breaking existing workflows | High | Feature flag, extensive backward compatibility tests |
| TensorFlow incompatibility | Medium | Test early, have fallback to numpy operations |
| Poor performance gains | Low | Document as experimental feature |

## Timeline

### Sequential Implementation (Recommended)
- **Day 1**: Phase 6A (Core Implementation) - 4 hours
- **Day 2**: Phase 6B (Integration & Testing) - 3 hours  
- **Day 3**: Phase 6C (Examples & Documentation) - 2 hours
- **Day 4**: Phase 6D (Validation & Metrics) - 2 hours

Total: ~11 hours over 4 days

### Parallel Fast-Track (if needed)
- **Track 1**: 6A + 6B (Core + Testing) - Can be done together
- **Track 2**: 6C (Examples) - Can start after 6A basics work
- **Track 3**: 6D (Validation) - Requires 6A+6B complete

Total: ~6-7 hours if parallelized

## Definition of Done

### Phase 6 Complete When:
- [ ] K choose C oversampling implemented and tested
- [ ] Examples demonstrate clear benefits
- [ ] Documentation updated with new capabilities
- [ ] Performance metrics collected and documented
- [ ] Backward compatibility verified
- [ ] PR ready with all tests passing

## Next Steps After Phase 6

1. **Phase 7**: Production optimization
   - GPU-accelerated combination generation
   - Distributed sampling for very large datasets
   - Advanced augmentation strategies

2. **Phase 8**: Advanced features
   - Curriculum learning with progressive oversampling
   - Adaptive K selection based on data density
   - Integration with other augmentation techniques

## Implementation Priority

Given the user's immediate need:
1. **FIRST**: Implement 6A core functionality (highest priority)
2. **SECOND**: Create the specific examples requested (6C partial)
3. **THIRD**: Complete testing and documentation
4. **FOURTH**: Run validation studies

This allows delivering value quickly while ensuring robustness.