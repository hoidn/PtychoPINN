# Legacy Coordinate Grouping Logic Deprecation Initiative

*Created: 2025-08-07*  
*Priority: Medium*  
*Estimated Duration: 1-2 days*

## 🎯 **OBJECTIVE**

**Initiative Name:** Legacy Coordinate Grouping Logic Deprecation

**Problem Statement:** The PtychoPINN codebase contains multiple coordinate grouping approaches - an old inefficient "group-then-sample" approach and a new efficient "sample-then-group" approach. The old logic causes severe performance issues with sample count explosion (e.g., `n_images=500` → 20,000 samples for gridsize=2) and should be systematically removed.

**Proposed Solution:** Systematically deprecate and remove the old inefficient coordinate grouping functions, migrating any remaining usage to the new efficient implementation.

**Success Criteria:**
- All deprecated functions removed from codebase
- No performance regressions in coordinate grouping
- All workflows use efficient "sample-then-group" logic
- Clean separation between legacy compatibility and core functionality

---

## 📚 **BACKGROUND & ANALYSIS**

### Current State

**✅ Modern Efficient Logic (Keep)**:
- `generate_grouped_data()` for gridsize > 1 - Uses "sample-then-group" approach
- Provides 40x performance improvement for large datasets
- Correctly respects `n_images` parameter without sample explosion
- Used by `simulate_and_save.py` and modern workflows

**❌ Legacy Inefficient Logic (Deprecate)**:
- `calculate_relative_coords()` - Wrapper around old group_coords
- `group_coords()` - Core inefficient "group-then-sample" implementation  
- `get_neighbor_diffraction_and_positions()` - Legacy compatibility function
- `RawData.from_simulation()` - Already deprecated but still exists

### Performance Impact

**Old Logic Problems:**
- **Sample Explosion**: Creates 40x more samples than requested
- **O(N²) Operations**: Inefficient neighbor discovery for all points
- **Memory Issues**: Causes OOM errors with large datasets
- **Incorrect Behavior**: Doesn't respect `n_images` parameter semantics

**New Logic Benefits:**
- **Correct Sampling**: Respects `n_images` parameter exactly  
- **O(N) Operations**: Efficient "sample-then-group" with KDTree
- **Memory Efficient**: Reasonable dataset sizes
- **Physics Correct**: Proper neighbor relationships for overlap training

### Usage Analysis

**Functions Still Using Old Logic:**
1. `RawData.from_simulation()` - **DEPRECATED** with warning
2. `get_neighbor_diffraction_and_positions()` - Used by gridsize=1 fallback
3. `generate_grouped_data()` gridsize=1 path - Legacy compatibility

**Impact Assessment:**
- **Critical**: Old logic no longer used in main simulation workflows ✅
- **Low Risk**: Remaining usage is mostly gridsize=1 (minimal sample explosion)
- **Safe to Remove**: Deprecated functions have modern replacements

---

## 🛠️ **PROPOSED APPROACH**

### Phase 1: Analysis and Warning Enhancement
**Duration:** 0.5 days

**Tasks:**
1. **Audit Legacy Function Usage**: Comprehensive search for all calls to deprecated functions
2. **Enhance Deprecation Warnings**: Add stacklevel and migration guidance to all deprecated functions
3. **Document Migration Paths**: Create clear migration guide from old to new APIs
4. **Test Coverage Analysis**: Ensure new logic has equivalent test coverage

### Phase 2: Function Removal
**Duration:** 1 day  

**Tasks:**
1. **Remove `RawData.from_simulation()`**: Already deprecated, safe to remove
2. **Remove `calculate_relative_coords()`**: Replace remaining calls with `generate_grouped_data()`  
3. **Refactor gridsize=1 fallback**: Use efficient logic for all gridsize values
4. **Clean up `group_coords()`**: Remove after ensuring no remaining usage

### Phase 3: Validation and Documentation
**Duration:** 0.5 days

**Tasks:**
1. **Regression Testing**: Verify all simulation workflows still function correctly
2. **Performance Benchmarking**: Confirm no performance regressions
3. **Update Documentation**: Remove references to deprecated functions
4. **Code Review**: Ensure clean separation of concerns

---

## 🎯 **DETAILED DEPRECATION PLAN**

### Functions to Remove

#### 1. `RawData.from_simulation()` (High Priority)
**Status:** Already deprecated with warning  
**Action:** Complete removal  
**Risk:** Low - already has deprecation warning and modern replacement  
**Migration:** Use `scripts/simulation/simulate_and_save.py` directly

#### 2. `calculate_relative_coords()` (High Priority)  
**Current Usage:** Called by deprecated `from_simulation()`  
**Action:** Remove after removing `from_simulation()`  
**Risk:** Low - only used by deprecated code  
**Migration:** Use `generate_grouped_data()` approach

#### 3. `group_coords()` (Medium Priority)
**Current Usage:** Called by `get_neighbor_diffraction_and_positions()` and `calculate_relative_coords()`  
**Action:** Remove after removing dependent functions  
**Risk:** Medium - core grouping function, needs careful validation  
**Migration:** Integrate efficient sampling logic into remaining callers

#### 4. `get_neighbor_diffraction_and_positions()` (Medium Priority)
**Current Usage:** Legacy fallback in `generate_grouped_data()` for gridsize=1  
**Action:** Replace with efficient implementation  
**Risk:** Medium - used by gridsize=1 workflows  
**Migration:** Extend efficient logic to handle gridsize=1 case

### Replacement Strategy

#### For gridsize=1 workflows:
```python
# OLD (inefficient)
result = get_neighbor_diffraction_and_positions(raw_data, N, K=4, nsamples=500)

# NEW (efficient) 
result = raw_data.generate_grouped_data(N, K=4, nsamples=500, config=config)
```

#### For coordinate generation:
```python  
# OLD (sample explosion)
global_offsets, local_offsets, nn_indices = calculate_relative_coords(xcoords, ycoords)

# NEW (controlled sampling)
grouped_data = raw_data.generate_grouped_data(N, K=4, nsamples=len(xcoords), config=config)
global_offsets = get_relative_coords(grouped_data['coords_nn'])[0]
```

---

## ✅ **SUCCESS CRITERIA & VALIDATION**

### Technical Criteria
1. **No Deprecated Functions**: All legacy coordinate grouping functions removed
2. **Performance Maintained**: No regressions in simulation performance  
3. **Correct Sampling**: All workflows respect `n_images` parameter correctly
4. **Memory Efficiency**: No sample explosion issues
5. **Test Coverage**: All functionality covered by tests

### Workflow Validation
1. **Simulation Workflows**: `simulate_and_save.py` continues to work correctly
2. **Training Workflows**: No impact on model training pipelines  
3. **Gridsize=1**: Legacy workflows continue to function efficiently
4. **Gridsize>1**: Modern overlap training remains optimal

### Documentation Updates
1. **API Documentation**: Remove references to deprecated functions
2. **Migration Guide**: Clear instructions for any remaining legacy usage
3. **Architecture Docs**: Updated to reflect single coordinate grouping approach
4. **CLAUDE.md Files**: Remove deprecated function guidance

---

## 🚀 **RISK ASSESSMENT & MITIGATION**

### Risks

**Low Risk - Already Deprecated:**
- `RawData.from_simulation()` removal
- `calculate_relative_coords()` removal  
- **Mitigation:** Functions already have deprecation warnings

**Medium Risk - Legacy Compatibility:**
- `get_neighbor_diffraction_and_positions()` replacement
- gridsize=1 workflow changes  
- **Mitigation:** Comprehensive testing of gridsize=1 scenarios

**Low Risk - Core Function:**  
- `group_coords()` removal
- **Mitigation:** Function only used by other deprecated functions

### Mitigation Strategies

1. **Comprehensive Testing**: Test all simulation workflows with both gridsize=1 and gridsize>1
2. **Gradual Migration**: Replace functions one at a time with validation at each step
3. **Performance Benchmarking**: Measure performance before/after each change
4. **Rollback Plan**: Keep deprecated functions commented out initially, remove after validation

---

## 📊 **EXPECTED OUTCOMES**

### Performance Benefits
- **Reduced Memory Usage**: Elimination of sample explosion scenarios
- **Faster Simulation**: Consistent use of efficient O(N) algorithms  
- **Predictable Behavior**: `n_images` parameter works consistently across all gridsize values

### Code Quality Benefits  
- **Reduced Complexity**: Single coordinate grouping approach
- **Better Maintainability**: No legacy compatibility burden
- **Cleaner APIs**: Modern functions only, no deprecated alternatives
- **Improved Documentation**: Clear, single-path guidance

### User Experience Benefits
- **Predictable Results**: `n_images=500` always means ~500-2000 samples depending on gridsize
- **Better Performance**: No unexpected memory issues or slow simulations
- **Clearer Errors**: No confusing behavior from legacy function interactions

---

## 📁 **IMPLEMENTATION TRACKING**

**Initiative Path:** `plans/future/legacy-coordinate-grouping-deprecation/`

**Files to Create:**
- `implementation.md` - Detailed phase-by-phase implementation plan
- `phase_1_checklist.md` - Analysis and enhanced warnings
- `phase_2_checklist.md` - Function removal and replacement  
- `phase_3_checklist.md` - Validation and documentation

**Key Metrics to Track:**
- Number of deprecated functions removed
- Performance benchmarks (simulation time, memory usage)
- Test coverage percentage for new logic paths
- Documentation completeness score

**Dependencies:**
- Should be completed after current Phase 2 verification work
- No blocking dependencies from other initiatives  
- Can be implemented independently

---

## 🔗 **CROSS-REFERENCES**

- **Related Initiative:** High-Performance Patch Extraction (completed) - established the efficient sampling patterns
- **Affected Documentation:** 
  - <doc-ref type="workflow-guide">scripts/simulation/CLAUDE.md</doc-ref>
  - <doc-ref type="workflow-guide">ptycho/CLAUDE.md</doc-ref>  
  - <doc-ref type="technical">ptycho/loader_structure.md</doc-ref>
- **Impact Assessment:** <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref> section on legacy code removal

**Next Steps:** Run `/implementation` to generate detailed phase checklists when ready to begin implementation.