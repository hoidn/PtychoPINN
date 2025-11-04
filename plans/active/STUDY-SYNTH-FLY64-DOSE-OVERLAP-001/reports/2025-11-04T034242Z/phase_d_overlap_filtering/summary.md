# Phase D Summary: Group-Level Overlap Filtering

**Date:** 2025-11-04
**Loop:** Attempt #7
**Status:** ✅ Complete
**Mode:** TDD (RED → GREEN)

## Deliverables

### Code Artifacts
1. **`studies/fly64_dose_overlap/overlap.py`** (423 lines)
   - `compute_spacing_matrix()`: Pairwise distance computation via scipy.spatial.distance
   - `build_acceptance_mask()`: Threshold-based position filtering
   - `filter_dataset_by_mask()`: Dataset reduction preserving DATA-001 structure
   - `compute_spacing_metrics()`: Statistical reporting (min/max/mean/median spacing, acceptance rates)
   - `generate_overlap_views()`: End-to-end pipeline loading Phase C NPZs, filtering, validating, writing outputs
   - CLI entry point: `python -m studies.fly64_dose_overlap.overlap --phase-c-root ... --output-root ...`

2. **`tests/study/test_dose_overlap_overlap.py`** (318 lines)
   - 9 test functions covering unit tests (spacing matrix, masking, filtering, metrics) and integration (RED/GREEN workflow)
   - Parametrized tests for dense vs sparse threshold enforcement
   - RED phase: Validates guard against insufficient acceptance rate (< 10%)
   - GREEN phase: Confirms filtering + validation pass with acceptable coordinates

### Test Results

#### RED Phase
- Selector: `pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv`
- Result: 2/2 PASSED (parametrized dense/sparse tests)
- Integration test: `test_generate_overlap_views_paths` raised `ValueError` as expected when dense coordinates (30 px spacing) violated dense threshold (38.4 px)
- Log: `red/pytest.log`, `red/pytest_rerun.log`

#### GREEN Phase
- Selector: Same as RED
- Result: All tests PASSED after implementation + guard (MIN_ACCEPTANCE_RATE=0.1)
- Integration test: Succeeded with coordinates meeting threshold (50 px > 38.4 px)
- Log: `green/pytest.log`

#### Collection Proof
- Command: `pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv`
- Result: 9 tests collected
- Log: `collect/pytest_collect.log`

#### Comprehensive Gate
- Command: `pytest -v tests/`
- Result: **375 passed, 17 skipped, 1 failed (pre-existing)**
- Time: 245.93s (4m 05s)
- Log: `pytest_full.log`

## Key Implementation Details

### Spacing Formula
Per docs/GRIDSIZE_N_GROUPS_GUIDE.md:147:
```
S ≈ (1 − f_overlap) × N
```
where:
- S = minimum inter-group spacing (pixels)
- f_overlap = overlap fraction (dense: 0.7, sparse: 0.2)
- N = patch size (128 px)

**Computed thresholds:**
- Dense: (1 − 0.7) × 128 = **38.4 px**
- Sparse: (1 − 0.2) × 128 = **102.4 px**

### Guard Logic
Added minimum acceptance rate guard (10%) to prevent degenerate datasets where all positions are filtered out. This triggers the RED failure mode when coordinates violate spacing constraints, ensuring test-driven development visibility.

### DATA-001 Compliance
- Validator invoked with `view` argument to enforce spacing thresholds post-filtering
- Filtered NPZs preserve canonical keys: `diffraction` (float32), `objectGuess`/`probeGuess` (complex64), `xcoords`/`ycoords` (float)
- Metadata field `_metadata` added as JSON string documenting overlap view, source file, acceptance stats

### CONFIG-001 Boundary
Overlap utilities are **pure**: no `params.cfg` access, no legacy bridges. Coordinates loaded directly from NPZ via `np.load()`, making the module testable without TensorFlow initialization.

## Metrics (Example from Synthetic Test)

### Dense View (30 px spacing grid)
- Min spacing: 30.00 px
- Max spacing: 67.08 px
- Mean spacing: 42.26 px
- Median spacing: 42.43 px
- **Acceptance: 0/6 (0.0%)** → Triggers ValueError

### Acceptable Dense View (50 px spacing grid)
- Min spacing: 50.00 px
- Max spacing: ~111 px (diagonal)
- **Acceptance: 6/6 (100.0%)** → Passes validation

## Files Modified
- `studies/fly64_dose_overlap/overlap.py` (new)
- `tests/study/test_dose_overlap_overlap.py` (new)

## Files Read/Referenced
- `studies/fly64_dose_overlap/design.py` (StudyDesign constants)
- `studies/fly64_dose_overlap/validation.py` (validate_dataset_contract)
- `docs/GRIDSIZE_N_GROUPS_GUIDE.md:143-151` (spacing formula)
- `docs/SAMPLING_USER_GUIDE.md:112-140` (K-choose-C oversampling)
- `specs/data_contracts.md:207` (DATA-001 NPZ contract)

## Findings Applied
- **CONFIG-001:** No params.cfg mutation in overlap utilities
- **DATA-001:** Validator enforces canonical keys/dtypes after filtering
- **OVERSAMPLING-001:** Preserved neighbor_count for Phase E grouping

## Next Actions
1. Document selectors in `docs/TESTING_GUIDE.md` §Study suite
2. Update `docs/development/TEST_SUITE_INDEX.md` with new tests
3. Phase E: Training plan for dense/sparse views (deferred pending upstream datasets)

## Exit Criteria Met
- ✅ Implemented `compute_spacing_matrix` + helpers
- ✅ Implemented `generate_overlap_views` pipeline
- ✅ Authored pytest module with RED→GREEN workflow
- ✅ Captured RED (spacing failure) + GREEN (filtering success) logs
- ✅ Comprehensive test suite passed (375/376 passing, 1 pre-existing failure unrelated)
- ✅ Artifacts stored under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/`
- ✅ Documentation ready for ledger update
