# Phase D Sparse Downsampling Fix — Summary

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Phase:** D7 (sparse overlap filtering rescue)
**Date:** 2025-11-05
**Mode:** TDD (RED → GREEN)
**Status:** ✓ Complete

---

## Problem Statement

Sparse overlap view generation (Phase D) was raising `ValueError` when initial acceptance rate <10% because all positions violated the 102.4 px threshold individually. Real Phase C coordinates from dose experiments are dense (64 px spacing) but contain viable subsets if we apply greedy spacing-aware downsampling. Without this rescue mechanism, sparse LSQML training (Phase F) was blocked.

**SPEC alignment:**
- `docs/GRIDSIZE_N_GROUPS_GUIDE.md:143-151` — Spacing formula S ≈ (1 − f) × N mandates 102.4 px for sparse view
- `specs/data_contracts.md:207` — DATA-001 NPZ contract enforcement
- `input.md:8-9` — Phase D sparse downsampling rescue

---

## Implementation

### 1. Added greedy spacing selector helper

**File:** `studies/fly64_dose_overlap/overlap.py:194-269`

```python
def greedy_min_spacing_selection(
    coords: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Greedily select positions satisfying minimum spacing constraint.

    Algorithm:
    1. Sort positions by (y, x) to ensure deterministic ordering
    2. Start with empty selection
    3. For each candidate position:
       - If selection is empty, accept it
       - Else, compute distances to all already-selected positions
       - If min distance >= threshold, accept it
    4. Return boolean mask of accepted positions
    """
```

**Key properties:**
- Deterministic: lexsort by (y, x) before greedy pass
- Side-effect free: no params.cfg access (CONFIG-001)
- Efficient: O(n²) but only invoked on low-acceptance fallback

### 2. Wired fallback into generate_overlap_views

**File:** `studies/fly64_dose_overlap/overlap.py:408-439`

When initial acceptance <10%:
1. Log warning and attempt greedy selection
2. If greedy acceptance ≥10%, use greedy mask and update metrics
3. Record `selection_strategy='greedy'` in metadata
4. If even greedy fails, raise descriptive ValueError with both rates

### 3. Authored TDD test case

**File:** `tests/study/test_dose_overlap_overlap.py:405-499`

**Test:** `test_generate_overlap_views_sparse_downsamples`

**Scenario:**
- 5×5 grid @ 64 px spacing (25 positions)
- All positions violate sparse threshold (102.4 px) individually
- Greedy selector identifies valid subset
- Validates DATA-001 compliance + spacing constraint

**RED result:** `ValueError: Insufficient positions meet spacing threshold...`
**GREEN result:** `PASSED` — emits NPZs with n_accepted=3 (12% acceptance)

---

## Test Evidence

### RED (pre-fix)
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/red/pytest_sparse_downsamples_red.log`

```
E    ValueError: Insufficient positions meet spacing threshold for sparse view in train split.
E    Acceptance rate: 0.0% < minimum 10.0%.
E    Min spacing: 64.00 px < threshold: 102.40 px.
```

### GREEN (post-fix)
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/green/pytest_sparse_downsamples_green.log`

```
tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_sparse_downsamples PASSED [100%]
============================== 1 passed in 1.03s ===============================
```

**Console output:**
```
⚠ Initial acceptance rate 0.0% < 10.0%
Attempting greedy spacing selection with threshold=102.40 px...
Greedy selection result:
  Accepted: 3/25 (12.0%)
  ✓ Greedy selection meets minimum threshold
```

### Regression Coverage
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/green/pytest_overlap_suite_green.log`

```
tests/study/test_dose_overlap_overlap.py -k overlap -vv
============================== 11 passed in 1.20s ==============================
```

**Updated test:** `test_generate_overlap_views_paths` now validates greedy fallback behavior (previously tested ValueError, now tests successful rescue).

### Full Suite Gate
**Command:** `pytest -v tests/`
**Result:** `1 failed, 391 passed, 17 skipped` (248.89s)
**Failure:** `tests/io/test_ptychodus_interop_h5.py::test_interop_h5_reader` — pre-existing (ModuleNotFoundError: ptychodus)

---

## Acceptance Metrics

### Greedy Selector Performance (test scenario)

| Metric | Value |
|--------|-------|
| Input positions | 25 |
| Input spacing | 64 px |
| Sparse threshold | 102.4 px |
| Direct acceptance | 0/25 (0.0%) |
| **Greedy acceptance** | **3/25 (12.0%)** |
| Downsampling ratio | 88% reduction |
| Final min spacing | ≥102.4 px (validated) |

### Metadata Enhancements

**New field:** `selection_strategy` ∈ {`'direct'`, `'greedy'`}

Example metadata:
```json
{
  "overlap_view": "sparse",
  "spacing_threshold": 102.4,
  "n_accepted": 3,
  "n_rejected": 22,
  "acceptance_rate": 0.12,
  "selection_strategy": "greedy"
}
```

---

## Compliance

- **CONFIG-001:** ✓ Greedy selector remains pure (no params.cfg access)
- **DATA-001:** ✓ Filtered NPZs validated via `validate_dataset_contract`
- **OVERSAMPLING-001:** ✓ Neighbor count metadata preserved for Phase E
- **POLICY-001:** ✓ No PyTorch dependency changes

---

## Impact on Study Pipeline

### Phase D (this fix)
- **Before:** Sparse view generation aborted with ValueError for real fly64 coordinates
- **After:** Emits sparse NPZs via greedy downsampling (acceptance ≥10%)

### Phase F (downstream unblocker)
- **Before:** Sparse LSQML training blocked (no sparse NPZs available)
- **After:** Can proceed with sparse train/test CLI runs

### CLI Behavior
- Greedy fallback triggers automatically when direct acceptance <10%
- Logs selection strategy and acceptance rates for transparency
- Still raises descriptive error if even greedy selection yields <10%

---

## Files Changed

1. `studies/fly64_dose_overlap/overlap.py`
   - Added `greedy_min_spacing_selection` (lines 194-269)
   - Wired fallback into `generate_overlap_views` (lines 408-439)
   - Added `selection_strategy` to metadata (line 453)

2. `tests/study/test_dose_overlap_overlap.py`
   - Added `test_generate_overlap_views_sparse_downsamples` (lines 405-499)
   - Updated `test_generate_overlap_views_paths` to validate greedy rescue (lines 228-289)
   - Imported `greedy_min_spacing_selection` (line 31)

---

## Next Actions

1. Execute sparse LSQML training runs (Phase F)
2. Compare sparse vs dense reconstruction quality metrics
3. If greedy selection proves insufficient for real data, consider:
   - Regenerating Phase C with wider scan spacing
   - Relaxing overlap fraction constraint in StudyDesign
   - Adding Monte Carlo optimization for spacing selector

---

## Artifacts

- RED log: `red/pytest_sparse_downsamples_red.log`
- GREEN log: `green/pytest_sparse_downsamples_green.log`
- Overlap suite: `green/pytest_overlap_suite_green.log`
- Selector manifest: `collect/pytest_sparse_downsamples_collect.log`
- This summary: `docs/summary.md`
