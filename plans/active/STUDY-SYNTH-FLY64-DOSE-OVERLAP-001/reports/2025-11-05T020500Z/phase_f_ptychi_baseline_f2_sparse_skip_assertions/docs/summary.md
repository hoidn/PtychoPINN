# Phase F2 Sparse Skip Assertions — Summary

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Phase:** F2 (pty-chi baseline execution)
**Attempt:** #84
**Date:** 2025-11-05T020500Z
**Mode:** TDD Implementation
**Status:** COMPLETE ✓

---

## Problem Statement

**SPEC Citation:**
From `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/docs/summary.md:12`:
> Summary calls out `missing_jobs`/`missing_phase_d_count`; tests must lock those fields down.

From `docs/TESTING_GUIDE.md:146`:
> Phase F section mandates manifest + skip summary proof for reconstruction workflows.

**Goal:** Extend `test_cli_skips_missing_phase_d` with assertions for:
1. `manifest["missing_jobs"]` — validate length==6 and sparse-only views
2. `skip_summary["missing_phase_d_count"]` — validate ==6 with schema documentation

---

## Implementation Summary

### Code Changes

**File:** `tests/study/test_dose_overlap_reconstruction.py`

**Lines 609-621:** Added manifest `missing_jobs` assertions
```python
# STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2: Assert manifest['missing_jobs'] structure
# Schema: missing_jobs = List[{dose, view, split, reason}]
assert 'missing_jobs' in manifest, \
    "Manifest must contain 'missing_jobs' field for Phase D skip tracking"

missing_jobs = manifest['missing_jobs']
assert len(missing_jobs) == 6, \
    f"Expected 6 missing sparse jobs in manifest['missing_jobs'], got {len(missing_jobs)}"

# Verify all missing jobs are sparse views only
for job in missing_jobs:
    assert job['view'] == 'sparse', \
        f"Expected all missing jobs to be sparse view, got {job['view']}"
```

**Lines 630-636:** Added skip summary `missing_phase_d_count` assertion
```python
# STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2: Assert skip_summary['missing_phase_d_count']
# Schema: skip_summary = {timestamp: str, skipped_count: int, skipped_jobs: List[...], missing_phase_d_count: int}
assert 'missing_phase_d_count' in skip_summary, \
    "Skip summary must contain 'missing_phase_d_count' field for Phase D tracking"

assert skip_summary['missing_phase_d_count'] == 6, \
    f"Expected missing_phase_d_count=6, got {skip_summary['missing_phase_d_count']}"
```

**Total changes:** 16 lines added (2 assertion blocks with schema documentation)

---

## Test Results

### Targeted Selector — GREEN

**Command:**
```bash
pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d -vv
```

**Log:** `green/pytest_sparse_skip_green.log`

**Result:** ✅ **1 passed in 1.43s**

**Validated:**
- Manifest contains `missing_jobs` field
- `missing_jobs` has length 6
- All 6 missing jobs are sparse view
- Skip summary contains `missing_phase_d_count` field
- `missing_phase_d_count == 6`

### Collection Proof

**Command:**
```bash
pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d --collect-only -vv
```

**Log:** `collect/pytest_sparse_skip_collect.log`

**Result:** ✅ **1 test collected in 0.83s**

---

## CLI Dry-Run Evidence

**Command:**
```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
python -m studies.fly64_dose_overlap.reconstruction \
  --phase-c-root tmp/phase_c_f2_cli \
  --phase-d-root tmp/phase_d_sparse_missing \
  --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/cli \
  --view sparse \
  --dry-run \
  --allow-missing-phase-d
```

**Log:** `cli/dry_run_sparse.log`

**Result:**
```
Total jobs enumerated: 12
Skipped 6 missing Phase D files
Filter --view=sparse: 12 → 0 jobs

Filtered jobs: 0 selected, 12 skipped

  - Missing Phase D files: 6
    1000.0/sparse/train
    1000.0/sparse/test
    10000.0/sparse/train
    10000.0/sparse/test
    100000.0/sparse/train
    100000.0/sparse/test
```

**Artifacts:**
- `cli/reconstruction_manifest.json` — contains `missing_jobs` array with 6 sparse view entries
- `cli/skip_summary.json` — contains `missing_phase_d_count: 6`

### Skip Summary Validation

**File:** `cli/skip_summary.json`

**Key fields verified:**
```json
{
  "timestamp": "2025-11-04T04:40:01.917975",
  "skipped_count": 18,
  "skipped_jobs": [...],
  "missing_phase_d_count": 6
}
```

**Missing Phase D jobs (6 entries):**
- dose_1000/sparse/train — Phase D overlap NPZ not found
- dose_1000/sparse/test — Phase D overlap NPZ not found
- dose_10000/sparse/train — Phase D overlap NPZ not found
- dose_10000/sparse/test — Phase D overlap NPZ not found
- dose_100000/sparse/train — Phase D overlap NPZ not found
- dose_100000/sparse/test — Phase D overlap NPZ not found

All 6 jobs correctly identified with `reason: "Phase D overlap NPZ not found: <path>"`

---

## Findings Applied

| ID | Synopsis | Application |
|:---|:---|:---|
| **CONFIG-001** | Builder stays pure; skip metadata collected without params.cfg mutation | Test validates skip tracking via injected accumulator list, no global state |
| **DATA-001** | Validate present NPZs against canonical contract | Temporary datasets use canonical keys/dtypes (float32 amplitude, complex64) |
| **POLICY-001** | PyTorch required dependency | CLI command respects PyTorch backend requirement |
| **OVERSAMPLING-001** | Skip reasons reference spacing guard | Phase D overlap rejection reasons documented with threshold context |

---

## Metrics

| Metric | Value |
|:---|:---|
| **Tests added** | 0 (extended existing test) |
| **Lines changed** | 16 (assertion blocks) |
| **Test pass rate** | 1/1 (100%) |
| **Execution time** | 1.43s (pytest), <1s (CLI dry-run) |
| **Missing jobs captured** | 6/6 sparse views |
| **Skip metadata fields** | 2 (`missing_jobs`, `missing_phase_d_count`) |

---

## Artifacts Structure

```
plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/
├── green/
│   └── pytest_sparse_skip_green.log        # Targeted selector GREEN evidence
├── collect/
│   └── pytest_sparse_skip_collect.log      # Collection proof (1 test)
├── cli/
│   ├── dry_run_sparse.log                  # CLI execution transcript
│   ├── reconstruction_manifest.json        # Manifest with missing_jobs array
│   └── skip_summary.json                   # Skip summary with missing_phase_d_count
└── docs/
    └── summary.md                          # This document
```

---

## Exit Criteria Validation

✅ **All criteria met:**

1. ✅ `manifest["missing_jobs"]` assertions added (length==6, sparse-only)
2. ✅ `skip_summary["missing_phase_d_count"]` assertion added (==6)
3. ✅ Schema documentation added via inline comments
4. ✅ GREEN pytest log captured
5. ✅ Collection proof captured
6. ✅ CLI dry-run evidence captured with manifest + skip summary artifacts
7. ✅ Findings compliance validated (CONFIG-001, DATA-001, POLICY-001, OVERSAMPLING-001)

---

## Phase F2 Status

**Task F2 (sparse skip instrumentation):** ✅ **COMPLETE**

**Outstanding:**
- Sparse/train and sparse/test LSQML real runs (once Phase D sparse NPZs are regenerated or scope adjusted)
- Documentation/registry sync if selectors change

**Ready for:** Phase G PINN vs pty-chi quality comparisons (once sparse LSQML baseline captured or deferred)

---

## Next Actions

1. **Documentation sync:** Update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` if test expectations change (current assertions extend existing selector, no new nodes)
2. **Sparse LSQML runs:** Execute real sparse/train and sparse/test reconstructions once Phase D overlap datasets include sparse views, or document scope limitation
3. **Phase G preparation:** Proceed with PINN vs pty-chi quality metric comparisons using dense baseline evidence

---

**Attempt #84 Status:** ✅ COMPLETE
**Ralph Signature:** Phase F2 sparse skip assertions delivered with GREEN evidence and CLI artifacts.
