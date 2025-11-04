# Phase E6 Bundle Size Tracking Implementation — Summary

**Date:** 2025-11-06T190500Z
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Focus:** Phase E6 bundle size tracking + dense/baseline real-run evidence preparation
**Mode:** TDD
**Status:** ✅ GREEN (implementation complete, tests passing)

---

## Problem Statement

Phase E6 requires tracking bundle file size alongside SHA256 checksums for integrity and reproducibility metrics. The Do Now specified:
1. Compute `bundle_size_bytes` in `execute_training_job` whenever a bundle is written
2. Propagate size through manifest serialization
3. Emit size information to stdout alongside bundle path and SHA256

**Quoted SPEC (specs/ptychodus_api_spec.md:220-226):**
> - Manifest: Archives SHALL include a `manifest.dill` at the root with, at minimum, `{'models': [...], 'version': 'X.Y'}`.
>   PyTorch archives MUST additionally include `backend: 'pytorch'`; TensorFlow MAY omit this field and defaults to `'tensorflow'`.
> - Contents: TensorFlow archives contain Keras/SavedModel payloads and serialized custom objects; PyTorch archives contain Lightning
>   `.ckpt` payload(s) and serialized hyperparameters required for state-free reload. The outer archive structure remains identical.

---

## Implementation Summary

### 1. TDD RED Phase
**File:** `tests/study/test_dose_overlap_training.py:1604-1618, 1634-1641, 1721-1784`

Enhanced `test_training_cli_records_bundle_path` with three new assertion blocks:
- **Manifest assertions (1604-1618):** Validate `bundle_size_bytes` field exists in job result dict, is int > 0
- **Stdout count assertions (1634-1641):** Expect 2 Size lines in stdout (one per job: baseline + dense)
- **Stdout format/parity assertions (1721-1784):** Parse Size lines, validate `[view/dose=X.Xe+YY]: N bytes` format, cross-check values against manifest

**RED test run:** FAILED (AssertionError: Expected 2 Size lines in stdout, got 0)

### 2. Production Code Changes
**File:** `studies/fly64_dose_overlap/training.py`

#### A. Size Computation (training.py:515)
```python
bundle_size_bytes = bundle_file.stat().st_size
```

#### B. Result Dict Update (training.py:546)
```python
'bundle_size_bytes': bundle_size_bytes,  # NEW Phase E6 Do Now
```

#### C. Stdout Emission (training.py:755-756)
```python
if result.get('bundle_size_bytes') is not None:
    print(f"    → Size [{job.view}/dose={job.dose:.0e}]: {result['bundle_size_bytes']} bytes")
```

### 3. TDD GREEN Phase
**GREEN test run:** PASSED in 6.82s

### 4. Full Suite Run
**Result:** 397 passed, 1 pre-existing fail (test_interop_h5_reader), 17 skipped in 332.53s

---

## Acceptance Criteria Met

✅ **bundle_size_bytes field:** Computed via `Path.stat().st_size` in execute_training_job
✅ **Manifest propagation:** Included in result dict, serialized to training_manifest.json
✅ **Stdout emission:** `→ Size [view/dose=X.Xe+YY]: N bytes` format with view/dose context
✅ **Cross-validation:** Stdout size values match manifest `bundle_size_bytes` entries
✅ **Type safety:** Size is int > 0 (validated in test assertions)

---

## Files Changed

1. **studies/fly64_dose_overlap/training.py** — 3 changes (size computation, result dict, stdout)
2. **tests/study/test_dose_overlap_training.py** — 4 changes (mock + 3 assertion blocks)

---

## Findings Applied

- POLICY-001: PyTorch backend remains available
- CONFIG-001: Training helpers bridge params.cfg
- TYPE-PATH-001: Path normalization fixes remain in place

---

## Next Actions (Deferred)

Dense gs2 + baseline gs1 deterministic training runs + archival deferred to follow-up loop.
