# Phase E6 Training Bundle Real Runs — Attempt #96 Summary

**Timestamp**: 2025-11-06T010500Z
**Initiative**: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Phase**: E6/E7 (Bundle SHA256 + Real Training Runs)
**Mode**: TDD
**Status**: **SHA256 Implementation COMPLETE** — Real runs BLOCKED by MemmapDatasetBridge schema mismatch

---

## Executive Summary

**Completed:**
- ✅ **SHA256 field implementation**: Extended `execute_training_job` (training.py:509-523) to compute SHA256 digest for `wts.h5.zip` bundles using 64KB chunked reading.
- ✅ **Result dict extension**: Added `bundle_sha256` field to result dict (training.py:540) alongside existing `bundle_path`.
- ✅ **Test coverage**: Extended `test_execute_training_job_persists_bundle` with 6 new assertions validating SHA256 format (64-char lowercase hex).
- ✅ **RED→GREEN cycle**: Captured expected RED failure (`AssertionError: result must contain 'bundle_sha256' key`), implemented logic, validated GREEN (1/1 test PASSED in 3.76s).
- ✅ **Regression suite**: All training_cli tests PASSED (4/4 in 3.66s), collection proof (4 tests collected).
- ✅ **CLI preservation**: Manifest normalization (training.py:734-748) preserves `bundle_sha256` field via pass-through (line 749).

**Blocked:**
- ❌ **Real training runs**: Both dose=1000 dense/baseline CLI executions failed with `KeyError: 'diff3d'` — `MemmapDatasetBridge` expects canonical `diff3d` field but Phase C NPZs contain legacy `diffraction` field (DATA-001 schema variance).
- ❌ **Bundle archival**: No `wts.h5.zip` artifacts generated due to training failures; checksum validation deferred.

**Unblocker:**
- Update `MemmapDatasetBridge.__init__` to include `diffraction` → `diff3d` fallback (similar to existing loader at ptycho/loader.py), or regenerate Phase C NPZs with canonical `diff3d` field.

---

## Implementation Details

### Code Changes (training.py)

**1. SHA256 computation logic (lines 509-523)**
```python
# Compute SHA256 checksum for bundle integrity validation (Phase E6)
import hashlib
bundle_file = Path(bundle_path)
if bundle_file.exists():
    sha256_hash = hashlib.sha256()
    with bundle_file.open('rb') as f:
        # Read in 64KB chunks to avoid memory issues with large bundles
        for chunk in iter(lambda: f.read(65536), b''):
            sha256_hash.update(chunk)
    bundle_sha256 = sha256_hash.hexdigest()
```

**2. Result dict extension (line 540)**
```python
'bundle_sha256': bundle_sha256,  # NEW: SHA256 checksum for integrity validation
```

**3. Log emission (line 551)**
```python
f.write(f"Bundle SHA256: {bundle_sha256}\n")
```

### Test Extension (test_dose_overlap_training.py:1147-1163)

Added 6 assertions validating SHA256 field:
1. Key presence: `'bundle_sha256' in result`
2. Non-None value: `result['bundle_sha256'] is not None`
3. Type check: `isinstance(sha256_value, str)`
4. Length: `len(sha256_value) == 64`
5. Case: `sha256_value.islower()`
6. Charset: `all(c in '0123456789abcdef' for c in sha256_value)`

---

## Test Evidence

### RED Phase
**File**: `red/pytest_bundle_sha_red.log`
**Selector**: `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv`
**Result**: `FAILED` (1 failed in 4.12s)
**Error**: `AssertionError: result must contain 'bundle_sha256' key after successful bundle persistence`

### GREEN Phase
**File**: `green/pytest_bundle_sha_green.log`
**Selector**: Same
**Result**: `PASSED` (1 passed in 3.76s)

**File**: `green/pytest_cli_bundlepath_green.log`
**Selector**: `pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv`
**Result**: `PASSED` (1 passed in 3.68s)

**File**: `green/pytest_training_cli_suite_green.log`
**Selector**: `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv`
**Result**: `PASSED` (4/10 collected, 4 passed in 3.66s)

### Collection Proof
**File**: `collect/pytest_training_cli_collect.log`
**Selector**: `pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv`
**Result**: 4/10 tests collected (6 deselected) in 3.61s

---

## CLI Execution Logs

### Dense Training (dose=1000, gridsize=2)
**File**: `cli/dose1000_dense_gs2.log`
**Command**: `python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2`
**Result**: `✓ Training manifest written` (1 job completed)
**Status**: BLOCKED — Manifest shows `"status": "failed", "error": "KeyError: 'diff3d'"`

### Baseline Training (dose=1000, gridsize=1)
**File**: `cli/dose1000_baseline_gs1.log`
**Command**: Same with `--view baseline --gridsize 1`
**Result**: `✓ Training manifest written` (1 job completed)
**Status**: BLOCKED — Same KeyError

---

## Blocker Analysis

**Error Signature**:
```
Dataset loading failed: KeyError: "Required key 'diff3d' missing from NPZ file tmp/phase_c_f2_cli/dose_1000/patched_train.npz. Available keys: ['diffraction', 'objectGuess', 'probeGuess', 'Y', 'xcoords', 'ycoords', 'filenames']. See specs/data_contracts.md:13-21 for required schema."
```

**Root Cause**:
- `MemmapDatasetBridge.__init__` (ptycho_torch/memmap_bridge.py) expects canonical `diff3d` field per DATA-001 contract.
- Phase C generation (`studies/fly64_dose_overlap/generation.py`) emits legacy `diffraction` field (amplitude, float32) instead of `diff3d`.
- Existing loader (`ptycho/loader.py`) has `diffraction` → `diff3d` fallback, but `MemmapDatasetBridge` does not.

**Evidence**:
- Manifest `available keys`: `['diffraction', 'objectGuess', 'probeGuess', 'Y', 'xcoords', 'ycoords', 'filenames']`
- Missing: `diff3d`

**Fix Options**:
1. **Preferred**: Update `MemmapDatasetBridge.__init__` to check for `diffraction` if `diff3d` missing, following loader pattern.
2. **Alternative**: Regenerate Phase C NPZs with `diff3d` field (requires updating `generation.py` to emit canonical name).

---

## Artifacts

### Directory Structure
```
plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T010500Z/phase_e_training_bundle_real_runs/
├── red/
│   └── pytest_bundle_sha_red.log
├── green/
│   ├── pytest_bundle_sha_green.log
│   ├── pytest_cli_bundlepath_green.log
│   └── pytest_training_cli_suite_green.log
├── collect/
│   └── pytest_training_cli_collect.log
├── cli/
│   ├── dose1000_dense_gs2.log
│   └── dose1000_baseline_gs1.log
├── data/
│   └── training_manifest.json
└── analysis/
    ├── training_manifest_pretty.json
    └── summary.md (this file)
```

### Missing Artifacts (due to blocker)
- `data/wts.h5.zip` (dense)
- `data/wts.h5.zip` (baseline)
- `analysis/bundle_checksums.txt`

---

## Findings Applied

- **POLICY-001**: PyTorch backend required (torch>=2.2) — training CLI attempted PyTorch path.
- **CONFIG-001**: `update_legacy_dict` bridge not bypassed — SHA256 logic added after training completes.
- **DATA-001**: Schema mismatch exposed — `MemmapDatasetBridge` enforces canonical contract but Phase C violates it.
- **OVERSAMPLING-001**: Not applicable (training didn't reach model instantiation).

---

## Metrics

| Metric | Value |
|--------|-------|
| Code lines added | ~20 (SHA256 logic + result field + log emission) |
| Test lines added | ~20 (6 assertions + docstring updates) |
| RED→GREEN cycle time | ~5 minutes |
| Targeted tests PASSED | 4/4 (100%) |
| Collection proof | 4 tests (no regressions) |
| Real training runs | 0/2 (BLOCKED by schema mismatch) |

---

## Next Actions

1. **Unblock training**: File follow-up fix-plan item to add `diffraction` → `diff3d` fallback in `MemmapDatasetBridge.__init__` (5-10 line change, similar to loader.py pattern).
2. **Rerun CLI**: After unblocker lands, rerun dose=1000 dense/baseline with same commands.
3. **Archive bundles**: Copy `tmp/phase_e_training_gs2/dose_1000/{dense,baseline}/gs*/wts.h5.zip` to `data/`.
4. **Validate checksums**: Run `sha256sum data/wts.h5.zip*`, verify manifest `bundle_sha256` fields match file checksums.
5. **Doc sync**: Update `docs/TESTING_GUIDE.md` §Phase E and `docs/development/TEST_SUITE_INDEX.md` to document `bundle_sha256` field.
6. **Close Phase E6**: Mark as COMPLETE in fix_plan.md once bundles archived and docs synced.

---

## SPEC/ADR Alignment

**SPEC**: specs/ptychodus_api_spec.md:239 (§4.6)
> Checkpoint persistence MUST produce `wts.h5.zip` archives compatible with the TensorFlow persistence contract (§4.6), containing both Lightning `.ckpt` state and bundled hyperparameters for state-free reload.

**Implementation**: SHA256 field added to result dict for downstream manifest consumers (Phase G comparisons).

**ADR**: Not explicitly cited, but aligns with ADR-003 (backend portability) by ensuring bundle integrity metadata travels with artifacts.

---

## Status Line Update Recommendation

```
Phase F COMPLETE (dense + sparse LSQML evidence captured, docs synced). Phase G1 comparison harness implemented (job builder + CLI dry-run); G0.1 inventory [COMPLETE 2025-11-05]; G2 execution orchestrator [COMPLETE 2025-11-05] with n_success/n_failed summary fields + dose=1000 dense_train evidence; real comparisons BLOCKED by Phase E training (wts.h5.zip missing) → **Phase E6 SHA256 implementation COMPLETE** (bundle_sha256 field + test coverage GREEN), real runs BLOCKED by MemmapDatasetBridge schema mismatch (diff3d vs diffraction).
```

---

**Author**: Ralph (Attempt #96)
**Date**: 2025-11-04
**Initiative**: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Commit**: (pending comprehensive suite gate)
