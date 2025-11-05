# Phase C Metadata Guard Follow-up — Canonical Transformation Enforcement

## Loop Summary

**Date:** 2025-11-07T190500Z
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
**Mode:** TDD
**Branch:** feature/torchapi-newprompt
**Status:** ✅ GREEN — Guard implementation complete, tests passing

## Acceptance Criteria Met

Enhanced `validate_phase_c_metadata()` to enforce canonical transformation history:

1. ✅ Requires `transpose_rename_convert` in `metadata["data_transformations"]` (case-sensitive list membership)
2. ✅ Raises `RuntimeError` mentioning both `_metadata` and `transpose_rename_convert` when transformation is missing
3. ✅ Accepts valid metadata with proper transformation records via `MetadataManager.add_transformation_record`
4. ✅ Maintains read-only validation (no mutations or deletions of Phase C outputs)
5. ✅ Follows TYPE-PATH-001 (Path normalization), DATA-001 (NPZ contract), CONFIG-001 (no params.cfg bypass)

## Implementation Details

### Guard Enhancement

**File:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:204-220`

Extended `validate_phase_c_metadata()` with transformation checking logic:

```python
# Require canonical transformation history (transpose_rename_convert)
transformations = metadata.get("data_transformations", [])
has_canonical_transform = any(
    t.get("tool") == "transpose_rename_convert"
    for t in transformations
)

if not has_canonical_transform:
    raise RuntimeError(
        f"Phase C NPZ file missing required canonical transformation in _metadata: {npz_path}. "
        f"Expected 'transpose_rename_convert' in data_transformations list. "
        f"Found transformations: {[t.get('tool') for t in transformations]}. "
        f"Please ensure Phase C pipeline includes transpose_rename_convert canonicalization."
    )
```

### Test Coverage

**File:** `tests/study/test_phase_g_dense_orchestrator.py:261-378`

Added three test cases covering all guard behaviors:

1. **`test_validate_phase_c_metadata_requires_canonical_transform`** (lines 261-318)
   - RED phase: Confirmed guard did not check for `transpose_rename_convert` (test failed with "DID NOT RAISE")
   - GREEN phase: Guard now raises `RuntimeError` matching regex `transpose_rename_convert`
   - Fabricates Phase C NPZ outputs with `_metadata` but missing the required transformation
   - Uses `MetadataManager.save_with_metadata()` to embed metadata with different transformation

2. **`test_validate_phase_c_metadata_accepts_valid_metadata`** (lines 321-378)
   - Positive-path test using `MetadataManager.add_transformation_record()` to embed canonical transformation
   - Verifies guard succeeds without raising when `transpose_rename_convert` is present
   - Demonstrates proper metadata construction pattern for Phase C pipelines

3. **`test_validate_phase_c_metadata_requires_metadata`** (lines 219-258, pre-existing)
   - Baseline check: guard still requires `_metadata` field presence
   - Ensures new transformation check doesn't bypass the original metadata requirement

## Test Results

### RED Phase
```
tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_canonical_transform FAILED
Failed: DID NOT RAISE <class 'RuntimeError'>
[validate_phase_c_metadata] SUCCESS: All Phase C NPZ files contain required _metadata
```
**Outcome:** Confirmed guard did not enforce `transpose_rename_convert` requirement

### GREEN Phase

**Targeted selectors:**
```bash
pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_canonical_transform -vv
pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv
```

**Results:**
- `test_validate_phase_c_metadata_requires_canonical_transform`: PASSED 0.91s
- `test_validate_phase_c_metadata_accepts_valid_metadata`: PASSED 0.92s
- `test_validate_phase_c_metadata_requires_metadata`: PASSED 0.91s

**Collection:**
```bash
pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv
```
- **7 tests collected** (4 summary helper tests + 3 metadata guard tests)

**Full suite:**
```bash
pytest -v tests/
```
- **413 passed**, 1 pre-existing fail (`test_interop_h5_reader`), 17 skipped in 259.36s

## Nucleus Complete

✅ **Core acceptance shipped:**
- Guard implementation enforces `transpose_rename_convert` transformation requirement
- Test coverage validates all three guard behaviors (missing metadata, missing transformation, valid metadata)
- Full test suite GREEN (no regressions introduced)

⏸️ **Deferred per Ralph nucleus principle:**
- Dense orchestrator CLI execution (`python bin/run_phase_g_dense.py --hub ... --dose 1000 --view dense --splits train test`)
- Full Phase C→G pipeline evidence generation
- Rationale: Guard acceptance vs. full pipeline evidence are orthogonal; shipping guard implementation + tests is the loop nucleus

## Artifacts

**Location:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T190500Z/phase_c_metadata_guard_followup/`

```
├── red/
│   └── pytest_guard_transform_red.log         # RED phase: guard did not raise
├── green/
│   ├── pytest_guard_transform_green.log       # GREEN phase: requires_canonical_transform PASSED
│   ├── pytest_guard_valid_green.log           # GREEN phase: accepts_valid_metadata PASSED
│   └── pytest_guard_metadata_green.log        # GREEN phase: requires_metadata PASSED
├── collect/
│   └── pytest_phase_g_guard_collect.log       # Selector collection: 7 tests discovered
└── summary.md                                 # This file
```

## Findings Applied

- **POLICY-001**: Guard remains backend-neutral; no TensorFlow/PyTorch-specific assumptions
- **CONFIG-001**: Guard does not bypass `params.cfg` bridge; read-only metadata validation only
- **DATA-001**: NPZ contract enforced (metadata structure, no payload mutation)
- **TYPE-PATH-001**: All filesystem interactions use `Path.resolve()` to avoid string path bugs
- **OVERSAMPLING-001**: Not applicable to metadata guard (no sampling/grouping logic)

## Next Steps (Optional)

After guard passes with dense evidence, extend CLI summary generation checks for sparse view coverage per `input.md` "Next Up" section.

---

**Commit Message Template:**
```
STUDY-SYNTH-FLY64-DOSE-OVERLAP-001: Enforce transpose_rename_convert in Phase C metadata guard

TDD cycle completed:
- RED confirmed guard did not check for canonical transformation
- GREEN validates three behaviors: missing metadata, missing transformation, valid metadata
- Extended validate_phase_c_metadata() to require transpose_rename_convert in data_transformations
- Added test coverage for all guard paths using MetadataManager helpers
- Full suite: 413 passed/1 pre-existing fail/17 skipped

Acceptance IDs: DATA-001 (NPZ contract enforcement with transformation history)
Findings applied: POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001

Tests: pytest tests/study/test_phase_g_dense_orchestrator.py -k metadata -vv
```
