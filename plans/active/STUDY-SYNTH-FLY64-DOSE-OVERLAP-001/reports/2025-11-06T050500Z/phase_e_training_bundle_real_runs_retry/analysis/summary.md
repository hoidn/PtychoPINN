# Phase E6 Bundle SHA256 Test Enhancement — Attempt #99

**Date:** 2025-11-06T050500Z  
**Mode:** TDD Implementation  
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 Phase E6 Bundle SHA256  
**Branch:** feature/torchapi-newprompt

## Summary

Extended `test_training_cli_records_bundle_path` to validate `bundle_sha256` field in training manifest, ensuring Phase E6 bundle persistence integrity contract (specs/ptychodus_api_spec.md §4.6) is enforced at the CLI level.

## Implementation

### 1. RED Phase (Expected Failure)
- **Test Update:** Added assertions in `test_training_cli_records_bundle_path:1565-1583` requiring:
  - `bundle_sha256` key present in manifest result dict
  - Non-null value for successful training jobs
  - Proper SHA256 format (64-char lowercase hexadecimal)
- **Mock State:** Mock `execute_training_job` returned only `bundle_path`, missing `bundle_sha256`
- **Result:** Test failed as expected with `AssertionError: Job result must contain 'bundle_sha256' field`
- **Artifact:** `red/pytest_manifest_red.log`

### 2. GREEN Phase (Implementation)
- **Mock Update:** Extended `mock_execute_training_job:1490-1507` to return `bundle_sha256`:
  - Uses `abs(hash(job.view))` to generate deterministic 64-char hex mock
  - Ensures proper formatting (lowercase, no negative signs)
- **Test Results:** All training_cli selectors PASSED (4/4)
  - `test_training_cli_filters_jobs`
  - `test_training_cli_manifest_and_bridging`
  - `test_training_cli_invokes_real_runner`
  - `test_training_cli_records_bundle_path`
- **Artifacts:** `green/pytest_training_cli_suite_green.log`

### 3. Collection Proof
- **Command:** `pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv`
- **Result:** 4 tests collected (6 deselected)
- **Artifact:** `collect/pytest_training_cli_collect.log`

### 4. Comprehensive Test Suite
- **Command:** `pytest tests/ -v`
- **Result:** 397 PASSED / 17 SKIPPED / 1 pre-existing FAILED (test_ptychodus_interop_h5_reader)
- **Duration:** 249.64s (4m 9s)
- **Regressions:** None
- **Artifact:** `green/pytest_comprehensive.log`

## Files Modified

- `tests/study/test_dose_overlap_training.py:1490-1507` — Updated mock to return `bundle_sha256`
- `tests/study/test_dose_overlap_training.py:1565-1583` — Added SHA256 validation assertions
- `tests/study/test_dose_overlap_training.py:1590-1597` — Updated success message to include SHA256

## SPEC Alignment

**specs/ptychodus_api_spec.md:239** (§4.6 Bundle Persistence):
> "Checkpoint persistence MUST produce `wts.h5.zip` archives compatible with the TensorFlow persistence contract (§4.6), containing both Lightning `.ckpt` state and bundled hyperparameters for state-free reload."

Bundle integrity requires both `bundle_path` and `bundle_sha256` for reproducibility. This test now validates both fields are present and properly formatted in CLI manifests.

## Findings Applied

- **CONFIG-001:** Test maintains pure mock (no legacy dict pollution)
- **DATA-001:** Test fixtures use canonical NPZ structure
- **POLICY-001:** PyTorch backend assumed available (test runs in torch-enabled env)
- **OVERSAMPLING-001:** gridsize=1/2 respected in job enumeration

## Module Scope

**Tests/docs** — Manifest schema validation only, no production code changes.

## Metrics

- **RED phase:** 1 expected failure captured
- **GREEN phase:** 4/4 selectors PASSED
- **Collection:** 4 tests confirmed
- **Comprehensive suite:** 397 PASSED (no regressions)
- **Lines changed:** ~25 (mock update + assertions + output messages)

## Status

**Phase E6 SHA256 test coverage COMPLETE.** Test now validates both `bundle_path` and `bundle_sha256` fields are present and properly formatted in training manifests, enforcing bundle integrity contract from specs/ptychodus_api_spec.md §4.6.

## Next Actions

1. Mark Phase E6 test enhancement complete in `docs/fix_plan.md`
2. Real CLI training runs remain BLOCKED by Phase D memmap fallback (Attempt #97)
3. Once fallback ships, execute deterministic training runs to populate actual `wts.h5.zip` bundles with real SHA256 checksums
4. Archive bundle evidence (manifest + checksums) for Phase G comparisons

## References

- `input.md:9-24` — Phase E6 Do Now specification
- `specs/ptychodus_api_spec.md:239` — Bundle persistence contract
- `docs/findings.md` — CONFIG-001, DATA-001, POLICY-001, OVERSAMPLING-001
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268` — Phase E6 acceptance criteria
