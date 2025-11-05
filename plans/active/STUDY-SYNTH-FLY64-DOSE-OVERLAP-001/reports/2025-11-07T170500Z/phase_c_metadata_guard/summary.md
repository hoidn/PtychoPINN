# Phase C Metadata Guard Implementation - Summary

## Turn Summary
Implemented Phase C metadata guard to enforce _metadata presence in NPZ outputs; added TDD test coverage and integrated guard into orchestrator main loop.
Fixed validation gap where Phase C outputs lacked metadata tracking, causing downstream failures in tools expecting _metadata field.
Next: execute dense orchestrator CLI to validate guard behavior with real Phase C generation.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T170500Z/phase_c_metadata_guard/ (pytest_guard_green.log, pytest_full_suite.log)

## Acceptance Focus
AT-TBD (Phase C metadata validation guard)
Module scope: tests/docs

## Implementation Details

### RED Phase (Test-Driven Development)
- Added `test_validate_phase_c_metadata_requires_metadata` to `tests/study/test_phase_g_dense_orchestrator.py:219-258`
- Test creates fake Phase C NPZ outputs WITHOUT `_metadata` field
- Expected failure: `AttributeError: module 'run_phase_g_dense' has no attribute 'validate_phase_c_metadata'`
- Logged to: `red/pytest_guard_red.log`

### Implementation
1. **validate_phase_c_metadata() helper** (run_phase_g_dense.py:137-208)
   - TYPE-PATH-001 compliant path normalization
   - Discovers Phase C outputs via glob patterns: `dose_*_{train|test}/fly64_{split}_simulated.npz`
   - Loads NPZ via `MetadataManager.load_with_metadata()` (no pickle=True vulnerability)
   - Raises RuntimeError mentioning `_metadata` if metadata is None
   - Read-only validation (does not mutate NPZ files)

2. **main() guard hook** (run_phase_g_dense.py:538-554)
   - Invoked immediately after Phase C command completes
   - Skipped when `--collect-only` flag is set (keeps dry runs fast)
   - Logs success/failure to stdout for CLI traceability
   - Writes blocker log on RuntimeError and halts pipeline (fail-fast)

### GREEN Phase
- Test passed: `PASSED [100%]` in 1.96s
- Validates:
  - Path normalization (TYPE-PATH-001)
  - Split discovery (train/test)
  - MetadataManager integration
  - RuntimeError with '_metadata' match on missing metadata
- Logged to: `green/pytest_guard_green.log`

### Full Test Suite (Hard Gate)
- Executed: `pytest -v tests/`
- Results: **411 passed**, 1 pre-existing fail (test_interop_h5_reader), 17 skipped
- Duration: 252.80s (4:12)
- Collection verified: 5 tests in `test_phase_g_dense_orchestrator.py`
- Logged to: `full/pytest_full_suite.log`

## SPEC/ADR Alignment
- **DATA-001** (docs/findings.md:14): NPZ contract enforcement via MetadataManager
- **TYPE-PATH-001** (docs/findings.md:21): Path normalization with `Path().resolve()`
- **CONFIG-001** (docs/findings.md:10): No params.cfg mutation; guard is read-only metadata check
- **POLICY-001** (docs/findings.md:8): Guard agnostic to TensorFlow/PyTorch backend choice

## Search Summary
- Existing metadata infrastructure: `ptycho/metadata.py:26` (MetadataManager.load_with_metadata)
- Orchestrator entry point: `plans/active/.../bin/run_phase_g_dense.py:303` (main)
- Test harness: `tests/study/test_phase_g_dense_orchestrator.py:1` (existing summary tests)

## Findings Applied
- POLICY-001: Guard allows both TensorFlow and PyTorch downstream paths
- CONFIG-001: No bypass of update_legacy_dict; guard is read-only
- DATA-001: Guard enforces canonical NPZ contract without altering content
- TYPE-PATH-001: Normalize filesystem paths before IO

## Pitfalls Avoided
- Guard skipped when `--collect-only` to keep dry runs fast ✓
- No mutation/deletion of Phase C outputs (read-only) ✓
- RuntimeError message contains `_metadata` for stable pytest match ✓
- Guard success/failure logged to stdout for CLI traceability ✓
- Path handling compliant with TYPE-PATH-001 via `Path.resolve()` ✓
- Avoided redundant large NPZ loads (MetadataManager only inspects headers) ✓
- pytest + CLI logs under artifacts hub, no repo-root debris ✓

## Next Actions
1. Execute dense orchestrator CLI with guard to validate real Phase C generation behavior
2. Update docs/TESTING_GUIDE.md §2 with new guard selector
3. Update docs/development/TEST_SUITE_INDEX.md with test registry entry
4. Commit with message: `STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 validation: Phase C metadata guard (tests: pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv)`

## Artifacts
- RED test log: `red/pytest_guard_red.log`
- GREEN test log: `green/pytest_guard_green.log`
- Collect log: `collect/pytest_phase_g_orchestrator_collect.log`
- Full suite log: `full/pytest_full_suite.log`
- This summary: `summary.md`

