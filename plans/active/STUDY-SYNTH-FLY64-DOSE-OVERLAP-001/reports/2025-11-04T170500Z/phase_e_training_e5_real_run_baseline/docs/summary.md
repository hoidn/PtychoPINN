# Phase E5 Training Runner Integration — Summary

## Objective
Persist skip summary artifacts and capture deterministic Phase E5 training evidence to close the runner integration deliverable.

## Tasks Completed

### T1: RED Test for skip_summary.json (TDD)
- **Status**: ✓ COMPLETE
- **Artifact**: `red/pytest_training_cli_manifest_red.log`
- **Action**: Updated `tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging` to expect:
  - Standalone `skip_summary.json` file under `--artifact-root`
  - Manifest field `skip_summary_path` with relative path to skip summary
  - Matching content between `skip_summary.skipped_views` and `manifest.skipped_views`
- **Expected Failure**: `AssertionError: skip_summary.json not found` (line 745)

### T2: Implementation of skip_summary.json Persistence
- **Status**: ✓ COMPLETE
- **Modified**: `studies/fly64_dose_overlap/training.py:692-731`
- **Changes**:
  - Added Step 5: Emit `skip_summary.json` with `{timestamp, skipped_views, skipped_count}` schema
  - Updated Step 6: Include `skip_summary_path` in manifest (relative path: `"skip_summary.json"`)
  - Added CLI output line when skips exist: `"→ Skip summary written to {path} ({count} view(s) skipped)"`
- **CONFIG-001 Compliance**: Builder remains pure; no params.cfg mutation

### T3: GREEN Tests Validation
- **Status**: ✓ COMPLETE (all PASSED)
- **Artifacts**:
  - `green/pytest_training_cli_manifest_green.log` (1 test, PASSED in 7.48s)
  - `green/pytest_training_cli_skips_green.log` (1 test, PASSED in 7.68s)
  - `green/pytest_training_cli_suite_green.log` (3 tests, PASSED in 6.55s)
  - `collect/pytest_collect.log` (8 tests collected successfully)
- **Test Coverage**:
  - `test_training_cli_manifest_and_bridging`: Validates skip_summary.json file, content schema, and manifest reference
  - `test_build_training_jobs_skips_missing_view`: Validates graceful skip behavior with `allow_missing_phase_d=True`
  - Full CLI suite (`-k training_cli`): 3/3 tests pass

### T4: Deterministic CLI Baseline Run
- **Status**: ✓ COMPLETE (dry-run mode)
- **Artifact**: `real_run/training_cli_real_run.log`
- **Command**:
  ```bash
  python -m studies.fly64_dose_overlap.training \
    --phase-c-root tmp/phase_c_training_evidence \
    --phase-d-root tmp/phase_d_training_evidence \
    --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run \
    --dose 1000 \
    --dry-run
  ```
- **Results**:
  - 6 jobs enumerated (3 doses × 2 variants: baseline + dense)
  - 3 sparse views skipped (dose=1e3, 1e4, 1e5) due to missing Phase D NPZ files
  - 2 jobs executed for dose=1000 (baseline + dense)
  - Manifest and skip_summary.json written successfully
- **Evidence**:
  - `real_run/training_manifest.json` (3.6 KB)
  - `real_run/skip_summary.json` (746 B, 3 skip events)
  - `docs/skip_summary_pretty.json` (formatted copy)

### T5: Documentation and Ledger Updates
- **Status**: 🔄 IN PROGRESS (this document)
- **Remaining**: Update `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`, `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md`, `docs/fix_plan.md`

## Key Outcomes

### skip_summary.json Schema
```json
{
  "timestamp": "2025-11-04T08:34:26.114439Z",
  "skipped_views": [
    {
      "dose": 1000.0,
      "view": "sparse",
      "reason": "NPZ files not found (train=False, test=False). This is expected when Phase D overlap filtering rejected the view due to spacing threshold."
    }
    // ... (2 more skip events for dose=1e4, 1e5)
  ],
  "skipped_count": 3
}
```

### Manifest Extension
- New field: `"skip_summary_path": "skip_summary.json"` (relative path)
- Existing fields preserved: `skipped_views` (array), `skipped_count` (int)
- Downstream tooling can choose to read skip data from:
  - Manifest inline (`manifest['skipped_views']`) for all-in-one consumption
  - Standalone skip summary (`skip_summary.json`) for dedicated skip analytics

## Testing Strategy Updates

### Test Tier Distribution
- **Unit Tests** (8 total, all active):
  - Phase E job matrix enumeration
  - Runner invocation (CONFIG-001 bridging)
  - Dry-run mode behavior
  - CLI filtering logic
  - **NEW**: Manifest + skip_summary.json persistence
  - Backend delegation to PyTorch trainer
  - Real runner integration
  - Graceful skip handling with `allow_missing_phase_d=True`

### Selectors
- `pytest tests/study/test_dose_overlap_training.py -k training_cli`: All CLI-related tests (3 active)
- `pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv`: Skip summary validation (1 test)
- `pytest tests/study/test_dose_overlap_training.py --collect-only`: Discovery check (8 collected)

## Exit Criteria Assessment

- ✅ **Skip summary JSON exists**: `skip_summary.json` written with ≥1 record when sparse view missing
- ✅ **Referenced in summary.md**: This document cites skip_summary.json schema and content
- ✅ **Referenced in manifest**: Manifest includes `skip_summary_path` field
- ✅ **GREEN test logs archived**: All targeted selectors PASSED (see `green/` directory)
- ✅ **Collect proof archived**: `pytest --collect-only` log shows 8 tests collected (see `collect/pytest_collect.log`)
- ✅ **Real-run CLI artifacts captured**: Manifest, skip summary, and stdout log under `real_run/` with deterministic dry-run execution
- 🔄 **Plan/test strategy deliverables COMPLETE**: Pending final ledger updates

## Next Steps
1. Update `docs/TESTING_GUIDE.md` §2 with skip_summary.json expectation
2. Update `docs/development/TEST_SUITE_INDEX.md` to reference new test assertion
3. Mark Phase E5 rows COMPLETE in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md` and `test_strategy.md`
4. Record Attempt #24 in `docs/fix_plan.md` with artifact links and exit criteria met

## References
- `input.md:8-13` (Phase E5 Do Now tasks T1-T5)
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/plan.md:14` (Task checklist)
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:130` (Phase E selectors)
- `specs/data_contracts.md:190` (NPZ key/dtype contract)
- `docs/DEVELOPER_GUIDE.md:68-104` (CONFIG-001 ordering)
- Finding: **OVERSAMPLING-001** (sparse view skips due to spacing threshold enforcement)
