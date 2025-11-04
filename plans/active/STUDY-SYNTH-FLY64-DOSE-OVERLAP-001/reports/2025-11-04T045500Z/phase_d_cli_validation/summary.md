# Phase D CLI Validation Summary

**Date:** 2025-11-04  
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Phase:** D (Group-Level Overlap Views)  
**Focus:** Metrics bundle emission and CLI artifact workflow

## Objective

Implement metrics_bundle_path JSON emission (train/test aggregated) in `generate_overlap_views()` and update CLI to copy the bundle to `--artifact-root`.

## Implementation

### Code Changes

1. **studies/fly64_dose_overlap/overlap.py:381-391**
   - Added metrics bundle creation after processing train/test splits
   - Bundle aggregates both train and test metrics into single JSON
   - Returns `metrics_bundle_path` in results dict

2. **studies/fly64_dose_overlap/overlap.py:500-511**
   - Updated CLI main() to copy metrics_bundle_path to artifact-root
   - Simplified from copying 2 separate files to 1 bundle file
   - Path pattern: `{artifact-root}/metrics/dose_{dose}/{view}.json`

3. **studies/fly64_dose_overlap/overlap.py:513-519**
   - Updated manifest to include `metrics_bundle` field

4. **tests/study/test_dose_overlap_overlap.py:377-402**
   - Enhanced test to validate metrics_bundle_path exists
   - Validates bundle contains both train and test entries
   - Checks JSON structure of aggregated metrics

## Test Results

### Targeted Tests (GREEN)
```bash
pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest -vv
# ✓ PASSED

pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv
# ✓ 2 passed
```

### Test Collection
```bash
pytest tests/study/test_dose_overlap_overlap.py --collect-only -vv
# ✓ 10 tests collected
```

### Comprehensive Suite
```bash
pytest tests/study/ -v
# ✓ 29 passed in 4.02s

pytest -v tests/
# ✓ 376 passed, 17 skipped, 1 failed (pre-existing ptychodus import failure)
```

## CLI Validation

### Artifact Workflow
```bash
python -m studies.fly64_dose_overlap.overlap \
  --phase-c-root tmp/fly64_phase_c_cli \
  --output-root tmp/phase_d_overlap_views \
  --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation
```

**Result:** ✓ Dense view succeeded, metrics bundle copied to artifact-root  
**Artifacts:**
- `metrics/dose_1000/dense.json` — metrics bundle with train/test aggregated

**Blockers:** Phase C generation script has pre-existing bug (TypeError: object of type 'float' has no len() in raw_data.py:227). Used synthetic test data to validate artifact workflow.

### Metrics Bundle Structure

```json
{
  "train": {
    "min_spacing": 50.0,
    "max_spacing": 111.80,
    "mean_spacing": 70.43,
    "median_spacing": 70.71,
    "threshold": 38.40,
    "n_positions": 6,
    "n_accepted": 6,
    "n_rejected": 0,
    "acceptance_rate": 1.0
  },
  "test": {
    "min_spacing": 50.0,
    "max_spacing": 111.80,
    "mean_spacing": 70.43,
    "median_spacing": 70.71,
    "threshold": 38.40,
    "n_positions": 6,
    "n_accepted": 6,
    "n_rejected": 0,
    "acceptance_rate": 1.0
  }
}
```

## Acceptance Criteria Met

- [x] `generate_overlap_views()` returns `metrics_bundle_path`
- [x] Metrics bundle contains both train and test entries
- [x] CLI `--artifact-root` flag copies bundle to reports hub
- [x] Manifest includes `metrics_bundle` field
- [x] Tests validate bundle structure and path existence
- [x] All study tests pass (29/29)
- [x] Comprehensive test suite passes (376 passed, pre-existing failure unrelated)

## Next Actions

1. Update `docs/TESTING_GUIDE.md` with metrics_manifest selector
2. Update `docs/development/TEST_SUITE_INDEX.md` with new test
3. Archive collect-only output
4. Update `docs/fix_plan.md` Attempts History (Attempt #9)
5. Document Phase C generation blocker in fix_plan.md for future resolution

## Artifacts

All artifacts stored at:
`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T045500Z/phase_d_cli_validation/`

- `red/pytest_metrics_bundle.log` — RED test (already passing)
- `green/pytest_metrics_bundle.log` — GREEN test results
- `green/pytest_spacing.log` — Regression test results
- `collect/pytest_collect.log` — Test collection proof
- `cli/phase_d_overlap.log` — CLI execution log
- `metrics/dose_1000/dense.json` — Copied metrics bundle
- `metrics/metrics_inventory.txt` — Artifact inventory
- `pytest.log` — Full test suite results
- `summary.md` — This file
