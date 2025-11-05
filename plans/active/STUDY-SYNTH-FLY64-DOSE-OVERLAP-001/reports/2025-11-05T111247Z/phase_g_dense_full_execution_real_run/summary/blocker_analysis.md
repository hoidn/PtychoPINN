# Phase G Dense Pipeline Blocker Analysis (2025-11-05T111247Z)

## Summary
Phase G dense pipeline execution blocked during Phase C metadata validation. The validator expects a directory structure (`dose_*_train/`, `dose_*_test/`) that doesn't match the actual Phase C generation output format.

## Timeline
- **Start**: 2025-11-05T11:21:32Z
- **Phase C Complete**: 2025-11-05T11:36:39Z (approximately 15 minutes)
- **Validation Error**: 2025-11-05T11:36:39Z
- **Exit**: Pipeline returned exit code 1

## Error Signature
```
RuntimeError: Phase C train split directory not found under
/home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T111247Z/phase_g_dense_full_execution_real_run/data/phase_c.
Expected pattern: dose_*_train/
```

## Root Cause
**Validator-Generator Mismatch**: The `validate_phase_c_metadata()` function in `run_phase_g_dense.py:231` expects:
```
phase_c/
  dose_1000_train/
    fly64_train_simulated.npz
  dose_1000_test/
    fly64_test_simulated.npz
```

But Phase C generation (`studies/fly64_dose_overlap/generation.py`) produces:
```
phase_c/
  dose_1000/
    patched_train.npz
    patched_test.npz
    canonical.npz
    patched.npz
    simulated_raw.npz
  dose_10000/
    ...
  dose_100000/
    ...
```

## Evidence
1. **Phase C outputs** (confirmed present):
   - `plans/.../phase_c/dose_1000/patched_train.npz` (581 MB)
   - `plans/.../phase_c/dose_1000/patched_test.npz` (597 MB)
   - Similar structure for dose_10000 and dose_100000

2. **Blocker log**: `plans/.../analysis/blocker.log`

3. **Phase C generation log**: `plans/.../cli/phase_c_generation.log` (367 lines, successful completion)

4. **Main orchestrator log**: `plans/.../cli/run_phase_g_dense.log`

## Phase C Provenance (Confirmed)
- All dose levels generated: 1000, 10000, 100000
- DATA-001 validation: PASSED for all splits
- CONFIG-001: AUTHORITATIVE_CMDS_DOC exported
- Metadata tracking: File manifest generated at `phase_c/run_manifest.json`

## Impact
- **Blocked**: Phases D/E/F/G cannot execute
- **Data**: Phase C datasets ARE valid and complete, just in a different location than validator expects
- **No data loss**: All Phase C outputs preserved

## Next Steps (Options)
### Option A: Fix Validator (Recommended)
Update `run_phase_g_dense.py::validate_phase_c_metadata()` to match actual Phase C output structure:
- Change glob pattern from `dose_*_{split}` to `dose_*/patched_{split}.npz`
- Remove file name check for `fly64_{split}_simulated.npz`
- Keep metadata validation logic intact

### Option B: Fix Generator
Update `studies/fly64_dose_overlap/generation.py` to create split subdirectories matching validator expectations. This would require restructuring the output logic.

### Option C: Bypass Validation (Not Recommended)
Comment out or skip the validator. This would lose the metadata guard but unblock pipeline execution.

## Recommendation
**Option A** is preferred because:
1. Phase C outputs are correct and DATA-001 compliant
2. Validator comment (line 225-226) is outdated documentation
3. Smaller code change surface
4. Preserves metadata validation (critical for provenance)

## Artifacts
- Blocker log: `plans/.../analysis/blocker.log`
- Phase C generation log: `plans/.../cli/phase_c_generation.log`
- Main log: `plans/.../cli/run_phase_g_dense.log`
- This analysis: `plans/.../summary/blocker_analysis.md`
