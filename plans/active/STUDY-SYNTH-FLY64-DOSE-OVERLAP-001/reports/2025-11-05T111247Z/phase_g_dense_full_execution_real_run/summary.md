# Phase G Dense Real-Run Evidence — Summary (2025-11-05T111247Z)

## Status: BLOCKED

**Ralph Loop Timestamp**: 2025-11-05T11:21:32Z → 2025-11-05T11:40:34Z (approximately 19 minutes)

## What Was Executed
- Pre-flight regression test: PASSED (pytest test_run_phase_g_dense_exec_runs_analyze_digest)
- Phase C dataset generation: **COMPLETED SUCCESSFULLY** (15 minutes)
- Phase C metadata validation: **FAILED** (validator mismatch)
- Phases D/E/F/G: NOT EXECUTED (blocked)

## Blocker Summary
**Error**: Phase C metadata validator expects directory structure `dose_*_train/` but Phase C generates `dose_*/patched_train.npz`

**Exit Code**: 1 (from orchestrator)

**Error Location**: `run_phase_g_dense.py:231` in `validate_phase_c_metadata()`

**Root Cause**: Validator-Generator Mismatch
- Validator (lines 225-226) has outdated comment claiming Phase C creates `dose_<dose>_<split>/` directories
- Actual Phase C output structure is `dose_<dose>/patched_{split}.npz`

## Phase C Outputs (Confirmed Present)
Generated at: `plans/.../phase_c/`

### Dose 1000
- `dose_1000/simulated_raw.npz` (1.1 GB)
- `dose_1000/canonical.npz` (1.1 GB)
- `dose_1000/patched.npz` (1.1 GB)
- `dose_1000/patched_train.npz` (581 MB)
- `dose_1000/patched_test.npz` (597 MB)

### Dose 10000
- Similar structure

### Dose 100000
- Similar structure

### Manifest
- `run_manifest.json` (Phase C provenance)

## Provenance Confirmation
- **CONFIG-001**: AUTHORITATIVE_CMDS_DOC exported before execution
- **DATA-001**: Phase C generation performed internal validation (PASSED for all splits)
- **Metadata**: Present in all NPZ files (confirmed by generation log)

## Artifacts
- Pre-flight test log: `green/pytest_orchestrator_dense_exec_recheck.log` (PASSED)
- Phase C generation log: `cli/phase_c_generation.log` (367 lines, SUCCESS)
- Main orchestrator log: `cli/run_phase_g_dense.log` (validation error at end)
- Blocker log: `analysis/blocker.log`
- Blocker analysis: `summary/blocker_analysis.md` (detailed root cause)

## Next Actions
1. **Fix validator** to match actual Phase C structure (RECOMMENDED)
   - Update glob pattern at line 231: `dose_*/patched_{split}.npz` instead of `dose_*_{split}/`
   - Remove fly64_{split}_simulated.npz filename check (lines 241-248)
   - Keep metadata validation logic (lines 252-279)

2. **Re-run pipeline** with fixed validator using same --clobber command

3. **Alternative**: Fix Phase C generation to create split subdirectories (larger change, not recommended)

## Runtime Metrics
- Pre-flight test: <1 second
- Phase C generation: ~15 minutes (3 dose levels × 5 stages each)
- Total elapsed: ~19 minutes (blocked at validation)
- Expected full pipeline runtime: 2-4 hours (if unblocked)

## MS-SSIM/MAE Deltas
NOT AVAILABLE (Phases D/E/F/G not executed due to blocker)

---

**Conclusion**: Phase C data is valid and complete. The blocker is a code-level mismatch between validator expectations and actual output structure, not a data quality issue. Fix the validator (Option A in blocker_analysis.md) and re-run.
