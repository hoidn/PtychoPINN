# Hub-Relative Path Implementation Summary

**Date**: 2025-11-11
**Commit**: 7dcb2297
**Branch**: feature/torchapi-newprompt

## Implementation Overview

Normalized all success-banner artifact path prints in `run_phase_g_dense.py::main` to hub-relative POSIX paths, enforcing TYPE-PATH-001 compliance throughout the Phase G orchestrator.

## Changes Made

### 1. run_phase_g_dense.py - Full Pipeline Success Banner (lines 1386-1431)

Converted the following absolute Path objects to hub-relative via `.relative_to(hub)`:
- CLI logs
- Analysis outputs
- Aggregate report
- Highlights
- Metrics digest (Markdown)
- Metrics digest log
- Delta metrics (JSON)
- Delta highlights (TXT)
- Delta highlights preview (phase-only, TXT)
- SSIM Grid Summary (phase-only)
- SSIM Grid log
- Verification report
- Verification log
- Highlights check log

### 2. run_phase_g_dense.py - Post-Verify-Only Success Banner (lines 1164-1191)

Converted the following paths to hub-relative:
- CLI logs
- Analysis outputs  
- SSIM Grid Summary (phase-only)
- SSIM Grid log
- Verification report
- Verification log
- Highlights check log

### 3. test_phase_g_dense_orchestrator.py - Test Extensions (lines 1970-1974)

Added two new assertions to `test_run_phase_g_dense_post_verify_only_executes_chain`:
```python
# Assert: stdout contains hub-relative CLI logs and Analysis outputs paths (TYPE-PATH-001)
assert "CLI logs: cli" in stdout, \
    f"Expected success banner to contain 'CLI logs: cli' (hub-relative), but stdout was:\n{stdout}"
assert "Analysis outputs: analysis" in stdout, \
    f"Expected success banner to contain 'Analysis outputs: analysis' (hub-relative), but stdout was:\n{stdout}"
```

## Test Evidence

### Collection Test
```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only -vv
```
**Result**: 2/18 tests collected (16 deselected)
**Log**: `plans/.../reports/.../collect/pytest_collect_orchestrator_post_verify_only.log`

### Targeted Test
```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
```
**Result**: 1 passed in 0.86s
**Log**: `plans/.../reports/.../green/pytest_post_verify_only.log`

## Acceptance Criteria Met

✓ **TYPE-PATH-001**: All success-banner artifact paths use hub-relative POSIX paths  
✓ **TEST-CLI-001**: Test explicitly asserts hub-relative "cli" and "analysis" in stdout  
✓ **Regression**: No existing tests broken, new assertions pass

## Rationale

Before this change, success banners printed absolute paths like:
```
CLI logs: /home/ollie/Documents/PtychoPINN2/plans/active/.../cli
Analysis outputs: /home/ollie/Documents/PtychoPINN2/plans/active/.../analysis
```

After this change, banners print hub-relative paths:
```
CLI logs: cli
Analysis outputs: analysis
```

This prevents brittle absolute paths from polluting logs and makes success banners portable across environments and users.

## Next Steps

1. Execute full Phase C→G dense run with `--clobber` to populate hub with real artifacts
2. Run `--post-verify-only` against fresh artifacts to validate hub-relative banner rendering
3. Update docs/fix_plan.md Attempts History with this implementation + evidence
