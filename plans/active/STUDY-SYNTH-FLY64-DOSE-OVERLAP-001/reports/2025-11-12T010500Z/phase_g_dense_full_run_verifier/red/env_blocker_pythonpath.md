# Environment Blocker: PYTHONPATH Required for Orchestrator Subprocess Execution

**Date:** 2025-11-11T20:18Z
**Agent:** Ralph (loop i=291)
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 Phase G dense full run verification

## Symptom
```
ModuleNotFoundError: No module named 'ptycho'
```

When running `run_phase_g_dense.py` via `python` or `python3`, the script fails to import `ptycho.metadata.MetadataManager` at line 43.

## Root Cause
The orchestrator script spawns subprocesses via `subprocess.Popen(..., shell=True)`. When the parent Python interpreter is invoked without explicit PYTHONPATH, the editable install's path hook is not propagated to shell subprocesses.

## Evidence
1. Direct Python REPL imports work: `python -c "from ptycho.metadata import MetadataManager"` → OK
2. Script execution fails: `python plans/.../run_phase_g_dense.py ...` → ModuleNotFoundError
3. Setting PYTHONPATH explicitly resolves: `PYTHONPATH=$PWD:$PYTHONPATH python3 plans/.../run_phase_g_dense.py ...` → Success

## Solution Applied
```bash
export PYTHONPATH=/home/ollie/Documents/PtychoPINN:$PYTHONPATH
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
python3 plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
    --hub <hub_path> \
    --dose 1000 \
    --view dense \
    --splits train test \
    --clobber
```

## Follow-up Actions
1. Add PYTHONPATH setup to orchestrator script documentation/shebang handling
2. Consider using `sys.path.insert(0, str(Path(__file__).parents[3]))` at top of orchestrator scripts
3. Update TESTING_GUIDE.md or initiative bin/README with PYTHONPATH requirement for orchestrator scripts

## Status
**Workaround applied; orchestration running under background shell c4a14a**

Exit code: TBD (still running)
Artifacts: Will be captured under cli/run_phase_g_dense_stdout.log when complete
