# Orchestrator Bug: Phase C Generates All Doses Regardless of --dose Parameter

## Issue
`run_phase_g_dense.py` was invoked with `--dose 1000` but Phase C generation (`studies.fly64_dose_overlap.generation`) ignores this parameter and generates all 3 doses (1000, 10000, 100000).

## Evidence
- Command: `python run_phase_g_dense.py --hub <hub> --dose 1000 --view dense --splits train test --clobber`
- Phase C stdout shows: "Dose levels: [1000.0, 10000.0, 100000.0]"
- Orchestrator at line 939-943 doesn't pass `--dose` parameter to generation module

## Root Cause
File: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:939-943`

```python
phase_c_cmd = [
    "python", "-m", "studies.fly64_dose_overlap.generation",
    "--base-npz", str(base_npz),
    "--output-root", str(phase_c_root),
]
# Missing: dose parameter not passed to generation module
```

The `generation` module has its own hardcoded dose list instead of accepting a CLI parameter.

## Impact
- 3x longer Phase C execution time (~12 minutes instead of ~4 minutes)
- Generates 7+ GB of unnecessary data (dose_10000, dose_100000)
- Wastes computational resources

## Status
- dose_1000 generation completed successfully (all 5 stages, 4.7GB, ~4 min)
- Pipeline killed at 10:52 while generating dose_10000 to avoid wasting time
- dose_1000 outputs are valid and can be used for Phases D-G

## Next Action
Phase D onwards appear to use `--dose` parameter correctly (based on code review), so a workaround is to manually delete dose_10000/dose_100000 directories and continue from Phase D, or let Phase C complete and waste the time.

## Recommendation for Future Fix
Add `--doses` parameter to `studies.fly64_dose_overlap.generation` module and pass it from orchestrator, OR have orchestrator filter/delete unwanted dose directories after Phase C completes.

## File Location
plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:939-943
