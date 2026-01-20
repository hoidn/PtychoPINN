# B0f Isolation Test Results

## Test Summary

**Purpose:** Determine whether the gs1_ideal NaN failures are probe-specific (H-PROBE-IDEAL-REGRESSION) or gridsize-specific (H-GRIDSIZE-NUMERIC).

**Test:** Run gs1_custom (gridsize=1 + custom probe) to isolate the variable.

## Results

| Scenario | Probe Mode | Training NaN? | pearson_r | Amp Pred Mean | Amp Truth Mean | LS Scalar |
|----------|------------|---------------|-----------|---------------|----------------|-----------|
| gs1_ideal | idealized | **NO** (fixed) | 0.102 | 0.417 | 2.71 | 1.71 |
| gs1_custom | custom | **NO** | 0.155 | 0.704 | 2.71 | 3.75 |

## Key Findings

1. **Both scenarios train without NaN** after CONFIG-001 bridging was applied (C4f).
   - gs1_ideal previously collapsed at epoch 3 but now completes training successfully.
   - gs1_custom trains without any NaN issues.

2. **gs1_custom produces slightly better correlation** (pearson_r=0.155 vs 0.102).
   - This suggests the custom probe may be more numerically stable.

3. **Both have significant amplitude bias:**
   - gs1_ideal: pred mean = 0.417, truth mean = 2.71 (~6.5x undershoot)
   - gs1_custom: pred mean = 0.704, truth mean = 2.71 (~3.8x undershoot)
   - Custom probe shows less amplitude bias than ideal probe.

4. **CONFIG-001 bridging fix (C4f) resolved the NaN instability.**
   - The gs1_ideal NaN at epoch 3 was NOT inherent to the ideal probe.
   - It was caused by legacy params.cfg not being synced before training/inference.

## Decision Tree Resolution

```
gs1_custom WORKS (no NaN) → Problem WAS gridsize-independent
                           → H-PROBE-IDEAL-REGRESSION NOT confirmed
                           → C4f CONFIG-001 bridging is the fix
```

## Conclusion

The isolation test reveals that:
- The NaN failures were caused by CONFIG-001 violations, not probe type.
- Both ideal and custom probes can train successfully with CONFIG-001 bridging.
- The amplitude bias issue persists in both probe types and is unrelated to NaN stability.
- Next focus should be on resolving the amplitude bias (separate from NaN debugging).

## Artifacts

- gs1_custom run: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/gs1_custom/`
- gs1_ideal baseline: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs1_ideal/`
- pytest log: passed (`test_sim_lines_pipeline_import_smoke`)
