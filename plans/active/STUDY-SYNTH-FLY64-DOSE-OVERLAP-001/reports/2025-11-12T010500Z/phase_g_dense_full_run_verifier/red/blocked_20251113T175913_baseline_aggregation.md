# Blocker: Baseline Model Missing from Aggregate Metrics

**Timestamp:** $(date -u +"%Y-%m-%dT%H:%M:%SZ")
**Phase:** Phase G metrics aggregation  
**Status:** BLOCKED (root cause identified)

## Root Cause

The `summarize_phase_g_outputs()` function that generates `metrics_summary.json` failed to aggregate Baseline model metrics, even though Baseline metrics exist in the raw comparison outputs.

**Evidence:**
- Raw metrics JSON shows Baseline model present with some null values
- `aggregate_metrics` section ONLY contains: ["Pty-chi (pty-chi)", "PtychoPINN"]
- Baseline is completely missing from aggregation

## Impact Chain

1. Baseline missing → `report_phase_g_dense_metrics.py` fails (cannot compute deltas without Baseline)
2. No aggregate_report.md → `analyze_dense_metrics.py` cannot generate deltas
3. No deltas → post-verify-only fails (missing preview file)
4. No preview → verification bundle incomplete (stays at 0/10)

## Raw Metrics Status

Baseline has null/empty metrics in jobs array:
```json
{
    "model": "Baseline",
    "metric": "mae"
    // No amplitude/phase values
}
```

This suggests either:
1. Baseline inference failed during Phase F or Phase G
2. Baseline metrics computation encountered errors
3. Baseline outputs exist but couldn't be parsed

## Next Investigation Steps

1. Check Phase E baseline training logs: `cli/phase_e_baseline_gs1_dose1000.log`
2. Check Phase G comparison logs for baseline: `analysis/dose_1000/dense/{train,test}/comparison.log`
3. Verify baseline model weights exist in Phase E outputs
4. Check if baseline reconstruction outputs exist in Phase F

## Commands That Will Fail

All downstream commands blocked until Baseline aggregation issue is resolved:
- `report_phase_g_dense_metrics.py`
- `analyze_dense_metrics.py`  
- `run_phase_g_dense.py --post-verify-only`

## Temporary Workaround

Could potentially generate partial reports with just PtychoPINN vs Pty-chi comparisons, but this violates the study design which requires all three models.

