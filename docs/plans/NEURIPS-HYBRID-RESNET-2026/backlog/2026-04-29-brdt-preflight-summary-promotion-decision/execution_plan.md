# BRDT Preflight Summary And Promotion Decision Execution Plan

## Scope

Summarize the completed BRDT preflight and decide whether to recommend
promotion, deferral, or rejection. This plan does not authorize manuscript-table
promotion by itself.

## Required Outputs

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`
- Summary of operator validity, dataset validity, four-row metrics, visual
  bundle availability, dependency issues, and claim boundary.
- Recommendation of `promote_to_evidence_amendment_plan`,
  `defer_after_preflight`, or `reject_for_current_manuscript`.

## Checks

```bash
python - <<'PY'
from pathlib import Path
required = [Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md")]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing BRDT preflight summary: {missing}")
print("brdt preflight summary present")
PY
```
