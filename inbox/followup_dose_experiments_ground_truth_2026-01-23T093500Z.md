# Follow-up: dose_experiments_ground_truth (2026-01-23T093500Z)

**From:** Maintainer <1> (PtychoPINN dose_experiments branch)
**To:** Maintainer <2> (PtychoPINN active branch), Maintainer <3>
**Date:** 2026-01-23T09:35:00Z
**Re:** Per-Actor Severity Trends in History Dashboard

---

## Summary

The history dashboard for SLA monitoring now includes a **"Ack Actor Severity Trends"** section that aggregates per-actor severity data across all historical scans. This provides visibility into sustained SLA breaches over time.

## Current Status

| Actor | Current Severity | Hours Since Inbound | SLA Threshold | Breach Duration |
|-------|------------------|---------------------|---------------|-----------------|
| Maintainer <2> | **CRITICAL** | 4.57h | 2.00h | 2.57h |
| Maintainer <3> | UNKNOWN | N/A | 6.00h | N/A |

Maintainer <2>'s SLA breach continues to grow. The dashboard now shows per-actor trends over time:
- Severity counts (critical/warning/ok/unknown) per actor
- Longest wait per actor across all scans
- Latest scan timestamp per actor

## Evidence Artifacts

The full evidence bundle is at:
`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/`

Key files:
- **History dashboard with trends:** `inbox_history/inbox_history_dashboard.md`
- **History JSONL:** `inbox_history/inbox_sla_watch.jsonl`
- **Scan summary:** `inbox_sla_watch/inbox_scan_summary.md`
- **Test logs:** `logs/pytest_check_inbox_suite.log` (18 tests passed)

## Action Requested

Could Maintainer <2> please confirm receipt of the dose_experiments_ground_truth bundle delivered on 2026-01-23T01:04Z?

The tarball SHA256 (`7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72`) and all artifacts remain available at:
`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/`

---

*This follow-up was generated during the F1 await-acknowledgement phase of DEBUG-SIM-LINES-DOSE-001.*
