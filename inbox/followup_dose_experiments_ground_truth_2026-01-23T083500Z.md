# Follow-up: dose_experiments_ground_truth — Per-Actor Severity History

**From:** Maintainer <1>
**To:** Maintainer <2>
**Date:** 2026-01-23T08:35:00Z
**Subject:** SLA breach history now tracked per-actor

---

## Summary

The inbox acknowledgement CLI now persists per-actor SLA severity classifications in the history logs. This provides an audit trail showing which actors were breaching SLA at each scan timestamp.

## Current Status

| Actor | Hours Since Inbound | SLA Threshold | Status |
|-------|---------------------|---------------|--------|
| Maintainer <2> | 4.36 hours | 2.00 hours | **CRITICAL BREACH** (2.36h over) |
| Maintainer <3> | N/A (no inbound) | 6.00 hours | UNKNOWN |

## New History Formats

**JSONL entry** now includes `ack_actor_summary`:
```json
{
  "ack_actor_summary": {
    "critical": [{"actor_id": "maintainer_2", "actor_label": "Maintainer 2", "hours_since_inbound": 4.36, ...}],
    "unknown": [{"actor_id": "maintainer_3", "actor_label": "Maintainer 3", ...}]
  }
}
```

**Markdown history table** now includes "Ack Actor Severity" column:
```
| Generated (UTC) | ... | Ack Actor Severity | Ack Files |
| 2026-01-23T03:44:49 | ... | [CRITICAL] Maintainer 2 (4.36h > 2.00h)<br>[UNKNOWN] Maintainer 3 | - |
```

## Action Requested

Please acknowledge receipt of the dose_experiments_ground_truth bundle by:
1. Confirming the tarball SHA256 matches: `7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72`
2. Confirming datasets load correctly in your environment
3. Replying to this thread with any questions or additional needs

## Artifact References

- History JSONL: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_history/inbox_sla_watch.jsonl`
- History Markdown: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_history/inbox_sla_watch.md`
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_status/status_snippet.md`
- Escalation note: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_status/escalation_note.md`
- Response log: `inbox/response_dose_experiments_ground_truth.md`

---

_Maintainer <1>_
