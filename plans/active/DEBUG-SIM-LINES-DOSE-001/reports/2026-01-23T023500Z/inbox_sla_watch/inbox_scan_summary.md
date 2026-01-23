# Inbox Scan Summary

**Generated:** 2026-01-23T02:19:44.337818+00:00

## Parameters

- **Inbox:** `inbox`
- **Request Pattern:** `dose_experiments_ground_truth`
- **Keywords:** `ack`, `acknowledged`, `confirm`, `received`, `thanks`
- **Since Filter:** None

## Summary

- **Files Scanned:** 5
- **Matches Found:** 3
- **Acknowledgement Detected:** No

## Waiting Clock

- **Last Inbound (from Maintainer <2>):** 2026-01-22T23:22:58.332668+00:00
- **Hours Since Last Inbound:** 2.95 hours
- **Last Outbound (from Maintainer <1>):** 2026-01-23T02:09:58.262344+00:00
- **Hours Since Last Outbound:** 0.16 hours
- **Total Inbound Messages:** 1
- **Total Outbound Messages:** 2

> **Note:** No acknowledgement from Maintainer <2> found. The bundle has been delivered
> but we are still awaiting confirmation from the receiving maintainer.

## SLA Watch

- **Threshold:** 2.00 hours
- **Hours Since Last Inbound:** 2.95 hours
- **Breached:** Yes
- **Notes:** SLA breach: 2.95 hours since last inbound exceeds 2.00 hour threshold and no acknowledgement detected

> **SLA BREACH:** The waiting time has exceeded the configured threshold and no acknowledgement has been received.

## Timeline

Messages sorted by timestamp (ascending):

| Timestamp (UTC) | Actor | Direction | Ack | Keywords | File |
|-----------------|-------|-----------|-----|----------|------|
| 2026-01-22T23:22:58.332668+00:00 | Maintainer 2 | Inbound | No | - | `request_dose_experiments_ground_truth_2026-01-22T014445Z.md` |
| 2026-01-23T01:04:13.890440+00:00 | Maintainer 1 | Outbound | No | ack, confirm | `followup_dose_experiments_ground_truth_2026-01-23T011900Z.md` |
| 2026-01-23T02:09:58.262344+00:00 | Maintainer 1 | Outbound | No | ack, acknowledged, confirm | `response_dose_experiments_ground_truth.md` |

## Matching Files

| File | Match Reason | Actor | Direction | Keywords | Ack | Modified |
|------|--------------|-------|-----------|----------|-----|----------|
| `request_dose_experiments_ground_truth_2026-01-22T014445Z.md` | filename, content | Maintainer 2 | Inbound | - | No | 2026-01-22T23:22:58.332668+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T011900Z.md` | filename, content | Maintainer 1 | Outbound | ack, confirm | No | 2026-01-23T01:04:13.890440+00:00 |
| `response_dose_experiments_ground_truth.md` | filename, content | Maintainer 1 | Outbound | ack, acknowledged, confirm | No | 2026-01-23T02:09:58.262344+00:00 |

## File Previews

### request_dose_experiments_ground_truth_2026-01-22T014445Z.md

```
Title: Request — legacy dose_experiments ground-truth artifacts
From: Maintainer <2> (PtychoPINN active branch, root_dir: ~/Documents/tmp/PtychoPINN/)
To: Maintainer <1> (PtychoPINN dose_experiments branch, root_dir: ~/Documents/PtychoPINN/)

Context
- Initiative: DEBUG-SIM-LINES-DOSE-001 (Phase D amplitude-bias invest...
```

### followup_dose_experiments_ground_truth_2026-01-23T011900Z.md

```
# Follow-Up — Legacy dose_experiments Ground-Truth Bundle

**From:** Maintainer <1> (PtychoPINN dose_experiments branch, root_dir: ~/Documents/PtychoPINN/)
**To:** Maintainer <2> (PtychoPINN active branch, root_dir: ~/Documents/tmp/PtychoPINN/)
**Re:** Follow-up on delivered ground-truth bundle (DEBUG-SIM-LINES-DOSE-00...
```

### response_dose_experiments_ground_truth.md

```
# Response — Legacy dose_experiments Ground-Truth Bundle

**From:** Maintainer <1> (PtychoPINN dose_experiments branch, root_dir: ~/Documents/PtychoPINN/)
**To:** Maintainer <2> (PtychoPINN active branch, root_dir: ~/Documents/tmp/PtychoPINN/)
**Re:** Request — legacy dose_experiments ground-truth artifacts (2026-01-22...
```
