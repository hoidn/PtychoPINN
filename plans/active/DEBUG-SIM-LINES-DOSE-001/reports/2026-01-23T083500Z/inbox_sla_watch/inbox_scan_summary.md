# Inbox Scan Summary

**Generated:** 2026-01-23T03:44:49.251081+00:00

## Parameters

- **Inbox:** `inbox`
- **Request Pattern:** `dose_experiments_ground_truth`
- **Keywords:** `acknowledged`, `confirm`, `received`, `thanks`
- **Ack Actors:** `maintainer_2`, `maintainer_3`
- **Since Filter:** None

## Summary

- **Files Scanned:** 6
- **Matches Found:** 4
- **Acknowledgement Detected:** No

## Waiting Clock

- **Last Inbound (from Maintainer <2>):** 2026-01-22T23:22:58.332668+00:00
- **Hours Since Last Inbound:** 4.36 hours
- **Last Outbound (from Maintainer <1>):** 2026-01-23T02:21:49.305956+00:00
- **Hours Since Last Outbound:** 1.38 hours
- **Total Inbound Messages:** 1
- **Total Outbound Messages:** 3

> **Note:** No acknowledgement from Maintainer <2> found. The bundle has been delivered
> but we are still awaiting confirmation from the receiving maintainer.

## SLA Watch

- **Threshold:** 2.50 hours
- **Hours Since Last Inbound:** 4.36 hours
- **Deadline (UTC):** 2026-01-23T01:52:58.332668+00:00
- **Breached:** Yes
- **Breach Duration:** 1.86 hours
- **Severity:** critical
- **Notes:** SLA breach: 4.36 hours since last inbound exceeds 2.50 hour threshold and no acknowledgement detected

> **SLA BREACH:** The waiting time has exceeded the configured threshold and no acknowledgement has been received.

## Ack Actor SLA Summary

Per-actor SLA status grouped by severity:

### Critical (Breach >= 1h)

- **Maintainer 2**: 4.36 hrs since inbound (threshold: 2.00 hrs) — Breached: Yes
  - SLA breach: 4.36 hours since last inbound exceeds 2.00 hour threshold

### Unknown (No Inbound)

- **Maintainer 3**: N/A hrs since inbound (threshold: 6.00 hrs) — Breached: No
  - No inbound messages from Maintainer 3

## Ack Actor Coverage

Per-actor wait metrics for monitored acknowledgement actors:

| Actor | Hours Since Inbound | Threshold (hrs) | Inbound Count | Ack | Deadline (UTC) | Breached | Severity | Notes |
|-------|---------------------|-----------------|---------------|-----|----------------|----------|----------|-------|
| Maintainer 2 | 4.36 | 2.00 | 1 | No | 2026-01-23T01:22:58.332668+00:00 | Yes | critical | SLA breach: 4.36 hours since last inbound exceeds 2.00 hour threshold |
| Maintainer 3 | N/A | 6.00 | 0 | No | N/A | No | unknown | No inbound messages from Maintainer 3 |

## Timeline

Messages sorted by timestamp (ascending):

| Timestamp (UTC) | Actor | Direction | Ack | Keywords | File |
|-----------------|-------|-----------|-----|----------|------|
| 2026-01-22T23:22:58.332668+00:00 | Maintainer 2 | Inbound | No | - | `request_dose_experiments_ground_truth_2026-01-22T014445Z.md` |
| 2026-01-23T01:04:13.890440+00:00 | Maintainer 1 | Outbound | No | confirm | `followup_dose_experiments_ground_truth_2026-01-23T011900Z.md` |
| 2026-01-23T02:20:12.676405+00:00 | Maintainer 1 | Outbound | No | confirm | `followup_dose_experiments_ground_truth_2026-01-23T023500Z.md` |
| 2026-01-23T02:21:49.305956+00:00 | Maintainer 1 | Outbound | No | acknowledged, confirm | `response_dose_experiments_ground_truth.md` |

## Matching Files

| File | Match Reason | Actor | Direction | Keywords | Ack | Modified |
|------|--------------|-------|-----------|----------|-----|----------|
| `request_dose_experiments_ground_truth_2026-01-22T014445Z.md` | filename, content | Maintainer 2 | Inbound | - | No | 2026-01-22T23:22:58.332668+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T011900Z.md` | filename, content | Maintainer 1 | Outbound | confirm | No | 2026-01-23T01:04:13.890440+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T023500Z.md` | filename, content | Maintainer 1 | Outbound | confirm | No | 2026-01-23T02:20:12.676405+00:00 |
| `response_dose_experiments_ground_truth.md` | filename, content | Maintainer 1 | Outbound | acknowledged, confirm | No | 2026-01-23T02:21:49.305956+00:00 |

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

### followup_dose_experiments_ground_truth_2026-01-23T023500Z.md

```
# Follow-up: Acknowledgement Request for dose_experiments_ground_truth

**From:** Maintainer <1> (PtychoPINN dose_experiments branch, root_dir: ~/Documents/PtychoPINN/)
**To:** Maintainer <2> (PtychoPINN active branch, root_dir: ~/Documents/tmp/PtychoPINN/)
**Re:** Follow-up — legacy dose_experiments ground-truth artif...
```

### response_dose_experiments_ground_truth.md

```
# Response — Legacy dose_experiments Ground-Truth Bundle

**From:** Maintainer <1> (PtychoPINN dose_experiments branch, root_dir: ~/Documents/PtychoPINN/)
**To:** Maintainer <2> (PtychoPINN active branch, root_dir: ~/Documents/tmp/PtychoPINN/)
**Re:** Request — legacy dose_experiments ground-truth artifacts (2026-01-22...
```
