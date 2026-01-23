# Inbox Scan Summary

**Generated:** 2026-01-23T05:17:13.054672+00:00

## Parameters

- **Inbox:** `inbox`
- **Request Pattern:** `dose_experiments_ground_truth`
- **Keywords:** `acknowledged`, `confirm`, `received`, `thanks`
- **Ack Actors:** `maintainer_2`, `maintainer_3`
- **Since Filter:** None

## Summary

- **Files Scanned:** 11
- **Matches Found:** 9
- **Acknowledgement Detected:** No

## Waiting Clock

- **Last Inbound (from Maintainer <2>):** 2026-01-22T23:22:58.332668+00:00
- **Hours Since Last Inbound:** 5.90 hours
- **Last Outbound (from Maintainer <1>):** 2026-01-23T05:05:35.670075+00:00
- **Hours Since Last Outbound:** 0.19 hours
- **Total Inbound Messages:** 1
- **Total Outbound Messages:** 8

> **Note:** No acknowledgement from Maintainer <2> found. The bundle has been delivered
> but we are still awaiting confirmation from the receiving maintainer.

## SLA Watch

- **Threshold:** 2.50 hours
- **Hours Since Last Inbound:** 5.90 hours
- **Deadline (UTC):** 2026-01-23T01:52:58.332668+00:00
- **Breached:** Yes
- **Breach Duration:** 3.40 hours
- **Severity:** critical
- **Notes:** SLA breach: 5.90 hours since last inbound exceeds 2.50 hour threshold and no acknowledgement detected

> **SLA BREACH:** The waiting time has exceeded the configured threshold and no acknowledgement has been received.

## Ack Actor SLA Summary

Per-actor SLA status grouped by severity:

### Critical (Breach >= 1h)

- **Maintainer 2**: 5.90 hrs since inbound (threshold: 2.00 hrs) — Breached: Yes
  - SLA breach: 5.90 hours since last inbound exceeds 2.00 hour threshold

### Unknown (No Inbound)

- **Maintainer 3**: N/A hrs since inbound (threshold: 6.00 hrs) — Breached: No
  - No inbound messages from Maintainer 3

## Ack Actor Coverage

Per-actor wait metrics for monitored acknowledgement actors:

| Actor | Hours Since Inbound | Threshold (hrs) | Inbound Count | Ack | Deadline (UTC) | Breached | Severity | Notes |
|-------|---------------------|-----------------|---------------|-----|----------------|----------|----------|-------|
| Maintainer 2 | 5.90 | 2.00 | 1 | No | 2026-01-23T01:22:58.332668+00:00 | Yes | critical | SLA breach: 5.90 hours since last inbound exceeds 2.00 hour threshold |
| Maintainer 3 | N/A | 6.00 | 0 | No | N/A | No | unknown | No inbound messages from Maintainer 3 |

## Ack Actor Follow-Up Activity

Follow-up messages from Maintainer <1> to each monitored actor:

| Actor | Last Outbound (UTC) | Hours Since Outbound | Outbound Count |
|-------|---------------------|----------------------|----------------|
| Maintainer 2 | 2026-01-23T05:05:35.670075+00:00 | 0.19 | 8 |
| Maintainer 3 | 2026-01-23T04:52:58.651755+00:00 | 0.40 | 3 |

## Timeline

Messages sorted by timestamp (ascending):

| Timestamp (UTC) | Actor | Direction | Ack | Keywords | File |
|-----------------|-------|-----------|-----|----------|------|
| 2026-01-22T23:22:58.332668+00:00 | Maintainer 2 | Inbound | No | - | `request_dose_experiments_ground_truth_2026-01-22T014445Z.md` |
| 2026-01-23T01:04:13.890440+00:00 | Maintainer 1 | Outbound | No | confirm | `followup_dose_experiments_ground_truth_2026-01-23T011900Z.md` |
| 2026-01-23T02:20:12.676405+00:00 | Maintainer 1 | Outbound | No | confirm | `followup_dose_experiments_ground_truth_2026-01-23T023500Z.md` |
| 2026-01-23T03:46:27.473006+00:00 | Maintainer 1 | Outbound | No | confirm | `followup_dose_experiments_ground_truth_2026-01-23T083500Z.md` |
| 2026-01-23T03:58:50.557595+00:00 | Maintainer 1 | Outbound | No | confirm | `followup_dose_experiments_ground_truth_2026-01-23T093500Z.md` |
| 2026-01-23T04:24:13.913697+00:00 | Maintainer 1 | Outbound | No | confirm | `followup_dose_experiments_ground_truth_2026-01-23T113500Z.md` |
| 2026-01-23T04:38:25.045297+00:00 | Maintainer 1 | Outbound | No | acknowledged, confirm, received | `followup_dose_experiments_ground_truth_2026-01-23T123500Z.md` |
| 2026-01-23T04:52:58.651755+00:00 | Maintainer 1 | Outbound | No | confirm | `followup_dose_experiments_ground_truth_2026-01-23T133500Z.md` |
| 2026-01-23T05:05:35.670075+00:00 | Maintainer 1 | Outbound | No | acknowledged, confirm, received, thanks | `response_dose_experiments_ground_truth.md` |

## Matching Files

| File | Match Reason | Actor | Direction | Keywords | Ack | Modified |
|------|--------------|-------|-----------|----------|-----|----------|
| `request_dose_experiments_ground_truth_2026-01-22T014445Z.md` | filename, content | Maintainer 2 | Inbound | - | No | 2026-01-22T23:22:58.332668+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T011900Z.md` | filename, content | Maintainer 1 | Outbound | confirm | No | 2026-01-23T01:04:13.890440+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T023500Z.md` | filename, content | Maintainer 1 | Outbound | confirm | No | 2026-01-23T02:20:12.676405+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T083500Z.md` | filename, content | Maintainer 1 | Outbound | confirm | No | 2026-01-23T03:46:27.473006+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T093500Z.md` | filename, content | Maintainer 1 | Outbound | confirm | No | 2026-01-23T03:58:50.557595+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T113500Z.md` | filename, content | Maintainer 1 | Outbound | confirm | No | 2026-01-23T04:24:13.913697+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T123500Z.md` | filename, content | Maintainer 1 | Outbound | acknowledged, confirm, received | No | 2026-01-23T04:38:25.045297+00:00 |
| `followup_dose_experiments_ground_truth_2026-01-23T133500Z.md` | filename, content | Maintainer 1 | Outbound | confirm | No | 2026-01-23T04:52:58.651755+00:00 |
| `response_dose_experiments_ground_truth.md` | filename, content | Maintainer 1 | Outbound | acknowledged, confirm, received, thanks | No | 2026-01-23T05:05:35.670075+00:00 |

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

### followup_dose_experiments_ground_truth_2026-01-23T083500Z.md

```
# Follow-up: dose_experiments_ground_truth — Per-Actor Severity History

**From:** Maintainer <1>
**To:** Maintainer <2>
**Date:** 2026-01-23T08:35:00Z
**Subject:** SLA breach history now tracked per-actor

---

## Summary

The inbox acknowledgement CLI now persists per-actor SLA severity classifications in the history...
```

### followup_dose_experiments_ground_truth_2026-01-23T093500Z.md

```
# Follow-up: dose_experiments_ground_truth (2026-01-23T093500Z)

**From:** Maintainer <1> (PtychoPINN dose_experiments branch)
**To:** Maintainer <2> (PtychoPINN active branch), Maintainer <3>
**Date:** 2026-01-23T09:35:00Z
**Re:** Per-Actor Severity Trends in History Dashboard

---

## Summary

The history dashboard f...
```

### followup_dose_experiments_ground_truth_2026-01-23T113500Z.md

```
# Follow-up: dose_experiments_ground_truth (2026-01-23T11:35Z)

**From:** Maintainer <1>
**To:** Maintainer <2>
**Date:** 2026-01-23T11:35:00Z

## Summary

This follow-up adds embedded breach timeline visibility to the status snapshot and escalation note outputs. Maintainers <2> and <3> can now review per-actor streak ...
```

### followup_dose_experiments_ground_truth_2026-01-23T123500Z.md

```
# Follow-up: SLA Breach Escalation to Maintainer <3>

**From:** Maintainer <1>
**To:** Maintainer <3> (CC: Maintainer <2>)
**Date:** 2026-01-23T12:35:00Z
**Subject:** Escalation - dose_experiments_ground_truth acknowledgement pending

---

Hello Maintainer <3>,

I am escalating an SLA breach regarding the `dose_experim...
```

### followup_dose_experiments_ground_truth_2026-01-23T133500Z.md

```
# Follow-up: dose_experiments_ground_truth (2026-01-23T133500Z)

**From:** Maintainer <1>
**To:** Maintainer <2>
**CC:** Maintainer <3>
**Date:** 2026-01-23T04:51:27Z
**Re:** dose_experiments_ground_truth

---

## Status Update

This is an automated follow-up regarding the `dose_experiments_ground_truth` request.

### ...
```

### response_dose_experiments_ground_truth.md

```
# Response — Legacy dose_experiments Ground-Truth Bundle

**From:** Maintainer <1> (PtychoPINN dose_experiments branch, root_dir: ~/Documents/PtychoPINN/)
**To:** Maintainer <2> (PtychoPINN active branch, root_dir: ~/Documents/tmp/PtychoPINN/)
**Re:** Request — legacy dose_experiments ground-truth artifacts (2026-01-22...
```
