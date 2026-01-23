# Escalation Note

**Generated:** 2026-01-23T05:17:13.054672+00:00
**Recipient:** Maintainer <2>
**Request Pattern:** `dose_experiments_ground_truth`

## Status: SLA Breach - Escalation Required

## Summary Metrics

| Metric | Value |
|--------|-------|
| Ack Detected | No |
| Hours Since Last Inbound | 5.90 |
| Hours Since Last Outbound | 0.19 |
| Total Inbound Messages | 1 |
| Total Outbound Messages | 8 |

## SLA Watch

| Metric | Value |
|--------|-------|
| Threshold | 2.50 hours |
| Deadline (UTC) | 2026-01-23T01:52:58.332668+00:00 |
| Breached | Yes |
| Breach Duration | 3.40 hours |
| Severity | critical |

**Notes:** SLA breach: 5.90 hours since last inbound exceeds 2.50 hour threshold and no acknowledgement detected

> **SLA BREACH:** Immediate attention required.

## Action Items

1. Send follow-up message to Maintainer <2> requesting acknowledgement
2. Reference the `dose_experiments_ground_truth` request pattern
3. Cite SLA breach: 5.90 hours since last inbound exceeds threshold
4. Request explicit confirmation or next steps

## Proposed Message

The following blockquote can be used as a starting point for the follow-up:

> **Follow-up: Acknowledgement Request**
>
> Hello Maintainer <2>,
>
> This is a follow-up regarding the `dose_experiments_ground_truth` request.
> It has been approximately **5.90 hours** since the last inbound message,
> which exceeds our SLA threshold.
>
> Could you please confirm receipt of the delivered artifacts and provide any feedback?
>
> If there are any issues or blockers, please let us know so we can assist.
>
> Thank you,
> Maintainer <1>

## Ack Actor SLA Summary

### Critical (Breach >= 1h)

- **Maintainer 2**: 5.90 hrs (threshold: 2.00 hrs) — Breached: Yes

### Unknown (No Inbound)

- **Maintainer 3**: N/A hrs (threshold: 6.00 hrs) — Breached: No

## Ack Actor Coverage

| Actor | Hrs Since Inbound | Threshold (hrs) | Deadline (UTC) | Breached | Severity | Notes |
|-------|-------------------|-----------------|----------------|----------|----------|-------|
| Maintainer 2 | 5.90 | 2.00 | 2026-01-23T01:22:58.332668+00:00 | Yes | critical | SLA breach: 5.90 hours since last inbound exceeds 2.00 hour threshold |
| Maintainer 3 | N/A | 6.00 | N/A | No | unknown | No inbound messages from Maintainer 3 |

## Ack Actor Follow-Up Activity

Follow-up messages from Maintainer <1> to each monitored actor:

| Actor | Last Outbound (UTC) | Hours Since Outbound | Outbound Count |
|-------|---------------------|----------------------|----------------|
| Maintainer 2 | 2026-01-23T05:05:35.670075+00:00 | 0.19 | 8 |
| Maintainer 3 | 2026-01-23T04:52:58.651755+00:00 | 0.40 | 3 |

## Timeline

| Timestamp | Actor | Direction | Ack |
|-----------|-------|-----------|-----|
| 2026-01-22T23:22:58 | Maintainer 2 | Inbound | No |
| 2026-01-23T01:04:13 | Maintainer 1 | Outbound | No |
| 2026-01-23T02:20:12 | Maintainer 1 | Outbound | No |
| 2026-01-23T03:46:27 | Maintainer 1 | Outbound | No |
| 2026-01-23T03:58:50 | Maintainer 1 | Outbound | No |
| 2026-01-23T04:24:13 | Maintainer 1 | Outbound | No |
| 2026-01-23T04:38:25 | Maintainer 1 | Outbound | No |
| 2026-01-23T04:52:58 | Maintainer 1 | Outbound | No |
| 2026-01-23T05:05:35 | Maintainer 1 | Outbound | No |

## Ack Actor Breach Timeline

**Active breaches:** 1 actor(s)

| Actor | Breach Start | Latest Scan | Current Streak | Hours Past SLA | Severity |
|-------|--------------|-------------|----------------|----------------|----------|
| Maintainer 2 | 2026-01-23T05:17:13 | 2026-01-23T05:17:13 | 1 | 3.90h | CRITICAL |
