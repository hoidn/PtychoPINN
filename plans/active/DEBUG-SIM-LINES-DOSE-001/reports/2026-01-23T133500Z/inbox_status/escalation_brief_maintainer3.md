# Escalation Brief

**Generated:** 2026-01-23T04:51:27.865512+00:00
**Recipient:** Maintainer <3>
**Blocking Actor:** Maintainer <2>
**Request Pattern:** `dose_experiments_ground_truth`

## Blocking Actor Snapshot

**Actor:** Maintainer 2

| Metric | Value |
|--------|-------|
| Hours Since Inbound | 5.47 |
| SLA Threshold | 2.00 hours |
| Deadline (UTC) | 2026-01-23T01:22:58.332668+00:00 |
| Hours Past SLA | 3.47 |
| Severity | CRITICAL |
| Ack Files | None |
| Last Outbound (UTC) | 2026-01-23T04:38:25.045297+00:00 |
| Hours Since Outbound | 0.22 |
| Outbound Count | 7 |

> **SLA BREACH (CRITICAL):** Maintainer 2 has exceeded the SLA threshold.

## Breach Streak Summary

| Metric | Value |
|--------|-------|
| Current Streak | 1 consecutive scan(s) |
| Breach Start | 2026-01-23T04:51:27 |
| Latest Scan | 2026-01-23T04:51:27 |
| Peak Hours Past SLA | 3.47 |

## Action Items

1. Review the SLA breach evidence for Maintainer 2
2. Contact Maintainer <3> to escalate the blocking issue
3. Reference request pattern: `dose_experiments_ground_truth`
4. Request acknowledgement or status update from Maintainer 2

## Proposed Message

The following can be used when contacting Maintainer <3>:

> **Escalation: SLA Breach on Request**
>
> Hello Maintainer <3>,
>
> I am escalating an SLA breach regarding the `dose_experiments_ground_truth` request.
> Maintainer 2 has not acknowledged receipt, and it has been **5.47 hours**
> since the last inbound message from them, exceeding our SLA threshold.
>
> Could you please assist in obtaining a response or status update from Maintainer 2?
>
> If there are any blockers or issues on their end, please let us know so we can adjust our plans.
>
> Thank you,
> Maintainer <1>

## Ack Actor Follow-Up Activity

Follow-up messages from Maintainer <1> to each monitored actor:

| Actor | Last Outbound (UTC) | Hours Since Outbound | Outbound Count |
|-------|---------------------|----------------------|----------------|
| Maintainer 2 | 2026-01-23T04:38:25.045297+00:00 | 0.22 | 7 |
| Maintainer 3 | 2026-01-23T04:38:25.045297+00:00 | 0.22 | 2 |

## Ack Actor Breach Timeline

**Active breaches:** 1 actor(s)

| Actor | Breach Start | Latest Scan | Current Streak | Hours Past SLA | Severity |
|-------|--------------|-------------|----------------|----------------|----------|
| Maintainer 2 | 2026-01-23T04:51:27 | 2026-01-23T04:51:27 | 1 | 3.47h | CRITICAL |
