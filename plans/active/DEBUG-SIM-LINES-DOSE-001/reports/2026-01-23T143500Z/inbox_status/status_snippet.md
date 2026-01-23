# Maintainer Status Snapshot

**Generated:** 2026-01-23T05:04:02.814928+00:00

## Ack Detected: No

Waiting for acknowledgement from monitored actor(s): Maintainer 2, Maintainer 3.

## Wait Metrics

| Metric | Value |
|--------|-------|
| Hours since last inbound | 5.68 |
| Hours since last outbound | 0.18 |
| Total inbound messages | 1 |
| Total outbound messages | 8 |

## SLA Watch

| Metric | Value |
|--------|-------|
| Threshold | 2.50 hours |
| Deadline (UTC) | 2026-01-23T01:52:58.332668+00:00 |
| Breached | Yes |
| Breach Duration | 3.18 hours |
| Severity | critical |

**Notes:** SLA breach: 5.68 hours since last inbound exceeds 2.50 hour threshold and no acknowledgement detected

> **SLA BREACH:** Action required.

## Ack Actor SLA Summary

### Critical (Breach >= 1h)

- **Maintainer 2**: 5.68 hrs (threshold: 2.00 hrs) — Breached: Yes

### Unknown (No Inbound)

- **Maintainer 3**: N/A hrs (threshold: 6.00 hrs) — Breached: No

## Ack Actor Coverage

| Actor | Hrs Since Inbound | Threshold (hrs) | Deadline (UTC) | Breached | Severity | Notes |
|-------|-------------------|-----------------|----------------|----------|----------|-------|
| Maintainer 2 | 5.68 | 2.00 | 2026-01-23T01:22:58.332668+00:00 | Yes | critical | SLA breach: 5.68 hours since last inbound exceeds 2.00 hour threshold |
| Maintainer 3 | N/A | 6.00 | N/A | No | unknown | No inbound messages from Maintainer 3 |

## Ack Actor Follow-Up Activity

Follow-up messages from Maintainer <1> to each monitored actor:

| Actor | Last Outbound (UTC) | Hours Since Outbound | Outbound Count |
|-------|---------------------|----------------------|----------------|
| Maintainer 2 | 2026-01-23T04:52:58.651755+00:00 | 0.18 | 8 |
| Maintainer 3 | 2026-01-23T04:52:58.651755+00:00 | 0.18 | 3 |

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
| 2026-01-23T04:52:46 | Maintainer 1 | Outbound | No |
| 2026-01-23T04:52:58 | Maintainer 1 | Outbound | No |

## Ack Actor Breach Timeline

**Active breaches:** 1 actor(s)

| Actor | Breach Start | Latest Scan | Current Streak | Hours Past SLA | Severity |
|-------|--------------|-------------|----------------|----------------|----------|
| Maintainer 2 | 2026-01-23T05:04:02 | 2026-01-23T05:04:02 | 1 | 3.68h | CRITICAL |
