### Turn Summary
Extended the inbox acknowledgement CLI with per-actor SLA fields (sla_deadline_utc, sla_breached, sla_breach_duration_hours, sla_severity, sla_notes) so each configured ack actor gets its own deadline and severity tracking.
Added test_ack_actor_sla_metrics_include_deadline regression test and updated Markdown outputs to show expanded Ack Actor Coverage table with Deadline/Breached/Severity/Notes columns when --sla-hours is used.
Next: Continue awaiting Maintainer <2> acknowledgement; per-actor SLA metrics now visible across all outputs.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T050500Z/ (inbox_scan_summary.json, status_snippet.md, escalation_note.md)
