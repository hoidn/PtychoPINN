### Turn Summary
Implemented SLA deadline, breach duration, and severity fields in the inbox acknowledgement CLI so maintainers can track exactly when SLA was due and how late the breach is.
Resolved by computing `deadline_utc` (last inbound + threshold), `breach_duration_hours`, and `severity` (ok/warning/critical/unknown) and surfacing them across JSON, Markdown, status snippet, escalation note, history outputs, and CLI stdout.
Added regression test `test_sla_watch_reports_deadline_and_severity` validating the new fields and severity flip behavior.
Next: await Maintainer <2> acknowledgement; current breach is 3.55 hours with critical severity.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T040500Z/ (inbox_scan_summary.json, pytest_check_inbox_suite.log)
