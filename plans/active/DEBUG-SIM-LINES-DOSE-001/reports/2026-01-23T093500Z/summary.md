### Turn Summary
Implemented per-actor severity trends aggregation in the history dashboard, adding a new "## Ack Actor Severity Trends" table that shows severity counts (critical/warning/ok/unknown), longest wait, and latest scan timestamp per actor.
Added `test_history_dashboard_actor_severity_trends` test validating the new section (18 tests now pass in the inbox CLI suite).
Next: Continue awaiting Maintainer <2> acknowledgement; the dashboard now proves sustained SLA breach over time.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/ (inbox_history_dashboard.md, pytest_check_inbox_suite.log)
