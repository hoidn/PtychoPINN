### Turn Summary
Implemented `--history-dashboard` CLI flag that reads JSONL history and emits an aggregated Markdown dashboard with total scans, ack count, breach count, longest wait, and recent scans timeline.
Resolved the missing SLA tracking aggregation by adding `write_history_dashboard()` helper with proper edge case handling for empty/missing JSONL files; tests pass (9/9).
Next: await Maintainer <2> acknowledgement; if no reply, dashboard data can support escalation to Maintainer <3>.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/ (inbox_history_dashboard.md, pytest_history_dashboard.log)
