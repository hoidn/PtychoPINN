### Turn Summary
Implemented `--status-snippet` flag for check_inbox_for_ack.py CLI that writes a reusable Markdown snapshot of wait state (ack status, hours since inbound/outbound, SLA breach, timeline).
Added `test_status_snippet_emits_wait_summary` test validating snippet content and idempotency; all 4 mapped tests pass.
Next: await Maintainer <2> acknowledgement or refresh the status snippet in future scans.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/ (status_snippet.md, pytest logs, inbox_sla_watch/)
