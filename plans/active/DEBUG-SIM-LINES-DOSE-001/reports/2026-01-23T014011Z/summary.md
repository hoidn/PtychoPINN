### Turn Summary
Implemented `--history-jsonl` and `--history-markdown` flags for the inbox acknowledgement CLI to create a persistent timeline of scan results.
Added `append_history_jsonl()` and `append_history_markdown()` helpers that append entries without duplicating headers.
Created `test_history_logging_appends_entries` test validating the 2-run ack-flip scenario; all tests pass.
Next: Maintainer <2> acknowledgement still pending; history log captures 2.38 hour SLA breach for escalation evidence.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014011Z/ (inbox_history/, inbox_sla_watch/, logs/)
