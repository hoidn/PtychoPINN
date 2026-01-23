### Turn Summary
Implemented per-actor wait metrics (`ack_actor_stats`) in the inbox acknowledgement CLI so Maintainer <2>/<3> coverage is visible in JSON/Markdown outputs.
Extended scan_inbox(), write_markdown_summary(), write_status_snippet(), write_escalation_note(), and CLI stdout to emit per-actor tables showing hours since inbound, inbound counts, and ack status for each monitored actor.
Added test_ack_actor_wait_metrics_cover_each_actor regression; all 12 tests pass.
Next: await Maintainer <2> acknowledgement; CLI now tracks both M2 and M3 independently.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T031500Z/ (inbox_scan_summary.json, pytest logs)
