### Turn Summary
Extended the inbox acknowledgement CLI to persist per-actor SLA severity classifications in history logs (JSONL gains `ack_actor_summary` field, Markdown gains "Ack Actor Severity" column).
Added `test_ack_actor_history_tracks_severity` regression test validating that Maintainer <2> appears in `critical` bucket and Maintainer <3> in `unknown` bucket; all 17 tests pass.
Next: await Maintainer <2> acknowledgement of the delivered bundle; per-actor severity is now tracked historically to prove breach duration over time.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/ (pytest_ack_actor_history.log, inbox_sla_watch.jsonl)
