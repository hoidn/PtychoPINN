### Turn Summary (2026-01-23T04:51Z — Ralph)
Implemented per-actor follow-up (outbound) activity tracking in the inbox acknowledgement CLI to prove how often Maintainer <1> pings each monitored actor.
Added `parse_target_actors()` to extract recipients from To:/CC: lines, extended `scan_inbox()` with outbound stats, and added "Ack Actor Follow-Up Activity" table to all Markdown outputs.
All 21 tests pass including the new `test_ack_actor_followups_track_outbound_targets` regression test.
Next: Close F1 once Maintainer <2> or <3> acknowledges by rerunning the CLI to capture the ack event.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/ (pytest_followups.log, status_snippet.md, escalation_brief_maintainer3.md)

### Turn Summary (2026-01-23T133500Z — Galph)
Scoped the Maintainer follow-up gap by reviewing the latest F1 evidence and updating docs/fix_plan so we target per-actor outbound metrics next.
Wrote a detailed Do Now (input.md) covering recipient parsing, follow-up tables, regression test, doc/test index updates, and the refreshed maintainer artifacts.
Staged the 2026-01-23T133500Z reports root so Ralph's upcoming CLI + pytest runs have a home for logs and outputs.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/ (input.md)
