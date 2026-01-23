### Turn Summary
Implemented --ack-actor repeatable CLI flag and custom keywords support for check_inbox_for_ack.py; now detects acks from Maintainer <3> when configured.
Extended detect_actor_and_direction() with M3 patterns and updated is_acknowledgement() to honor user keywords exactly (no hidden hard-coded list).
Added 2 tests (11 total passing) and ran CLI with both M2/M3 ack actors; SLA remains breached (3.16h > 2h) with no acknowledgement.
Next: Monitor inbox for Maintainer <2> response or escalate to Maintainer <3> using the new ack-actor monitoring.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T024800Z/ (inbox_scan_summary.json, pytest_check_inbox_suite.log)
