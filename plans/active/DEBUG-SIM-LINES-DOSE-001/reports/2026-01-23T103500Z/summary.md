### Turn Summary
Implemented `_build_actor_breach_timeline_section()` in `check_inbox_for_ack.py` to add an "Ack Actor Breach Timeline" section to the history dashboard showing per-actor breach start timestamps, consecutive streak counts, and hours past SLA.
The test validates Maintainer 2 appears in the breach timeline with streak=2, CRITICAL severity, and numeric hours-past-SLA; Maintainer 3 is correctly excluded as unknown status.
Next: await Maintainer <2> acknowledgement; escalation note and breach timeline now provide actionable context for Maintainer <3>.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/ (inbox_history_dashboard.md, pytest_history_dashboard_actor_breach.log)
