### Turn Summary
Implemented `--escalation-brief` CLI feature with three new flags to generate a Markdown brief for third-party escalation.
Updated `_build_actor_breach_timeline_section` to return breach state data alongside Markdown lines.
Added `write_escalation_brief` function producing Blocking Actor Snapshot, Breach Streak Summary, Action Items, and Proposed Message sections.
All 20 tests pass; new test `test_escalation_brief_targets_blocker` validates the brief format.
Next: monitor for acknowledgement from Maintainer <2> or <3>.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/ (escalation_brief_maintainer3.md, pytest_escalation_brief.log)
