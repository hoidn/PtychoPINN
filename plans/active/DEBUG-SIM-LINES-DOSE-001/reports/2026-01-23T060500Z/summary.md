### Turn Summary
Implemented `--ack-actor-sla` repeatable CLI flag to allow per-actor SLA threshold overrides in `check_inbox_for_ack.py`.
Each actor's breach status now uses their actor-specific threshold (falling back to global `--sla-hours`), and JSON/Markdown outputs show the effective threshold per actor.
Next: Await Maintainer <2> acknowledgement; per-actor thresholds ready for differentiated SLA monitoring.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T060500Z/ (pytest_sla_override_collect.log)
