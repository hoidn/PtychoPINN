Summary: Add a breach timeline view to the inbox history dashboard plus docs/inbox updates so Maintainer <3> can see when Maintainer <2> crossed SLA and how long the current streak lasts.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline -q; pytest tests/tools/test_check_inbox_for_ack_cli.py -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::write_history_dashboard — add an `_build_actor_breach_timeline_section()` helper that scans JSONL history, tracks per-actor breach start timestamps/latest scans/current streak counts/hours past SLA for warning+critical actors, and renders a sanitized "## Ack Actor Breach Timeline" table (sorted by severity priority, graceful fallback when history lacks per-actor data).
- Implement (test): tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline — run the CLI twice with history logging/dashboard enabled (Maintainer <2> ~3.5h overdue, Maintainer <3> still unknown) and assert the dashboard emits the new section with Maintainer 2 row showing `Current Streak = 2`, non-empty breach start/latest scan timestamps, and hours-past-SLA text while Maintainer 3 stays absent.
- Update: docs/TESTING_GUIDE.md (§Inbox acknowledgement CLI) and docs/development/TEST_SUITE_INDEX.md with the new selector/log; append the 2026-01-23T103500Z breach timeline status block to inbox/response_dose_experiments_ground_truth.md; draft inbox/followup_dose_experiments_ground_truth_2026-01-23T103500Z.md summarizing the new evidence; record this attempt in docs/fix_plan.md Attempts History.
- Capture: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch} && python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.5 --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --history-dashboard "$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --escalation-recipient "Maintainer <2>" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/logs/check_inbox.log" (exit code 2 expected until Maintainer <2> replies).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline -q | tee "$ARTIFACT_ROOT/logs/pytest_history_dashboard_actor_breach.log"; pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline -q | tee "$ARTIFACT_ROOT/logs/pytest_history_dashboard_actor_breach_collect.log"; pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"; pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".
- Artifacts: Collect JSON/Markdown summaries, dashboard/status/escalation outputs, pytest logs, docs/inbox diffs, and copy this turn summary into "$ARTIFACT_ROOT/summary.md".

How-To Map
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}` to stage the artifact root referenced by every command.
2. Extend `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::write_history_dashboard` by adding `_build_actor_breach_timeline_section()` (reuse JSONL parsing already in `_build_actor_severity_trends_section`); iterate entries chronologically, track per-actor streak state (when severity enters warning/critical, set or increment streak, record first breach timestamp, update hours-past-SLA via `max(hours_since - threshold, 0)`), and append a Markdown table (Actor | Breach Start | Latest Scan | Current Streak | Hours Past SLA | Severity) plus a fallback paragraph when no active breaches exist.
3. Add `tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline` mirroring the severity-trends test: create a tmp inbox with Maintainer <2> inbound ~3.5h ago (no ack keywords) and no Maintainer <3> inbound, run the CLI twice with all history/dashboard flags + ack-actor overrides, then assert the dashboard text includes the new section, Maintainer 2 row with `Current Streak = 2`, non-empty timestamps, and a numeric hours-past-SLA string; also assert Maintainer 3 is not listed in the breach table.
4. Execute the mapped pytest selectors (targeted test, collect-only, full CLI suite, loader guard) piping logs into `$ARTIFACT_ROOT/logs/` as specified; do not skip the collect-only run (halt if it fails).
5. Run the real inbox capture command so `$ARTIFACT_ROOT` contains fresh `inbox_sla_watch/`, `inbox_history/`, `inbox_status/`, and `logs/check_inbox.log` reflecting the new breach timeline view (exit 2 is expected because the SLA is breached).
6. Update docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md with the new selector/log reference, append the 2026-01-23T103500Z status block to inbox/response_dose_experiments_ground_truth.md, draft inbox/followup_dose_experiments_ground_truth_2026-01-23T103500Z.md citing the breach timeline, and record the attempt in docs/fix_plan.md; then copy this Turn Summary into `$ARTIFACT_ROOT/summary.md`.

Pitfalls To Avoid
- Do not mutate or truncate existing history JSONL/Markdown files; only append and emit new dashboard snapshots.
- Keep severity ordering stable (critical > warning > ok > unknown) in both trends and breach timeline tables to avoid nondeterministic tests.
- Sanitize Markdown strings (pipes/newlines) before inserting actor labels, timestamps, or notes to keep tables valid.
- Ensure `_build_actor_breach_timeline_section` gracefully reports "No active breaches" when every actor is OK/unknown or when history lacks per-actor data.
- Maintain `--fail-when-breached` semantics; do not change exit codes even if the new table is empty.
- Avoid hard-coding Maintainer names anywhere except in docs copy; CLI must continue to rely on --ack-actor inputs and normalized IDs.
- Keep the per-actor streak counter reset when severity returns to OK/unknown so stale breaches do not linger.
- Capture every CLI/pytest invocation with `tee` into `$ARTIFACT_ROOT/logs/` to preserve evidence.

If Blocked
- Stop immediately if the new selector fails to collect or the CLI command errors; save the stderr to `$ARTIFACT_ROOT/logs/blocker.log`, note the issue in docs/fix_plan.md Attempts History + galph_memory.md, and wait for supervisor guidance instead of guessing.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:572 — Defines the per-actor breach timeline scope and deliverables for DEBUG-SIM-LINES-DOSE-001.F1.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:1505 — Current history dashboard helpers where the breach timeline section must be added.
- tests/tools/test_check_inbox_for_ack_cli.py:1721 — Existing history dashboard regression tests to mirror for the new breach timeline test.
- docs/TESTING_GUIDE.md:60 — Inbox acknowledgement CLI sections that require the new selector/log entry.
- inbox/response_dose_experiments_ground_truth.md:200 — Maintainer status log where the 2026-01-23T103500Z breach timeline summary must be appended.

Next Up (optional)
1. If Maintainer <2> still fails to respond after this drop, prepare an escalation targeting Maintainer <3> that quotes the breach timeline table.
2. Explore generating an HTML dashboard once Maintainer <2> acknowledges, so follow-ups become passive monitoring.

Doc Sync Plan (Conditional)
- After code/tests pass, rerun `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline -q | tee "$ARTIFACT_ROOT/logs/pytest_history_dashboard_actor_breach_collect.log"` (already listed) and update docs/TESTING_GUIDE.md plus docs/development/TEST_SUITE_INDEX.md with the selector/log paths before finishing.

Mapped Tests Guardrail
- Treat `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline -q` as mandatory; if it collects 0 tests or errors, halt and report.

Hard Gate
- Do not mark DEBUG-SIM-LINES-DOSE-001.F1 attempts complete unless `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline -q` collects >0, passes, and the CLI capture writes the breach timeline table into `$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md`.

Normative Math/Physics
- Not applicable; rely on docs/TESTING_GUIDE.md for authoritative CLI behavior (no physics specs touched).
