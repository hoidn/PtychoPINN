Summary: Add SLA breach deadline/severity outputs to the inbox acknowledgement CLI and capture a Maintainer <3> escalation drop while we wait for ack.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement of the delivered bundle
Branch: dose_experiments
Mapped tests: pytest tests/tools/test_check_inbox_for_ack_cli.py -q; pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T040500Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.F1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::scan_inbox + write_markdown_summary + write_status_snippet + write_escalation_note + write_history_dashboard + append_history_jsonl + append_history_markdown — compute `sla_deadline_utc`, `breach_duration_hours`, and a severity label whenever `--sla-hours` is provided, surface those fields across JSON/Markdown/stdout/history outputs, and keep ack-actor coverage plus SLA breach exit semantics backward compatible.
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_reports_deadline_and_severity — add a regression that fabricates dated Maintainer <2>/<3> messages, runs the CLI with a tight SLA threshold, and asserts the JSON/Markdown summaries expose the new deadline/breach/severity fields plus severity resets to `ok` once the threshold exceeds the wait time.
- Update: docs/TESTING_GUIDE.md::Inbox acknowledgement CLI + docs/development/TEST_SUITE_INDEX.md::Inbox acknowledgement entry + docs/fix_plan.md (Attempts) + inbox/response_dose_experiments_ground_truth.md + inbox/followup_dose_experiments_ground_truth_2026-01-23T040500Z.md — document the new selector/log paths, record the SLA severity/per-actor evidence (including the Maintainer <3> escalation), and log the ongoing wait-state narrative.
- Capture: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T040500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch}; python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py --inbox inbox --request-pattern dose_experiments_ground_truth --keywords acknowledged --keywords confirm --keywords received --keywords thanks --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" --sla-hours 2.0 --fail-when-breached --history-jsonl "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.jsonl" --history-markdown "$ARTIFACT_ROOT/inbox_history/inbox_sla_watch.md" --history-dashboard "$ARTIFACT_ROOT/inbox_history/inbox_history_dashboard.md" --status-snippet "$ARTIFACT_ROOT/inbox_status/status_snippet.md" --escalation-note "$ARTIFACT_ROOT/inbox_status/escalation_note.md" --escalation-recipient "Maintainer <3>" --output "$ARTIFACT_ROOT/inbox_sla_watch" | tee "$ARTIFACT_ROOT/logs/check_inbox.log" (exit 2 expected when the SLA breach persists).
- Validate: pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log" && pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log".

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && export ARTIFACT_ROOT=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T040500Z && mkdir -p "$ARTIFACT_ROOT"/{logs,inbox_history,inbox_status,inbox_sla_watch} so every artifact lands under the new timestamp.
2. Update `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py`: import `timedelta`, teach `scan_inbox()` to derive `sla_deadline_utc` (last inbound + sla_hours), `breach_duration_hours` (max(hours_since - threshold, 0)), and a severity string (`ok` when not breached, `warning` for <1 hour late, `critical` for >=1 hour). Thread those fields through `results['sla_watch']`, CLI stdout, `write_markdown_summary()`, `write_status_snippet()`, `write_escalation_note()`, `append_history_jsonl()`, `append_history_markdown()` (e.g., annotate the SLA column with severity), and `write_history_dashboard()` (add severity + breach-duration rows/columns). Keep JSON serialization (strings/floats only) and guard for missing inbound timestamps.
3. Extend `tests/tools/test_check_inbox_for_ack_cli.py` with `test_sla_watch_reports_deadline_and_severity`: fabricate one Maintainer <2> message 3.5h old plus one Maintainer <3> message 1h old, run the CLI twice (threshold 2.0h vs 10.0h), assert the JSON `sla_watch` block contains `deadline_utc`, `breach_duration_hours`, `severity`, and that the Markdown summary includes the new “Severity”/“Deadline” lines; confirm severity flips back to `ok` when no breach.
4. Run `pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee "$ARTIFACT_ROOT/logs/pytest_check_inbox_suite.log"` and `pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_ROOT/logs/pytest_loader.log"`; treat non-zero exits (aside from the CLI run below) as blockers.
5. Execute the CLI command under Capture (with Maintainer <2>/<3> ack actors and `--escalation-recipient "Maintainer <3>"`); keep all generated JSON/MD/snippets/history files plus stdout/stderr log under `$ARTIFACT_ROOT`, and accept exit code 2 because `--fail-when-breached` is intentional.
6. Update `inbox/response_dose_experiments_ground_truth.md` with a new “Status as of 2026-01-23T040500Z” subsection that cites the latest SLA severity, deadline, breach age, and per-actor metrics (Maintainer <2> vs <3> rows). Draft `inbox/followup_dose_experiments_ground_truth_2026-01-23T040500Z.md` addressed to Maintainer <3> summarizing the SLA breach, per-actor coverage, and explicit ask for acknowledgement; link to the status snippet/escalation note/history dashboard paths. Capture the new attempt in `docs/fix_plan.md` (Attempts History) noting the severity instrumentation + Maintainer <3> escalation and reference `$ARTIFACT_ROOT`. Refresh `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the new selector plus suite/collect log paths, then drop the Turn Summary block into `$ARTIFACT_ROOT/summary.md` once everything is staged.

Pitfalls To Avoid
- Keep the default ack actor list unchanged (Maintainer <2>) when `--ack-actor` is omitted so older scripts still work.
- Handle missing inbound timestamps gracefully (deadline/severity should read `None`/`unknown`, not raise parsing errors).
- Preserve timezone info when computing deadlines; never emit naive datetime objects in JSON.
- History Markdown already exists—append new rows with severity text but do not rewrite the header or older entries.
- Treat only the CLI run with `--fail-when-breached` as an acceptable exit 2; all other commands must exit 0.
- Use ASCII tables for the new Markdown rows; avoid fancy Unicode or tabs.
- Don’t move or edit production modules outside `plans/active/DEBUG-SIM-LINES-DOSE-001`, `tests/tools/`, `docs/`, and `inbox/`.
- Ensure new severity fields are documented consistently across JSON, Markdown, docs, and follow-up notes to prevent drift.

If Blocked
- Capture the failing command/output with `tee "$ARTIFACT_ROOT/logs/blocker.log"`, leave the working tree untouched, add a short “Blocked” note to `docs/fix_plan.md` Attempts + `galph_memory.md`, and stop so we can reassess or request maintainer input next loop.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:745 — DEBUG-SIM-LINES-DOSE-001.F1 requirements and latest next-actions.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:360 — `scan_inbox` SLA watch logic to extend with deadline/severity.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:820 — `write_status_snippet` waiting-clock/SLA section that must show the new fields.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:980 — `write_escalation_note` SLA + action item text to update with severity data.
- tests/tools/test_check_inbox_for_ack_cli.py:700 — Current inbox CLI regressions (history dashboard + ack actor coverage) to mirror when adding the new severity test.
- docs/TESTING_GUIDE.md:19 and docs/development/TEST_SUITE_INDEX.md:12 — Inbox acknowledgement CLI selector listings that need the new regression + log paths.
- inbox/response_dose_experiments_ground_truth.md:200 — Maintainer status sections to augment with the latest severity/per-actor evidence.

Next Up (optional)
1. If Maintainer <3> stays silent as well, script a cron-friendly wrapper that reruns the CLI hourly and drops severity deltas into `inbox_history/` automatically.
2. Once acknowledgement lands, draft the closure update in `docs/fix_plan.md` and archive excess reports to keep the fix plan lean.

Doc Sync Plan (Conditional)
- After the pytest suite succeeds, run `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_reports_deadline_and_severity -q | tee "$ARTIFACT_ROOT/logs/pytest_sla_severity_collect.log"`.
- Update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` to list the new selector plus both the suite log (`pytest_check_inbox_suite.log`) and collect-only log (`pytest_sla_severity_collect.log`).

Mapped Tests Guardrail
- Ensure `tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_reports_deadline_and_severity` collects (>0) during the collect-only run; treat collection failures as blockers before finishing the loop.

Normative Math/Physics
- Not applicable — maintainer-monitoring tooling only (no physics/math spec changes involved).
