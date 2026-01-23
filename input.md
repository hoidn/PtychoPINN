Summary: Persist the ack-actor follow-up stats into the inbox history JSONL/Markdown/dashboard so Maintainer <3> can audit outbound cadence without opening the latest snippet.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement
Branch: dose_experiments
Mapped tests:
- `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist -q`
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist -q`
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q`
- `pytest tests/test_generic_loader.py::test_generic_loader -q`
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/

Do Now (hard validity contract):
- Checklist IDs: F1 (Maintainer acknowledgement follow-up)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::append_history_jsonl (carry ack_actor_followups into JSONL/Markdown + dashboard helpers)
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist
- Test: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist -q`
- Artifact target: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/`

How-To Map:
1. Prep dirs: `mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/{logs,inbox_sla_watch,inbox_status,inbox_history}`.
2. In `scan_inbox()` add `actor_label` to each `ack_actor_stats` entry, keep outbound stats in place, and make the data available to history writers (e.g., stash a lightweight `ack_actor_followups` dict on `results`).
3. Extend `append_history_jsonl()` to serialize the per-actor follow-up stats (label, `last_outbound_utc`, `hours_since_last_outbound`, `outbound_count`) alongside the existing severity summary.
4. Add `_format_ack_actor_followup_cell()` and update `append_history_markdown()` so the history table gains an “Ack Actor Follow-Ups” column that lists each actor’s last outbound time/age/count; sanitize via the existing helper.
5. Introduce `_build_actor_followup_trends_section(entries)` and call it from `write_history_dashboard()` after the severity trends block. The table should report, per actor, the latest outbound UTC, hours since outbound (from the most recent scan), max outbound count observed, and how many scans reported outbound traffic. Mention the section in the dashboard docstring.
6. Implement `tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist`:
   - Build a synthetic inbox (same pattern as earlier tests) with Maintainer <2> inbound + two outbound follow-ups (one to <2>, one to <3>).
   - Run the CLI with `--history-jsonl`, `--history-markdown`, and `--history-dashboard` pointing at tmp paths plus the usual `--ack-actor` / SLA flags.
   - Assert the JSONL line contains `ack_actor_followups.maintainer_2.outbound_count == 2` (and Maintainer 3 == 1), the Markdown history file contains the new column header + Maintainer labels, and the dashboard file includes the “## Ack Actor Follow-Up Trends” section with both actors.
7. Testing (log everything under the new artifact root):
   - `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/logs/pytest_history_followups_collect.log`
   - `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/logs/pytest_history_followups.log`
   - `pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/logs/pytest_check_inbox_suite.log`
   - `pytest tests/test_generic_loader.py::test_generic_loader -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/logs/pytest_loader.log`
8. CLI evidence run (after tests pass):
```
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --keywords acknowledged --keywords confirm --keywords received --keywords thanks \
  --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" \
  --sla-hours 2.5 \
  --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" \
  --fail-when-breached \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_sla_watch \
  --history-jsonl plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_history/inbox_sla_watch.jsonl \
  --history-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_history/inbox_sla_watch.md \
  --history-dashboard plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_history/inbox_history_dashboard.md \
  --status-snippet plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_status/status_snippet.md \
  --escalation-note plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_status/escalation_note.md \
  --escalation-recipient "Maintainer <2>" \
  --escalation-brief plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_status/escalation_brief_maintainer3.md \
  --escalation-brief-recipient "Maintainer <3>" \
  --escalation-brief-target "Maintainer <2>"
```
   - Verify the snippet, escalation note, and escalation brief reference the persistent follow-up metrics and cite the new artifacts path.
9. Docs + maintainer comms:
   - `docs/TESTING_GUIDE.md`: add a subsection for “Inbox Acknowledgement CLI (History Follow-Up Trends)” documenting the new selector + log paths and mentioning the dashboard section.
   - `docs/development/TEST_SUITE_INDEX.md`: add the selector + artifact log entry under the inbox CLI list.
   - `docs/fix_plan.md`: append the execution details (tests/metrics/artifacts) under the new 2026-01-23T14:35Z attempt.
   - `inbox/response_dose_experiments_ground_truth.md`: add a 2026-01-23T14:35Z block quoting the updated history JSONL/dashboard evidence and link to the new artifact directory.
   - Draft `inbox/followup_dose_experiments_ground_truth_2026-01-23T143500Z.md` citing the persistent follow-up table and hours-since-outbound metrics for Maintainers <2>/<3>.

Pitfalls To Avoid:
1. Keep all code confined to `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/`; never touch shipped modules under `ptycho*/`.
2. Preserve backward compatibility for existing JSON consumers—additive fields only, null-safe defaults when no outbound data exists.
3. Sanitize new Markdown content so table pipes/newlines don’t break the history files.
4. Don’t double-count outbound messages; only treat Maintainer <1> messages targeting the actor (To/CC) as follow-ups.
5. Avoid mutating the real `inbox/` during tests; rely on tmp fixtures.
6. Leave SLA breach logic untouched; follow-up metrics supplement the story but must not flip severity states.
7. Ensure helper functions guard against missing history data (empty JSONL should still emit the “no data” message).
8. Keep timestamps consistent (UTC ISO strings) so lexicographic comparisons remain valid within the dashboard helper.
9. Do not relax `--fail-when-breached`; exit code 2 should persist on SLA violations.
10. Capture every pytest/CLI log inside the new artifact root for traceability.

If Blocked:
- Capture the failure signature plus command in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/logs/blocked.log`, append a note to `docs/fix_plan.md` Attempts History summarizing the blocker, and ping Galph before retrying. Leave the user’s dirty working tree untouched.

Findings Applied (Mandatory): No relevant findings in the knowledge base.

Pointers:
- `docs/fix_plan.md:1288` — describes the scoped gap and next actions for this loop.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:512` — `scan_inbox()` block where per-actor stats are computed and must feed the new history fields.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:995` — `append_history_jsonl`/`append_history_markdown` helpers targeted by this change.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:2119` — `write_history_dashboard` currently lacking follow-up trends.
- `tests/tools/test_check_inbox_for_ack_cli.py:2252` — location of the existing follow-up test to mimic when adding `test_history_followups_persist`.

Next Up (optional): Once the history persistence lands, rerun the CLI periodically until Maintainer <2> (or Maintainer <3>) replies so we can close F1 with an acknowledgement record.

Doc Sync Plan (Conditional): After code passes, rerun `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/logs/pytest_history_followups_collect.log`, then update `docs/TESTING_GUIDE.md` §Inbox CLI and `docs/development/TEST_SUITE_INDEX.md` with the selector/log references before finishing.
