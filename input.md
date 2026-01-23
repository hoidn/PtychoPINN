Summary: Add per-actor follow-up tracking to the inbox CLI so we can prove how often Maintainer <1> is pinging Maintainers <2>/<3>, then capture the refreshed evidence bundle and maintainer docs.
Focus: DEBUG-SIM-LINES-DOSE-001.F1 — Await Maintainer <2> acknowledgement
Branch: dose_experiments
Mapped tests:
- `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets -q`
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets -q`
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q`
- `pytest tests/test_generic_loader.py::test_generic_loader -q`
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/

Do Now (hard validity contract):
- Checklist IDs: F1 (Maintainer acknowledgement follow-up)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py::scan_inbox (plus downstream writers)
- Implement: tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets
- Test: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets -q`
- Artifact target: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/`

How-To Map:
1. Prep directories: `mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/{logs,inbox_sla_watch,inbox_status,inbox_history}`.
2. Extend `check_inbox_for_ack.py` (keep changes inside `plans/active/.../bin/`):
   - Add a helper that parses `To:` / `CC:` lines (case-insensitive, sanitized) and returns normalized actor IDs using `normalize_actor_alias`. Store the resulting `target_actors` list on each `match_entry` and `timeline` row.
   - In `scan_inbox()`, derive per-actor outbound stats for every configured ack actor by scanning timeline entries where `actor == maintainer_1` and `target_actors` contains the ack actor. Record `last_outbound_utc`, `hours_since_last_outbound`, and `outbound_count` on each `ack_actor_stats` entry while preserving existing inbound/SLA fields.
   - Add a formatter (e.g., `_build_ack_actor_followup_section`) that renders a Markdown table only when at least one actor has outbound data. Reuse it in `write_markdown_summary()`, `write_status_snippet()`, and `write_escalation_note()` after the existing coverage tables so the maintainer docs show outbound cadence.
   - Update `write_escalation_brief()` to include the same follow-up section and add `Last Outbound (UTC)` + `Hours Since Outbound` rows to the Blocking Actor Snapshot when data is available.
   - Ensure JSON summaries embed the new fields (no breaking removals); sanitize Markdown strings via existing helpers and keep backwards-compatible defaults (`None`/`0` when no outbound messages exist).
3. Add `test_ack_actor_followups_track_outbound_targets` in `tests/tools/test_check_inbox_for_ack_cli.py`:
   - Build a synthetic inbox with (a) an old inbound from Maintainer <2>, (b) a new outbound from Maintainer <1> to Maintainer <2>, and (c) another outbound to Maintainer <3> (different timestamp). Include `--ack-actor` flags for both maintainers plus SLA thresholds/overrides.
   - Run the CLI with `--status-snippet` pointing at a temp file so the new section renders. Assert the JSON summary contains `last_outbound_utc`/`hours_since_last_outbound`/`outbound_count` for each actor with the expected ordering, and that the snippet includes an “Ack Actor Follow-Up Activity” table listing both actors (look for the table header + each actor label).
4. Testing + logs:
   - `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/logs/pytest_followups_collect.log`
   - `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/logs/pytest_followups.log`
   - `pytest tests/tools/test_check_inbox_for_ack_cli.py -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/logs/pytest_check_inbox_suite.log`
   - `pytest tests/test_generic_loader.py::test_generic_loader -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/logs/pytest_loader.log`
5. CLI evidence run (after code/tests pass):
```
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --keywords acknowledged --keywords confirm --keywords received --keywords thanks \
  --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" \
  --sla-hours 2.5 \
  --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" \
  --fail-when-breached \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_sla_watch \
  --history-jsonl plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_history/inbox_sla_watch.jsonl \
  --history-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_history/inbox_sla_watch.md \
  --history-dashboard plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_history/inbox_history_dashboard.md \
  --status-snippet plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_status/status_snippet.md \
  --escalation-note plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_status/escalation_note.md \
  --escalation-recipient "Maintainer <2>" \
  --escalation-brief plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_status/escalation_brief_maintainer3.md \
  --escalation-brief-recipient "Maintainer <3>" \
  --escalation-brief-target "Maintainer <2>"
```
   - Confirm the status snippet, escalation note, and escalation brief now include the follow-up table with Maintainer <2>/<3> outbound metrics.
6. Docs + maintainer updates:
   - `docs/TESTING_GUIDE.md`: add a “Inbox Acknowledgement CLI (Follow-Up Activity)” subsection documenting the new test selector + behavior + log path (`reports/2026-01-23T133500Z/logs/pytest_followups.log`).
   - `docs/development/TEST_SUITE_INDEX.md`: add the new selector entry under the inbox CLI bullet list with the same log path/description.
   - `docs/fix_plan.md`: append an F1 Attempts History entry summarizing the shipped follow-up instrumentation + artifact root 2026-01-23T133500Z.
   - `inbox/response_dose_experiments_ground_truth.md`: append a “Status as of 2026-01-23T133500Z” block that quotes the new follow-up table (hours since outbound + count per actor) and links to the artifacts.
   - Draft `inbox/followup_dose_experiments_ground_truth_2026-01-23T133500Z.md` (Maintainer <2> primary, CC Maintainer <3>) referencing the per-actor outbound cadence so Maintainer <3> can confirm they were pinged.

Pitfalls To Avoid:
1. Keep all code changes inside `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/` and associated tests/docs—in-production modules under `ptycho*/` must remain untouched.
2. Preserve compatibility for existing JSON readers (never remove fields, default new outbound stats to `null`/`0`).
3. Sanitize Markdown output (pipes/newlines) via the existing helpers so tables don’t break.
4. Detect recipients flexibly (`To`, `TO`, `**To:**`, inline CC blocks); don’t assume only Maintainer <2> exists or that `To` is on its own line.
5. Ensure outbound stats only count Maintainer <1> messages (don’t mis-classify inbound Maintainer <2> messages as follow-ups).
6. Keep SLA severity logic unchanged; outbound stats are additive, not a replacement for inbound breach checks.
7. Tests must rely on the synthetic inbox fixtures—never read or mutate the real `inbox/` during pytest runs.
8. When updating docs/inbox files, append (not overwrite) and keep timestamps consistent with the artifacts directory.
9. Log every pytest/CLI command output under the new reports path for traceability.
10. Do not relax the `--fail-when-breached` guard; the CLI should still exit 2 on a breach.

If Blocked:
- Capture the failure details in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/logs/blocked.log`, note the issue in `docs/fix_plan.md` Attempts History, and ping Galph before retrying. Keep the repo dirty state untouched unless explicitly instructed.

Findings Applied (Mandatory): No relevant findings in the knowledge base.

Pointers:
- `docs/fix_plan.md:1256` — documents the new per-actor follow-up instrumentation scope and expectations for this loop.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py:338` — `scan_inbox()` currently computes per-actor inbound stats and needs the outbound tracking extension.
- `tests/tools/test_check_inbox_for_ack_cli.py:1` — regression suite where the new follow-up test must live.
- `docs/TESTING_GUIDE.md:83` — latest inbox CLI testing sections; extend with the follow-up activity selector.
- `docs/development/TEST_SUITE_INDEX.md:32` — inbox CLI selector list to update with the new test.
- `inbox/response_dose_experiments_ground_truth.md:699` — most recent status block that must gain the updated outbound table + artifact links.

Next Up (optional): Close F1 once Maintainer <2> (or escalated Maintainer <3>) acknowledges by rerunning the CLI to capture the ack event + mark the checklist done.

Doc Sync Plan (Conditional): After the new test passes, ensure `pytest --collect-only tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets -q` log is archived (see step 4) and update `docs/TESTING_GUIDE.md` §Inbox CLI + `docs/development/TEST_SUITE_INDEX.md` with the selector description and log path before finishing the loop.
