# Testing Guide

## Running Tests

From the repository root:

- `pytest tests/test_generate_data.py`
- `pytest tests/test_generic_loader.py`
- `pytest tests/test_tf_helper.py`

### D0 Parity Logger CLI

The D0 parity logger (`scripts/tools/d0_parity_logger.py`) captures dose parity evidence for maintainer coordination. To run its tests:

- `pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q`

This selector validates multi-dataset Markdown coverage with all three stage tables (raw/normalized/grouped) and the `--limit-datasets` filter. See `inbox/README_prepare_d0_response.md` for the CLI scope.

### Inbox Acknowledgement CLI (SLA Watch)

The inbox acknowledgement checker (`plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py`) monitors maintainer inboxes for acknowledgements and tracks SLA compliance. To run its tests:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach -q`

This selector validates:
- `--sla-hours` flag computes breach status based on hours since last inbound
- `--fail-when-breached` flag returns exit code 2 when SLA is breached and no ack detected
- Breach does NOT trigger when acknowledgement is already received
- No breach when no inbound messages exist

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/logs/pytest_check_inbox.log`

### Inbox Acknowledgement CLI (History Logging)

The CLI also supports persistent history logging via `--history-jsonl` and `--history-markdown` flags. To run the history logging test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries -q`

This selector validates:
- JSONL history entries are appended with each CLI run
- Markdown history table rows are appended correctly
- Headers are written exactly once (not duplicated on subsequent runs)
- Second run with ack detected flips `ack_detected` to true

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/logs/pytest_check_inbox_history.log`

### Inbox Acknowledgement CLI (Status Snippet)

The CLI also supports generating a Markdown status snippet via the `--status-snippet` flag. This snippet provides a concise summary of the current wait state, including ack status, hours since inbound/outbound, SLA breach status, and a timeline table. When `--history-jsonl` is provided, the snippet also includes an "Ack Actor Breach Timeline" section showing per-actor breach start, latest scan, consecutive streak, hours past SLA, and severity. To run the status snippet test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q`

This selector validates:
- `--status-snippet` flag writes a Markdown file with "Maintainer Status Snapshot" heading
- Snippet includes "Ack Detected" status (Yes/No)
- Snippet includes SLA breach indicator when threshold is exceeded
- Snippet includes a timeline table with Maintainer <2> entries
- Snippet is idempotent (overwrites rather than appends)
- Without `--history-jsonl`: no "Ack Actor Breach Timeline" section
- With `--history-jsonl`: "Ack Actor Breach Timeline" section shows per-actor breach data

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z/logs/pytest_status_snippet.log`

### Inbox Acknowledgement CLI (Escalation Note)

The CLI also supports generating a Markdown escalation note via the `--escalation-note` flag. This note provides a prefilled follow-up draft when SLA is breached, including Summary Metrics, SLA Watch, Action Items, a Proposed Message blockquote, and a Timeline. When `--history-jsonl` is provided, the note also includes an "Ack Actor Breach Timeline" section showing per-actor breach start, latest scan, consecutive streak, hours past SLA, and severity. To run the escalation note test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q`

This selector validates:
- `--escalation-note` flag writes a Markdown file with "Escalation Note" heading
- Note includes Summary Metrics with ack status
- Note includes SLA breach indicator and warning text when threshold is exceeded
- Note includes blockquote call-to-action referencing the recipient and request pattern
- Note includes a timeline table with message entries
- Note is idempotent (overwrites rather than appends)
- Note shows "No Escalation Required" when SLA is not breached
- Without `--history-jsonl`: no "Ack Actor Breach Timeline" section
- With `--history-jsonl`: "Ack Actor Breach Timeline" section shows per-actor breach data

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z/logs/pytest_escalation_note.log`

### Inbox Acknowledgement CLI (Escalation Brief)

The CLI supports generating a Markdown escalation brief via the `--escalation-brief` flag. This brief is designed for third-party escalation (e.g., to Maintainer <3>) about a blocking actor (e.g., Maintainer <2>) who has not acknowledged a request. To run the escalation brief test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker -q`

This selector validates:
- `--escalation-brief` flag writes a Markdown file with "Escalation Brief" heading
- Brief includes "Blocking Actor Snapshot" section with actor stats (hours since inbound, SLA threshold, deadline, hours past SLA, severity, ack files)
- Brief includes "Breach Streak Summary" section with current streak, breach start, latest scan (when `--history-jsonl` provided)
- Brief includes "Action Items" section listing escalation steps
- Brief includes "Proposed Message" blockquote referencing the `--escalation-brief-recipient`
- With `--history-jsonl`: "Ack Actor Breach Timeline" section shows per-actor breach data
- Brief is idempotent (overwrites rather than appends)

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/logs/pytest_escalation_brief.log`

### Inbox Acknowledgement CLI (Follow-Up Activity)

The CLI now tracks per-actor follow-up (outbound) activity to prove how often Maintainer <1> is pinging each monitored actor. Follow-up stats are derived from `To:`/`CC:` lines in outbound messages. The new fields (`last_outbound_utc`, `hours_since_last_outbound`, `outbound_count`) appear in the JSON summary, status snippet, escalation note, and escalation brief. To run the follow-up activity test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets -q`

This selector validates:
- `ack_actor_stats` in JSON includes `last_outbound_utc`, `hours_since_last_outbound`, `outbound_count` per actor
- Timeline and match entries include `target_actors` (parsed from To:/CC: lines)
- Status snippet includes "Ack Actor Follow-Up Activity" table with outbound metrics per actor
- Outbound count is correctly computed from messages where actor is listed in To:/CC:

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/logs/pytest_followups.log`

### Inbox Acknowledgement CLI (History Follow-Up Trends)

The CLI persists per-actor follow-up (outbound) activity to history files when `--history-jsonl`, `--history-markdown`, and `--history-dashboard` flags are used with `--ack-actor`. This allows historical tracking of how often Maintainer <1> followed up with each monitored actor.

History files include:
- **JSONL**: Each entry gains an `ack_actor_followups` field containing per-actor outbound stats (`actor_label`, `last_outbound_utc`, `hours_since_last_outbound`, `outbound_count`)
- **Markdown**: Table gains an "Ack Actor Follow-Ups" column showing entries like `Maintainer 2: 8 (0.2h ago)<br>Maintainer 3: 3 (0.2h ago)`
- **Dashboard**: Gains "## Ack Actor Follow-Up Trends" section showing latest outbound UTC, hours since outbound, max outbound count, and scans with outbound per actor

To run the history follow-up persistence test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist -q`

This selector validates:
- JSONL entry contains `ack_actor_followups` with correct outbound counts per actor
- Markdown history table includes "Ack Actor Follow-Ups" column header and Maintainer labels
- Dashboard includes "## Ack Actor Follow-Up Trends" section with both actors
- Dashboard table includes "Max Outbound Count" and "Scans w/ Outbound" columns

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/logs/pytest_history_followups.log`

### Inbox Acknowledgement CLI (History Dashboard)

The CLI also supports generating a Markdown history dashboard via the `--history-dashboard` flag. This dashboard aggregates data from the JSONL history log to show total scans, ack count, breach count, longest wait, and a recent scans timeline. To run the history dashboard test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_summarizes_runs -q`

This selector validates:
- `--history-dashboard` flag writes a Markdown file with "Inbox History Dashboard" heading
- Dashboard includes Summary Metrics (Total Scans, Ack Count, Breach Count)
- Dashboard includes SLA Breach Stats (Longest Wait, Last Ack Timestamp)
- Dashboard includes Recent Scans table with timestamps from the JSONL entries
- Dashboard is idempotent (overwrites rather than appends)
- `--history-dashboard` requires `--history-jsonl` to be specified (validation test)

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/logs/pytest_history_dashboard.log`

### Inbox Acknowledgement CLI (History Dashboard - Actor Severity Trends)

The history dashboard now includes an "Ack Actor Severity Trends" section that aggregates per-actor severity data across all JSONL entries. This shows severity counts (critical/warning/ok/unknown), longest wait, and latest scan timestamp per actor. To run the actor severity trends test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q`

This selector validates:
- Dashboard includes "## Ack Actor Severity Trends" section
- Table shows per-actor severity counts aggregated across scans
- Table includes Longest Wait and Latest Scan columns
- Actors are sorted by severity priority (critical > warning > ok > unknown)
- Gracefully handles history entries without per-actor data

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/logs/pytest_history_dashboard_actor_severity.log`

### Inbox Acknowledgement CLI (History Dashboard - Actor Breach Timeline)

The history dashboard now includes an "Ack Actor Breach Timeline" section that tracks per-actor breach state across JSONL entries. This shows breach start timestamps, latest scans, consecutive breach streaks, and hours past SLA for actors in warning/critical states. Actors that return to OK/unknown have their streaks reset. To run the actor breach timeline test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline -q`

This selector validates:
- Dashboard includes "## Ack Actor Breach Timeline" section
- Table shows per-actor breach start and latest scan timestamps
- Table shows Current Streak (consecutive scans in warning/critical)
- Table shows Hours Past SLA (hours_since_inbound - sla_threshold)
- Actors are sorted by severity priority (critical > warning)
- Actors in OK/unknown status are NOT included in the timeline
- Gracefully shows "No active breaches" when all actors are within SLA

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/logs/pytest_history_dashboard_actor_breach.log`

### Inbox Acknowledgement CLI (Multi-Actor Ack)

The CLI supports configurable ack actors via the `--ack-actor` repeatable flag. By default, only `Maintainer <2>` is treated as an acknowledgement source. Use `--ack-actor "Maintainer <3>"` to add additional actors. To run the multi-actor ack test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_supports_multiple_inbound_maintainers -q`

This selector validates:
- Default behavior: only Maintainer <2> messages trigger ack detection
- With `--ack-actor "Maintainer <3>"`: messages from M3 also trigger ack detection
- JSON output includes normalized `ack_actors` in parameters

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T024800Z/logs/pytest_ack_actor_collect.log`

### Inbox Acknowledgement CLI (Custom Keywords)

The CLI honors user-provided `--keywords` exactly, with no hidden hard-coded keyword list. Keywords must match for ack detection to trigger. To run the custom keywords test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_custom_keywords_enable_ack_detection -q`

This selector validates:
- User-specified keywords are honored (no hidden overrides)
- Ack detection requires at least one keyword hit
- Default keywords are used when `--keywords` is not specified

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T024800Z/logs/pytest_keywords_collect.log`

### Inbox Acknowledgement CLI (Per-Actor Wait Metrics)

The CLI tracks per-actor wait metrics via the `ack_actor_stats` block in JSON output and the "Ack Actor Coverage" table in Markdown outputs. Each configured ack actor (via `--ack-actor` flags) gets its own metrics: last_inbound_utc, hours_since_last_inbound, inbound_count, ack_files. To run the per-actor wait metrics test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_wait_metrics_cover_each_actor -q`

This selector validates:
- `ack_actor_stats` block is present in JSON output with entries for each configured actor
- Each actor has distinct `hours_since_last_inbound` values based on their message timestamps
- Inbound counts are correct per actor
- Markdown summary includes "Ack Actor Coverage" table with per-actor rows

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T031500Z/logs/pytest_ack_actor_wait_collect.log`

### Inbox Acknowledgement CLI (SLA Deadline/Severity)

The CLI computes SLA deadline, breach duration, and severity fields when `--sla-hours` is provided. The `sla_watch` block in JSON output includes:
- `deadline_utc`: last inbound timestamp + sla_hours
- `breach_duration_hours`: hours elapsed beyond the threshold (0 when not breached)
- `severity`: "ok" (not breached), "warning" (<1 hour late), "critical" (>=1 hour late), "unknown" (no inbound messages)

These fields appear in JSON, Markdown summary, status snippet, escalation note, history JSONL, and CLI stdout. To run the deadline/severity test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_reports_deadline_and_severity -q`

This selector validates:
- JSON `sla_watch` block contains `deadline_utc`, `breach_duration_hours`, and `severity`
- Markdown summary includes "Deadline (UTC)", "Breach Duration", and "Severity" lines
- Severity is "critical" when breach >= 1 hour
- Severity resets to "ok" when threshold exceeds wait time (no breach)

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T040500Z/logs/pytest_sla_severity_collect.log`

### Inbox Acknowledgement CLI (Per-Actor SLA Metrics)

When `--sla-hours` is set, the CLI computes per-actor SLA fields within the `ack_actor_stats` block for each configured `--ack-actor`. Each actor entry includes:
- `sla_deadline_utc`: last inbound from that actor + sla_hours (None if no inbound)
- `sla_breached`: whether that specific actor's wait has exceeded the threshold
- `sla_breach_duration_hours`: hours elapsed beyond threshold for that actor
- `sla_severity`: "ok", "warning", "critical", or "unknown" (no inbound from actor)
- `sla_notes`: descriptive text explaining the actor's SLA status

These per-actor fields appear in JSON `ack_actor_stats`, the Markdown "Ack Actor Coverage" table (with Deadline/Breached/Severity/Notes columns when --sla-hours is used), status snippet, escalation note, and CLI stdout. To run the per-actor SLA metrics test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_metrics_include_deadline -q`

This selector validates:
- JSON `ack_actor_stats` includes per-actor `sla_deadline_utc`, `sla_breached`, `sla_breach_duration_hours`, `sla_severity`, `sla_notes`
- Breached actors show severity "warning" (<1 hour late) or "critical" (>=1 hour late)
- Actors with no inbound messages show `sla_severity == "unknown"` and appropriate notes
- Markdown Ack Actor Coverage table shows expanded columns (Deadline/Breached/Severity/Notes)

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T050500Z/logs/pytest_sla_metrics_collect.log`

### Inbox Acknowledgement CLI (Per-Actor SLA Overrides)

The CLI supports per-actor SLA threshold overrides via the repeatable `--ack-actor-sla` flag. This allows different actors to have different SLA thresholds. Use the format `--ack-actor-sla "actor=hours"` (e.g., `--ack-actor-sla "Maintainer <2>=2.0"`). Actors without overrides inherit the global `--sla-hours` threshold.

When per-actor overrides are in use, each actor entry in `ack_actor_stats` includes:
- `sla_threshold_hours`: the effective threshold for that actor (override or global)

The override map is also stored in `parameters["ack_actor_sla_hours"]` for reproducibility. The Markdown "Ack Actor Coverage" table gains a "Threshold (hrs)" column when overrides are present. To run the per-actor SLA overrides test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_overrides_thresholds -q`

This selector validates:
- JSON `parameters["ack_actor_sla_hours"]` contains the override map
- JSON `ack_actor_stats` includes `sla_threshold_hours` per actor
- Per-actor breach status uses the actor-specific threshold (not global)
- Markdown Ack Actor Coverage table shows "Threshold (hrs)" column
- CLI stdout shows "Per-actor SLA overrides" and per-actor threshold values

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T060500Z/logs/pytest_sla_override_collect.log`

### Per-Actor SLA Summary (Grouped by Severity)

The `ack_actor_summary` structure groups monitored actors by severity (critical/warning/ok/unknown), providing a quick view of which actors are breaching SLA vs within SLA. When `--sla-hours` is set:
- `critical`: actors breaching SLA by >= 1 hour
- `warning`: actors breaching SLA by < 1 hour
- `ok`: actors within their SLA threshold (or acknowledged)
- `unknown`: actors with no inbound messages

The summary appears in:
- JSON `ack_actor_summary` with bucket arrays
- Markdown "## Ack Actor SLA Summary" section with severity subsections
- Status snippet "## Ack Actor SLA Summary" section
- Escalation note "## Ack Actor SLA Summary" section
- CLI stdout "Ack Actor SLA Summary:" with `[CRITICAL]`/`[WARNING]`/`[OK]`/`[UNKNOWN]` labels

To run the per-actor SLA summary test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_sla_summary_flags_breach -q`

This selector validates:
- JSON `ack_actor_summary` contains correct buckets (critical/warning/ok/unknown)
- Breaching actors (e.g., Maintainer <2>) appear in the `critical` bucket
- Within-SLA actors (e.g., Maintainer <3>) appear in the `ok` bucket
- Markdown includes "## Ack Actor SLA Summary" with severity subsections
- CLI stdout includes "Ack Actor SLA Summary:" with severity bucket labels

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T070500Z/logs/pytest_ack_actor_summary.log`

### Inbox Acknowledgement CLI (History Severity Persistence)

The CLI persists the `ack_actor_summary` in both history logging formats when `--history-jsonl` and `--history-markdown` flags are used with `--sla-hours`. This allows historical tracking of which actors were breaching SLA at each scan.

History files include:
- **JSONL**: Each entry gains an `ack_actor_summary` field containing the full severity bucket structure
- **Markdown**: Table gains an "Ack Actor Severity" column showing entries like `[CRITICAL] Maintainer 2 (4.36h > 2.00h)<br>[UNKNOWN] Maintainer 3`

To run the history severity persistence test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity -q`

This selector validates:
- JSONL entry contains `ack_actor_summary` with `critical[0]["actor_id"] == "maintainer_2"`
- JSONL entry contains `ack_actor_summary` with `unknown[0]["actor_id"] == "maintainer_3"`
- Markdown row contains `[CRITICAL] Maintainer 2`
- Markdown row contains `[UNKNOWN] Maintainer 3`
- Markdown table has "Ack Actor Severity" column header

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/logs/pytest_ack_actor_history.log`

### Maintainer Status Automation CLI

The `update_maintainer_status.py` CLI (`plans/active/DEBUG-SIM-LINES-DOSE-001/bin/update_maintainer_status.py`) automates the generation of maintainer status blocks and follow-up notes from inbox scan results. It reads `inbox_scan_summary.json` and produces:
1. A Markdown status block appended to the response document
2. A follow-up note with SLA metrics and artifact references

To run the automation CLI tests:

- `pytest tests/tools/test_update_maintainer_status.py::test_cli_generates_followup -q`

This selector validates:
- Response document gains a status block with "### Status as of" heading and actor tables
- Follow-up note contains To/CC recipients, SLA metrics, and action items
- Artifacts are referenced in both outputs
- CLI exits 0 on success, non-zero on missing inputs

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T153500Z/logs/pytest_update_status.log`

## Test Areas

- `tests/tools/`: CLI and tooling tests (e.g., D0 parity logger).
- `tests/workflows/`: end-to-end workflow checks.
- `tests/io/`, `tests/image/`: IO and image utilities.
- `tests/torch/`: Torch-related tests (if present).

## Notes

- Tests assume local datasets are available where required.
- Some modules include legacy tests under `ptycho/tests/`.
