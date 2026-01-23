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

The CLI also supports generating a Markdown status snippet via the `--status-snippet` flag. This snippet provides a concise summary of the current wait state, including ack status, hours since inbound/outbound, SLA breach status, and a timeline table. To run the status snippet test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q`

This selector validates:
- `--status-snippet` flag writes a Markdown file with "Maintainer Status Snapshot" heading
- Snippet includes "Ack Detected" status (Yes/No)
- Snippet includes SLA breach indicator when threshold is exceeded
- Snippet includes a timeline table with Maintainer <2> entries
- Snippet is idempotent (overwrites rather than appends)

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/logs/pytest_status_snippet.log`

### Inbox Acknowledgement CLI (Escalation Note)

The CLI also supports generating a Markdown escalation note via the `--escalation-note` flag. This note provides a prefilled follow-up draft when SLA is breached, including Summary Metrics, SLA Watch, Action Items, a Proposed Message blockquote, and a Timeline. To run the escalation note test:

- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q`

This selector validates:
- `--escalation-note` flag writes a Markdown file with "Escalation Note" heading
- Note includes Summary Metrics with ack status
- Note includes SLA breach indicator and warning text when threshold is exceeded
- Note includes blockquote call-to-action referencing the recipient and request pattern
- Note includes a timeline table with message entries
- Note is idempotent (overwrites rather than appends)
- Note shows "No Escalation Required" when SLA is not breached

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/logs/pytest_escalation_note.log`

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

## Test Areas

- `tests/tools/`: CLI and tooling tests (e.g., D0 parity logger).
- `tests/workflows/`: end-to-end workflow checks.
- `tests/io/`, `tests/image/`: IO and image utilities.
- `tests/torch/`: Torch-related tests (if present).

## Notes

- Tests assume local datasets are available where required.
- Some modules include legacy tests under `ptycho/tests/`.
