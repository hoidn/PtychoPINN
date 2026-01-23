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

## Test Areas

- `tests/tools/`: CLI and tooling tests (e.g., D0 parity logger).
- `tests/workflows/`: end-to-end workflow checks.
- `tests/io/`, `tests/image/`: IO and image utilities.
- `tests/torch/`: Torch-related tests (if present).

## Notes

- Tests assume local datasets are available where required.
- Some modules include legacy tests under `ptycho/tests/`.
