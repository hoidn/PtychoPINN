# Test Suite Index

This is a minimal index of key tests in this branch.

- `tests/test_generate_data.py`: data generation sanity checks.
- `tests/test_generic_loader.py`: loader and container checks.
- `tests/test_tf_helper.py`: core TensorFlow helper utilities.
- `tests/workflows/`: workflow-level tests and scripts.
- `tests/tools/test_d0_parity_logger.py`: D0 parity logger CLI acceptance tests.
  - Selector: `pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q`
  - Validates multi-dataset Markdown coverage with raw/normalized/grouped stage tables.
  - See `inbox/README_prepare_d0_response.md` for CLI scope.
- `tests/tools/test_check_inbox_for_ack_cli.py`: Inbox acknowledgement CLI tests (SLA watch + history logging + status snippet).
  - Selector: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach -q`
  - Tests `--sla-hours` and `--fail-when-breached` flags for SLA breach detection.
  - Logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/logs/pytest_check_inbox.log`
  - Selector: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries -q`
  - Tests `--history-jsonl` and `--history-markdown` flags for persistent history logging.
  - Logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/logs/pytest_check_inbox_history.log`
  - Selector: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q`
  - Tests `--status-snippet` flag for generating a Markdown status snapshot with ack status, SLA breach, and timeline.
  - Logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/logs/pytest_status_snippet.log`
