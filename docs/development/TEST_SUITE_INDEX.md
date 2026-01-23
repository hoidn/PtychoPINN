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
  - Selector: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q`
  - Tests `--escalation-note` flag for generating a Markdown escalation draft with Summary Metrics, SLA Watch, Action Items, Proposed Message blockquote, and Timeline.
  - Logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/logs/pytest_escalation_note.log`
  - Selector: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_summarizes_runs -q`
  - Tests `--history-dashboard` flag for generating a Markdown history dashboard with Summary Metrics, SLA Breach Stats, and Recent Scans timeline.
  - Logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/logs/pytest_history_dashboard.log`
  - Selector: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_supports_multiple_inbound_maintainers -q`
  - Tests `--ack-actor` repeatable flag for configurable acknowledgement sources (default: Maintainer <2>; can add Maintainer <3>).
  - Logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T024800Z/logs/pytest_ack_actor_collect.log`
  - Selector: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_custom_keywords_enable_ack_detection -q`
  - Tests that user-provided `--keywords` are honored exactly (no hidden hard-coded list).
  - Logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T024800Z/logs/pytest_keywords_collect.log`
  - Selector: `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_wait_metrics_cover_each_actor -q`
  - Tests per-actor wait metrics in `ack_actor_stats` block: each configured ack actor gets hours_since_last_inbound, inbound_count, ack_files.
  - Logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T031500Z/logs/pytest_ack_actor_wait_collect.log`
