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

Artifact logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T020500Z/logs/pytest_check_inbox.log`

## Test Areas

- `tests/tools/`: CLI and tooling tests (e.g., D0 parity logger).
- `tests/workflows/`: end-to-end workflow checks.
- `tests/io/`, `tests/image/`: IO and image utilities.
- `tests/torch/`: Torch-related tests (if present).

## Notes

- Tests assume local datasets are available where required.
- Some modules include legacy tests under `ptycho/tests/`.
