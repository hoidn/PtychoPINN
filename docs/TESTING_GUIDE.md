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

## Test Areas

- `tests/tools/`: CLI and tooling tests (e.g., D0 parity logger).
- `tests/workflows/`: end-to-end workflow checks.
- `tests/io/`, `tests/image/`: IO and image utilities.
- `tests/torch/`: Torch-related tests (if present).

## Notes

- Tests assume local datasets are available where required.
- Some modules include legacy tests under `ptycho/tests/`.
