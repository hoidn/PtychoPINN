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
