### Turn Summary
Extracted TensorFlow inference helper `_run_tf_inference_and_reconstruct()` from `scripts/inference/inference.py` for programmatic API parity with PyTorch.
Added `extract_ground_truth()` utility and refactored `perform_inference()` as deprecated wrapper that calls the new helper.
Created 7 tests validating signature, defaults, and deprecation; all passed. Integration workflow test also passed.
Next: Task 2 — create `scripts/pytorch_api_demo.py` demonstrating both backends.
Artifacts: plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/ (pytest_tf_helper.log, pytest_integration.log, pytest_collect.log)
