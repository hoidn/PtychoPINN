# PARALLEL-API-INFERENCE — Programmatic TF/PyTorch API parity

## Goal
Provide a single Python entry point that exercises the Ptychodus backend selector with both TensorFlow and PyTorch backends (programmatic training + inference) without relying on the CLI wrappers.

## Scope
1. **Training parity**
   - Define a thin helper that wraps `run_cdi_example_with_backend` for a given backend and dataset.
   - Ensure TensorFlow persistence (`model_manager.save`) and PyTorch bundle logging mirror the CLI.

2. **Inference parity**
   - Extract the TensorFlow inference logic currently embedded in `scripts/inference/inference.py` so programmatic callers can reuse it without re-implementing CLI glue.
   - Reuse `_run_inference_and_reconstruct` for PyTorch and the new helper for TF so outputs (PNG, tensor dumps) match the CLI UX.

3. **Artifacts**
   - Save code under `scripts/pytorch_api_demo.py` (or similar) with parallel TF/PyTorch flows.
   - Add a regression test (smoke-level) that imports the script and runs both backends with small N=64 fixtures.

## Tasks
1. **Refactor TensorFlow inference helper**
   - Move reusable pieces from `scripts/inference/inference.py` (data container prep, `perform_inference`, output saving) into a callable function that accepts `RawData`, model bundle path, and configuration.

2. **Implement unified demo script**
   - Load `RawData` fixture once.
   - Build `TrainingConfig` and `InferenceConfig` objects for each backend.
   - Call the backend selector for training (with optional stitching), invoke the new TF helper or existing PyTorch helper for inference, and write artifacts to `tmp/api_demo/<backend>/`.

3. **Add a smoke test**
   - Under `tests/scripts/test_api_demo.py`, import the script’s `run_backend` function and run both backends with `tmp_path` outputs (skip or mark slow if CUDA unavailable).

4. **Document the usage**
   - Reference `specs/ptychodus_api_spec.md` and add a short “Programmatic usage” section to `docs/workflows/pytorch.md` or an API README.

## Risks / Open Questions
- TensorFlow inference currently depends on CLI-specific logging; need to isolate the minimal reusable subset.
- Smoke test runtime may be high; consider a reduced dataset or mocks.
