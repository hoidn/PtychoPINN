# PyTorch Integration Test Plan

## Goal

Create an automated integration test that exercises the PyTorch (``ptycho_torch``) training and inference path end to end, providing parity with the TensorFlow workflow test and preventing regressions in the nascent PyTorch stack.

## Context

- TensorFlow already has `tests/test_integration_workflow.py`, which launches `scripts/training/train.py` and then `scripts/inference/inference.py` with real NPZ data.
- The PyTorch backend currently lacks a similar guardrail; existing notebooks and parity scaffolds are manual or partially broken.
- `ptycho_torch/train.py` expects memory-mapped diffraction data (`ptycho_torch/dset_loader_pt_mmap.py`) plus companion probe files, uses Lightning for training, and enables MLflow autologging by default.

## Proposed Test Shape

1. **Runtime harness**: new `tests/test_pytorch_integration.py` mirroring the TensorFlow suite—use `tempfile.TemporaryDirectory()` and invoke training and inference as separate subprocesses.
2. **Training command**: run `python -m ptycho_torch.train --ptycho_dir <tmp>/ptycho --probe_dir <tmp>/probes` with overrides to keep runtime under ~2 minutes.
3. **Inference check**: after training completes, run a lightweight inference script (to author) that loads the Lightning checkpoint and executes a `trainer.predict` or manual forward pass on a single batch from the saved memmap.
4. **Assertions**: confirm
   - Lightning checkpoint or state dictionary exists.
   - Prediction tensor has expected shape `(batch, channels, N, N)` and finite values.
   - No errors in stderr/stdout and optional MLflow artifacts suppressed in CI scenario.

<plan_update version="1.0">
  <trigger>Shift the PyTorch integration regression to use GPU execution by default per new baseline directive.</trigger>
  <focus_id>TEST-PYTORCH-001</focus_id>
  <documents_read>docs/index.md, docs/workflows/pytorch.md, plans/pytorch_integration_test_plan.md</documents_read>
  <current_plan_path>plans/pytorch_integration_test_plan.md</current_plan_path>
  <proposed_changes>Update the prerequisites and acceptance criteria so CUDA GPU execution is required; replace CPU-only guidance with GPU language in setup steps.</proposed_changes>
  <impacts>Test infrastructure must guarantee at least one CUDA device; CPU execution is now a fallback path and no longer satisfies the baseline contract.</impacts>
  <ledger_updates>Record this GPU-baseline shift in docs/fix_plan.md when logging the next attempt for TEST-PYTORCH-001.</ledger_updates>
  <status>approved</status>
</plan_update>

## Prerequisites & Utilities

- **Environment requirement**: run on a CUDA-capable host and pin the test harness to a specific GPU by setting `CUDA_VISIBLE_DEVICES="0"` (or the appropriate ID). Capture driver + CUDA versions in the report; CPU runs should be called out as reduced-capability evidence only.
- **Dataset fixture**: assemble a minimal NPZ + probe pair with only a handful of patterns (e.g. subsample an existing fly dataset). Store under `tests/fixtures/pytorch_integration/` for deterministic runs.
- **Script adjustments**:
  - Add CLI flags or environment overrides to `ptycho_torch/train.py` for `--max_epochs`, `--accelerator`, and `--disable_mlflow`.
  - Ensure the dataloader respects a fixed random seed and small batch size when running in test mode.
  - Provide an inference helper (e.g. `ptycho_torch/inference.py`) that loads the trained Lightning module without requiring MLflow.

## Test Flow (Detailed)

1. **Setup**
   - Copy fixture NPZ/probe into temporary ptycho/probe directories.
   - Export `CUDA_VISIBLE_DEVICES="0"` (or another specific GPU ID) to guarantee CUDA execution and deterministic device placement.
2. **Train**
   - Invoke train module with `--max_epochs=1`, `--batch_size=4`, `--disable_mlflow`, and `--accelerator=cuda` while `CUDA_VISIBLE_DEVICES` pins the GPU selection.
   - Assert return code `0`.
3. **Infer**
   - Call inference helper with paths to checkpoint and memmap directory.
   - Collect output tensor (or reconstructed object) into numpy for verification.
4. **Validate**
   - Check artifact paths exist (checkpoint, memmap state file).
   - Assert tensor dtype (`float32`/`complex64`) and shapes align with `DataConfig().get('N')` and channel expectations.

## Acceptance Criteria

- Test passes reliably on a single-GPU CUDA environment within 2 minutes. CPU-only runs are acceptable only as explicitly documented fallbacks and must be flagged in summaries.
- Fails loudly if training script, dataloader contracts, or inference pipeline regress.
- No external services (MLflow, GPU drivers) required.

## Open Questions

- What is the minimal fixture resolution that still exercises the forward model? Need to measure runtime vs. fidelity.
- Should inference live in an importable module or be implemented directly inside the test to avoid new binaries?
- How should we manage generated memmap files—delete after test or reuse across steps?

## Next Steps

1. Implement configuration overrides and optional MLflow disablement in `ptycho_torch/train.py`.
2. Build fixture dataset and probe for testing.
3. Author inference helper capable of batch prediction from saved checkpoint.
4. Implement the pytest integration test following the flow above.
