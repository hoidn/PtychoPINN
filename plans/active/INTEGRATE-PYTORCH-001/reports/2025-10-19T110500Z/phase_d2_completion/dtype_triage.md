# Phase D1d — PyTorch Inference Dtype Mismatch Triage

**Date:** 2025-10-19
**Related Focus:** INTEGRATE-PYTORCH-001-STUBS Phase D1d
**Upstream Tasks:** D1c checkpoint serialization fix complete; integration test now fails later in inference.

## Failure Signature
- Selector: `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv`
- Artifact: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T134500Z/phase_d2_completion/pytest_integration_checkpoint_green.log`
- Stack trace terminates at `torch.nn.Conv2d._conv_forward` with:
  ```
  RuntimeError: Input type (double) and bias type (float) should be the same
  ```
- Lightning reports "Successfully loaded model from checkpoint" immediately before the crash, confirming D1c success and isolating the new issue to the inference forward pass.

## Immediate Observations
- The failing tensor is the first convolution input inside `ptycho_torch/model.py:97`, meaning the minibatch handed to `PtychoPINN_Lightning` arrives as `torch.float64`.
- Training path remains green, implying `_train_with_lightning` feeds float32 tensors. The dtype flip therefore occurs along the inference/stitching pipeline introduced in Phase C.
- `PtychoDataContainerTorch` enforces `X_full.dtype == np.float32` and converts to `torch.float32` (`data_container_bridge.py:172-210`). Any float64 conversion must happen after container construction.
- `_build_inference_dataloader` reuses `_get_tensor`, which returns tensors unchanged when already `torch.Tensor`. No explicit `.double()` conversion occurs.
- `_reassemble_cdi_image_torch` iterates over the loader and calls `lightning_module(X_batch)`. There is no explicit cast, so either
  1. `infer_loader` yields float64 batches (unlikely given container guarantees), or
  2. Lightning module sets its precision context to double when `self.training` is False, or
  3. A pre-processing helper invoked on the first forward pass casts to double (e.g., `IntensityScalerModule`).

## Hypotheses
1. **Loader Collation Hypothesis** — During inference we feed a mix of float64 + float32 tensors into `TensorDataset`. PyTorch may upcast all tensors in the tuple to the widest dtype during batching. `coords_nominal` is float32, but `global_offsets` (float64) is not part of the dataset; however the container also exposes `local_offsets` (float64). If `_get_tensor` inadvertently picks a float64 tensor (e.g., due to attribute aliasing), the dataset could propagate doubles.
2. **Module Preprocessing Hypothesis** — `PtychoPINN_Lightning.forward` or one of its helpers might call `.double()` on inputs when `self.predict` is True. Need to trace the forward path for evaluation mode.
3. **Torch Default Tensor Type Hypothesis** — Somewhere in inference we convert NumPy arrays back to tensors without specifying dtype, causing default `float64`. Example: if `_ensure_container` receives RawDataTorch without bridging config, some arrays might be `np.float64`. The container would flag `X_full` but not intermediate arrays used later.
4. **Lightning Trainer Precision Setting** — The stored Lightning module might set `precision='64-true'` during training (unlikely but should confirm). Inspect trainer configuration captured in D1c fix.

## Proposed Evidence Tasks
- Instrument `_build_inference_dataloader` to log `infer_X.dtype` before constructing `TensorDataset` (temporary print via targeted test) — confirm dataset dtype.
- Add quick assertion in `TestReassembleCdiImageTorch` to check `X_batch.dtype` using a simple stub dataset. This can become the RED test for Phase D1d.
- Explore `PtychoPINN_Lightning.forward` to ensure it does not cast to double when `self.predict=True`. Grep for `.double()` or `.to(torch.float64)` (none found so far).
- If loader dtype is correct, investigate `lightning_module` internals — e.g., the `IntensityScalerModule` might convert to double by combining with float64 scale factors.

## References
- `specs/data_contracts.md` §1 — Diffraction arrays must be float32.
- `specs/ptychodus_api_spec.md` §4.4–4.5 — Loader and model dtype requirements (TensorFlow parity).
- `docs/workflows/pytorch.md` §§5–7 — Notes on deterministic loaders and dtype expectations.
- `tests/torch/test_workflows_components.py` — Candidate location for new dtype regression tests.

## Next Steps (for plan)
1. Add Phase D1d checklist row to `phase_d2_completion.md` covering RED test, implementation, and green integration rerun for dtype enforcement.
2. Update `docs/fix_plan.md` Attempts with dtype mismatch discovery and link to this triage note.
3. Move `train_debug.log` into this report directory (or delete) to keep repo root clean.
