# Blocker: Device Mismatch in PyTorch Inference

**Date:** 2025-11-12T19:30:51Z
**Focus:** INTEGRATE-PYTORCH-PARITY-001 (Phase R config defaults)
**Status:** BLOCKED

## Summary
PyTorch inference CLI successfully loads the trained bundle but fails during inference with a device mismatch error. The input tensors are moved to CUDA but the model weights remain on CPU.

## Error Signature
```
Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

## Command Executed
```bash
CUDA_VISIBLE_DEVICES="0" python scripts/inference/inference.py \
  --model_path "$HUB"/cli/pytorch_cli_smoke_training/train_outputs \
  --test_data tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
  --backend pytorch \
  --torch-accelerator cuda \
  --torch-num-workers 0 \
  --torch-inference-batch-size 2 \
  --output_dir "$HUB"/cli/pytorch_cli_smoke_training/inference_outputs
```

## Evidence
- Model loaded successfully from `wts.h5.zip` via `load_torch_bundle`
- Both models present: `['diffraction_to_obj', 'autoencoder']`
- params.cfg restored with correct keys
- Inference started on 64 images with accelerator=cuda
- **FAILURE:** Device mismatch when model forward pass executed

## Root Cause Hypothesis
The `load_torch_bundle` function loads model state_dicts but does not move the reconstructed model to the requested accelerator device. The inference code moves input tensors to CUDA via `--torch-accelerator cuda`, but the model remains on CPU, causing the type mismatch.

##Expected Fix
The `_run_inference_and_reconstruct` function (or the bundle loader) must call `model.to(device)` after model reconstruction when an accelerator is specified.

## Logs
- CLI log: `$HUB/cli/pytorch_cli_smoke_training/inference.log`
- Exit code: 1

## Dependencies
None - standalone inference issue

## Next Actions
1. File finding DEVICE-MISMATCH-001 in docs/findings.md
2. Either:
   a) Fix `load_torch_bundle` to accept device parameter and move model, OR
   b) Fix `_run_inference_and_reconstruct` to move model to execution_config.accelerator
3. Rerun inference CLI smoke test
4. Update hub summaries with successful inference evidence

