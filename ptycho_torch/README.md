# PyTorch Model Loading & Inference Guide

This guide explains the supported strict-bundle inference path for PyTorch
models in PtychoPINN.

## Recommended: CLI Inference Path

Use the CLI for the most reliable workflow. It performs config bridging, calls
`update_legacy_dict(params.cfg, config)`, wires the generator registry, and runs
batched inference with consistent output handling.

Minimal example:

```bash
python -m ptycho_torch.inference \
  --model_path outputs/training \
  --test_data datasets/test.npz \
  --output_dir outputs/inference
```

Use this when you have a full training directory (checkpoints, configs, metadata)
and want a safe, supported inference path.

## Bundle Contract

A standalone `model.pt` is not a supported serving artifact. Training writes a
strict bundle containing resolved model/data identity and weights; load that
bundle through `ptycho_inference` so architecture reconstruction cannot drift
from training.

## Pitfalls & Verification

- **CONFIG-001**: Use the factory or CLI. Direct instantiation can silently mis-sync
  gridsize and channel count.
- **training_groups is required**: The factory rejects missing `training_groups`.
  Use the test sample count as a safe default.
- **Output mode matters**: `generator_output_mode="amp_phase"` applies sigmoid/tanh
  inside the generator. Downstream consumers expect physical values.
- **Bundle mismatches**: strict loading rejects architecture or scaling identity
  that disagrees with the saved bundle.

Quick verification checklist:

- Does strict bundle loading complete without identity or weight errors?
- Does inference produce the expected reconstruction artifacts?

## Related Docs

- `docs/workflows/pytorch.md` (end-to-end PyTorch workflow)
