# PyTorch Model Loading & Inference Guide

This guide explains the supported strict-bundle inference path for PyTorch
models in PtychoPINN.

## Recommended: CLI Inference Path

The CLI resolves explicit Torch configuration and calls the public
`ptycho_torch.inference.reconstruct` door. Direct Torch paths do not project
configuration into `params.cfg`; strict bundle decoding restores any scoped
legacy state.

No inference payload factory is required or supported. The CLI validates
stitching/VarPro/diagnostic knobs and resolves execution settings; the checkpoint
owns model/data and probe identity. Native probe-mask overrides are removed.
Unified Torch inference rejects explicit group counts (`--inference_groups`,
`--n_groups`, `--n_images`); omit them to reconstruct the full scan.

Minimal example:

```bash
python -m ptycho_torch.inference \
  --model_path outputs/training \
  --test_data datasets/test.npz \
  --output_dir outputs/inference
```

Pass the training directory containing `wts.h5.zip`; serving does not require
its sibling Lightning checkpoints.

## Call paths: before and after the inference factory retirement

Both frontends remain: `python -m ptycho_torch.inference` (native) and
`ptycho_inference --backend pytorch` (unified). Commit `874a5d0e9` removed
their duplicate configuration-building path, not a CLI or reconstruction algorithm.
The diagrams omit path checks and ancillary arguments.

### Before (historical, no longer supported)

```text
Native Torch CLI ──┐
                  ├─→ create_inference_payload()
Unified Torch CLI ┘     → _resolve_inference_payload()
                          → normalize_inference_patch()
                          → resolve_inference_bundle()
                            • temporary model/data/inference configs
                          → TensorFlow-style configuration projection
                          → resolve runtime execution settings
                          → InferencePayload
                               │
                               │ retain execution settings + four inference knobs
                               │ discard temporary model/data configuration
                               ▼
                          reconstruct()
                            → strict checkpoint identity + weights
                            → load diffraction dataset
                            → predict patches → stitch → return arrays
                          → CLI saves PNGs
```

The factory's projected config was not reconstruction's model authority.
`reconstruct()` loaded that identity from the checkpoint. The CLI used the
projection for a misleading group-count status line, not scan selection.
At this point the factory did not populate `params.cfg`; constructing a
TensorFlow-style config and projecting it into that global dictionary are
distinct operations.

### Now

```text
Native Torch CLI ──┐
                  ├─→ construct + validate inference knobs
Unified Torch CLI ┘   → resolve runtime execution settings
                      → reconstruct()
                        → strict checkpoint identity + weights
                        → load diffraction dataset
                        → predict patches → stitch → return arrays
                      → CLI saves PNGs
```

The four configuration overlays are `patch_weighting`, `varpro_scaling`,
`log_patch_stats`, and `patch_stats_limit`. Other retained runtime arguments,
such as `groups_per_center` and `patch_phase_alignment`, are passed separately.

Also removed: the alternate `resolve_inference_payload()` entry point and the
uncalled `_reassemble_cdi_image_torch_mmap()` wrapper, which consumed an
`InferencePayload` and called barycentric reconstruction. No replacement
payload or compatibility alias was added.

Training factories, the three legacy config-bridge functions, TensorFlow
inference, strict checkpoint schemas, and reconstruction physics are unchanged.
The strict serving contract is documented in [PyTorch workflows](../docs/workflows/pytorch.md).

## Bundle Contract

A standalone `model.pt` is not a supported serving artifact. Training writes a
strict bundle containing resolved model/data identity and weights; load that
bundle through `ptycho_inference` so architecture reconstruction cannot drift
from training.

## Pitfalls & Verification

- **Configuration boundary**: Use `ptycho_torch.inference.reconstruct` or the
  CLI; bundle identity owns model shape and scaling without legacy projection.
- **Inference coverage**: Reconstruction covers the full scan;
  `training_groups` is a training control, not a required inference setting.
- **Output mode matters**: `generator_output_mode="amp_phase"` applies sigmoid/tanh
  inside the generator. Downstream consumers expect physical values.
- **Bundle mismatches**: strict loading rejects architecture or scaling identity
  that disagrees with the saved bundle.

Quick verification checklist:

- Does strict bundle loading complete without identity or weight errors?
- Does inference produce the expected reconstruction artifacts?

## Related Docs

- `docs/workflows/pytorch.md` (end-to-end PyTorch workflow)
