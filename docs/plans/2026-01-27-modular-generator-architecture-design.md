# Modular Generator Architecture Design (Unsupervised Ptychography)

Date: 2026-01-27

## Summary
We will modularize the unsupervised ptychography pipeline at the generator boundary so the current CNN PINN can be replaced by alternative unsupervised architectures (FNO, hybrid U-NO, etc.) without rewriting the physics or consistency pipeline. The generator will output per-patch object estimates in a canonical real/imag format. Spatial consistency (stitch/extract) and the forward physics decoder remain unchanged and outside the generator.

## Goals
- Swap generator architectures without changing the physics-informed loop.
- Keep both TensorFlow and PyTorch CLIs stable.
- Avoid overengineering: use a thin registry + minimal adapters.
- Maintain performance for large datasets (up to ~10k patches).

## Non-goals
- Backward compatibility for existing CNN checkpoints.
- Refactoring core physics or model files marked stable.
- Introducing plugin frameworks or dynamic loading.

## Design Decisions
- Backend scope: both TensorFlow and PyTorch.
- Generator boundary: outputs patch-wise object estimates; consistency + physics decoder stay outside.
- Output representation: real/imag patches (canonical), converted to complex with a single tensor op.
- Layout: backend-native (TF: [B, N, N, C, 2], Torch: [B, C, N, N, 2]).
- Config selection: YAML path `model.architecture` (string alias, default `cnn`).
- Wiring location: workflow layers only (`ptycho.workflows.components` and `ptycho_torch/workflows/components.py`).
- Inputs: amplitude patches + positions + probe (positions/probe may be passthrough for now).

## Architecture Overview

### Generator Contract (Minimal)
Define a tiny generator contract shared by both backends:
- Inputs: amplitude patches, positions, probe.
- Output: real/imag patch tensor in backend-native layout.
- Conversion: a single helper to create complex patches for downstream steps.

The generator does not perform spatial consistency. It only predicts patch-wise object estimates.

### Registry
Introduce a simple registry per backend:
- `ptycho/generators/registry.py`
- `ptycho_torch/generators/registry.py`

Registry maps `model.architecture` to generator classes:
- `cnn` (existing CNN PINN wrapped as generator)
- `fno` (future)
- `hybrid` (future)

No plugin system, no dynamic loading beyond this dict.

### Pipeline Wiring
Workflow layers will:
1. Resolve `model.architecture`.
2. Build the generator via registry.
3. Run generator -> spatial consistency -> physics decoder.

All other pipeline components remain unchanged.

## Performance Notes
- Conversion to complex must be single-op and graph/JIT-friendly:
  - TF: `tf.complex(out[..., 0], out[..., 1])`
  - Torch: `torch.view_as_complex(out.contiguous())`
- No Python loops over patches; only batched tensor ops.
- Optional, lightweight shape validation behind a debug flag.

## Mapping to Proposed Architectures
- Arch C (CNN PINN): direct generator implementation.
- Arch A (Cascaded FNO -> CNN): generator returns real/imag patches (FNO + CNN inside generator).
- Arch B (Hybrid U-NO): generator returns real/imag patches (integrated model).

## FNO/Hybrid Lifting Layer Decision
- Use a lightweight, spatially aware lifter: two 3x3 convs with GELU in between (padding "same").
- Applies to both cascaded FNO (Arch A) and hybrid U-NO (Arch B).
- Do not use 1x1-only lift (too weak for speckle geometry) or deep 4-6 layer stacks (memory heavy, can blur physics signal).
- Lifter must precede the first Fourier layer; avoid FFT of raw intensity directly (prevent "double Fourier" trap).

## Risks and Mitigations
- Risk: mis-specified output shape. Mitigation: strict output shape checks in generator boundary.
- Risk: divergence between TF and Torch contracts. Mitigation: shared contract spec and mirrored registries.
- Risk: config mismatch. Mitigation: default `cnn` and clear error on unknown arch.

## Follow-up Work (Implementation Plan)
- Add `model.architecture` to config parsing + docs.
- Implement generator registry and CNN adapter in both backends.
- Wire selection in workflow components (TF + Torch).
- Add minimal tests for registry resolution and output shape checks.
