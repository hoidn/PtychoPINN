# PyTorch CDI Generators

This package contains the learned modules used in PyTorch Lightning PINN models.

## Overview

The core builder in `ptycho_torch/model.py` selects modules via
`config.model.architecture`. Both config types share the private Literal in
`ptycho/_architecture_names.py`; the authoritative enumeration is in
`docs/specs/spec-ptycho-config-bridge.md` §3. The table below is illustrative only.

| Architecture | Description | Status |
|--------------|-------------|--------|
| `cnn` (default) | U-Net based CNN generator | ✅ Integrated |
| `ffno` | Constant-resolution factorized Fourier operator | ✅ Integrated |
| `fno` | Cascaded FNO + CNN refiner (Arch A) | ✅ Integrated |
| `fno_vanilla` | Constant-resolution FNO baseline | ✅ Integrated |
| `neuralop_uno` | Locked Lines128 adapter for `neuraloperator` U-NO | ✅ Integrated |

All selectable generator architectures in this package train through `PtychoPINN_Lightning` with the same physics loss and stitching behavior. Study-specific supervised adapters that reuse generator components define their own `model(x) -> y` channel contract.

## Architecture Details

### CNN (default)
The default CNN architecture uses a U-Net encoder-decoder with physics-informed forward model. See `ptycho_torch/model.py` for implementation.

**Output mode (`ModelConfig.cnn_output_mode`, Task 2.3 / backlog B1):** `Literal['amp_phase', 'real_imag'] = 'amp_phase'`.

- `'amp_phase'` (default, unchanged): separate amplitude head (`Amplitude_activation`) and phase head (`pi*tanh`), combined as `amp * exp(1j*phase)`. No representability ceiling.
- `'real_imag'` (opt-in, **Unsupervised-only** — resolution is centralized in `_effective_cnn_output_mode()`; Supervised mode always resolves to `'amp_phase'` regardless of this knob, so the supervised path and its tests are unaffected): the Autoencoder emits a `(real, imag)` tuple, each `(B, C, H, W)`, combined via `torch.complex(real, imag)` in `_predict_complex_patches()`. This is a **different adapter branch** from the FNO/Hybrid `real_imag` tensor path below (tuple vs. single tensor) — see "Integration Contract". The heads carry main's hardwired `ScaledTanh` box in `ptycho_torch.model.ScaledTanh`: real via `tanh + 0.2` (range `(-0.8, 1.2)`), imag via `1.2 * tanh` (range `(-1.2, 1.2)`). This is a **hard representability constraint**: a unit-amplitude object at `|phase| -> pi` maps to `real ~ -1`, below the `-0.8` floor, so it cannot be represented. On the amplitude forward, use `'amp_phase'` for high-phase-contrast objects; `rectangular_scaled` requires effective `real_imag`.

The amplitude forward accepts both output modes. The rectangular-scaled CI
forward requires effective `real_imag` output because its `s1`/`s2` factors
scale the generator's directly learned normalized object-plane real and
imaginary textures before probe multiplication and the FFT. Converting learned
amplitude/phase outputs to a complex tensor does not satisfy that direct-output
factorization. Compatibility is checked after the effective CNN or registered-
generator output mode is resolved.

### FNO (Cascaded FNO)
The FNO architecture (`architecture='fno'`) uses a cascaded design:
1. Spatial lifter (3x3 convs)
2. Fourier Neural Operator blocks (spectral convolutions)
3. CNN refiner blocks (3x3 convs)
4. Output projection to real/imag format

**Key parameters:**
- `fno_blocks`: Number of FNO blocks (default: 4)
- `fno_cnn_blocks`: Number of CNN refiner blocks (default: 2)
- `fno_modes`: Spectral modes (default: min(12, N//4))

### FNO Vanilla (constant-resolution)
The FNO Vanilla architecture (`architecture='fno_vanilla'`) removes down/upsampling entirely:
1. Spatial lifter (3×3 convs)
2. Constant‑resolution FNO block stack
3. 1×1 output projection


### Placement-block arms
`fno_li` provides a Li-style FNO with a pointwise linear path; `vit` provides an isotropic ViT with `vit_patch_size`, `vit_width`, `vit_depth`, and `vit_heads` controls. `ffno_encoder_share_weights=False` selects unshared FFNO weights.
### NeuralOperator U-NO (locked Lines128 CDI adapter)
The `neuralop_uno` architecture wraps the external `neuralop.models.UNO` implementation behind the existing CDI generator contract.

Current scope is intentionally narrow:
- requires external `neuraloperator==2.0.0`
- supports only the locked Lines128 CDI lane (`N=128`, `gridsize=1`, `C=1`)
- supports only `generator_output_mode='real_imag'`
- validates that raw UNO output is exactly `(B, 2, 128, 128)` before adapting to `(B, H, W, 1, 2)`

### FFNO (constant-resolution factorized Fourier flow)
The FFNO architecture (`architecture='ffno'`) keeps the constant-resolution CDI shell but swaps the spectral stack to factorized Fourier operators:
1. Spatial lifter (3×3 convs)
2. Constant‑resolution FFNO block stack
3. Optional local residual refiners controlled by `fno_cnn_blocks`
4. 1×1 output projection

**Key parameters:**
- `fno_blocks`: Number of FFNO blocks (default: 4)
- `fno_cnn_blocks`: Number of local residual refiners after the FFNO stack
  (default: 2). Set `0` for paper-facing pure FFNO comparisons. Positive
  values define an FFNO-local-refiner proxy, not the canonical no-refiner FFNO
  row.
- `fno_modes`: Spectral modes per axis (default: 12)

## Integration Contract

The registered non-CNN generators integrate with `PtychoPINN_Lightning` via:

1. **Output format**: Generators output `(B, H, W, C, 2)` real/imag tensor
2. **Adapter function**: `_real_imag_to_complex_channel_first()` converts to `(B, C, H, W)` complex
3. **Physics pipeline**: The complex patches flow through `ForwardModel` for physics loss
4. **Stitching**: Same TF reassembly helper as CNN (no stitching changes)

The CNN generator's opt-in `cnn_output_mode='real_imag'` path (see "CNN (default)" above)
uses the **same** `generator_output="real_imag"` contract name inside
`_predict_complex_patches()`, but a **different input shape**: a `(real, imag)` tuple of
`(B, C, H, W)` tensors, not the non-CNN `(B, H, W, C, 2)` single tensor. Both branches
combine to `torch.complex` and share everything downstream (physics pipeline, stitching);
only the adapter's tuple-vs-tensor dispatch differs (`ptycho_torch.model._predict_complex_patches`).

## Adding a New Generator

Follow the [Custom PyTorch CDI Architecture
Guide](../../docs/workflows/custom_torch_architecture.md) for the single
extension recipe, output layouts, construction, and checkpoint replay checks.

## Output Format Options

Generators can use these output formats:

| Format | Shape | Description |
|--------|-------|-------------|
| `amp_phase` | Two tensors: `(B, C, H, W)` each | Amplitude and phase channels (CNN default; also the only Supervised-mode contract) |
| `amp_phase_logits` | Single tensor: `(B, H, W, C, 2)` | Amplitude and phase logits in the last dimension; the shared adapter applies `sigmoid` to amplitude, `pi*tanh` to phase, then combines them as complex object patches |
| `real_imag` (tensor) | Single tensor: `(B, H, W, C, 2)` | Real and imaginary parts in last dimension (FNO/Hybrid) |
| `real_imag` (tuple) | Two tensors: `(B, C, H, W)` each, `(real, imag)` | CNN opt-in (`cnn_output_mode='real_imag'`, Unsupervised-only, Task 2.3 / backlog B1) |

The `generator_output` parameter in `PtychoPINN_Lightning` controls which adapter path is
used. `_predict_complex_patches()` converts `amp_phase_logits` with the transforms above
and dispatches `real_imag` to the tuple or tensor branch based on the generator's actual
return type (`isinstance(patches, (tuple, list))`).

## PyTorch-Specific Considerations

- Learned modules are `nn.Module` bodies; the existing application factory
  constructs the Lightning model and owns checkpoint configuration.
- Models should handle channel ordering (PyTorch uses NCHW, TensorFlow uses NHWC)
- See `ptycho_torch/model.py` for the reference `PtychoPINN_Lightning` implementation

## See Also

- `ptycho/config/config.py`: ModelConfig with architecture field
- `ptycho_torch/application_factory.py`: single application-construction boundary
- `ptycho_torch/workflows/components.py`: training, persistence, and reconstruction workflow
- `ptycho_torch/model.py`: PtychoPINN_Lightning implementation
- `ptycho_torch/generators/fno.py`: FNO implementation
- `docs/workflows/custom_torch_architecture.md`: end-to-end custom architecture guide
- `docs/workflows/pytorch.md`: PyTorch workflow documentation
