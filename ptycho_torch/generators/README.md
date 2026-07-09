# PyTorch Generator Registry

This module provides a central registry for generator architectures used in PyTorch Lightning PINN models.

## Overview

The generator registry enables architecture selection via the `config.model.architecture` field. The registry (`ptycho_torch.generators.registry._REGISTRY`, 14 entries) and its authoritative enumeration in `docs/specs/spec-ptycho-config-bridge.md` §3 are the source of truth for the full architecture list; the table below is illustrative only.

| Architecture | Description | Status |
|--------------|-------------|--------|
| `cnn` (default) | U-Net based CNN generator | ✅ Integrated |
| `fno` | Cascaded FNO + CNN refiner (Arch A) | ✅ Integrated |

All registered generator architectures in this package train through `PtychoPINN_Lightning` with the same physics loss and stitching behavior. Study-specific supervised adapters that reuse generator components live outside this registry and define their own `model(x) -> y` channel contract.

## Architecture Details

### CNN (default)
The default CNN architecture uses a U-Net encoder-decoder with physics-informed forward model. See `ptycho_torch/model.py` for implementation.

**Output mode (`ModelConfig.cnn_output_mode`, Task 2.3 / backlog B1):** `Literal['amp_phase', 'real_imag'] = 'amp_phase'`.

- `'amp_phase'` (default, unchanged): separate amplitude head (`Amplitude_activation`) and phase head (`pi*tanh`), combined as `amp * exp(1j*phase)`. No representability ceiling.
- `'real_imag'` (opt-in, **Unsupervised-only** — resolution is centralized in `_effective_cnn_output_mode()`; Supervised mode always resolves to `'amp_phase'` regardless of this knob, so the supervised path and its tests are unaffected): the Autoencoder emits a `(real, imag)` tuple, each `(B, C, H, W)`, combined via `torch.complex(real, imag)` in `_predict_complex_patches()`. This is a **different adapter branch** from the FNO `real_imag` tensor path below (tuple vs. single tensor) — see "Integration Contract". The heads carry main's hardwired `ScaledTanh` box in `ptycho_torch.model.ScaledTanh`: real via `tanh + 0.2` (range `(-0.8, 1.2)`), imag via `1.2 * tanh` (range `(-1.2, 1.2)`). This is a **hard representability constraint**: a unit-amplitude object at `|phase| -> pi` maps to `real ~ -1`, below the `-0.8` floor, so it cannot be represented. Use `'amp_phase'` for high-phase-contrast objects.

**Why real/imag only, not amp/phase, for the rectangular scaling mode:** `s1`/`s2` (see `ModelConfig.physics_forward_mode='rectangular_scaled'`, `RectangularScaledDiffraction`) are a representation-space scaling — two independent scalars on the separable real and imaginary channels. The polar analogue of that scaling is a single amplitude scale; amp/phase-parameterized outputs have no two-degree-of-freedom decomposition to apply it to. `'real_imag'` output is therefore a prerequisite for `physics_forward_mode='rectangular_scaled'`, not an arbitrary pairing — see `docs/findings.md#RECTANGULAR-SCALED-001`.

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
- `fno_modes`: Spectral modes per axis (default: min(12, N//4))

## Integration Contract

All FNO generators integrate with `PtychoPINN_Lightning` via:

1. **Output format**: Generators output `(B, H, W, C, 2)` real/imag tensor
2. **Adapter function**: `_real_imag_to_complex_channel_first()` converts to `(B, C, H, W)` complex
3. **Physics pipeline**: The complex patches flow through `ForwardModel` for physics loss
4. **Stitching**: Same TF reassembly helper as CNN (no stitching changes)

The CNN generator's opt-in `cnn_output_mode='real_imag'` path (see "CNN (default)" above)
uses the **same** `generator_output="real_imag"` contract name inside
`_predict_complex_patches()`, but a **different input shape**: a `(real, imag)` tuple of
`(B, C, H, W)` tensors, not the FNO `(B, H, W, C, 2)` single tensor. Both branches
combine to `torch.complex` and share everything downstream (physics pipeline, stitching);
only the adapter's tuple-vs-tensor dispatch differs (`ptycho_torch.model._predict_complex_patches`).

## Adding a New Generator

1. **Create the generator module** in `ptycho_torch/generators/`:

```python
# ptycho_torch/generators/my_arch.py
class MyArchGenerator:
    """My new architecture generator."""
    name = 'my_arch'

    def __init__(self, config):
        """
        Initialize the generator.

        Args:
            config: TrainingConfig or InferenceConfig with model settings
        """
        self.config = config

    def build_model(self, pt_configs):
        """
        Build the Lightning module for training.

        Args:
            pt_configs: Dict containing PyTorch config objects:
                - model_config: PTModelConfig
                - data_config: PTDataConfig
                - training_config: PTTrainingConfig
                - inference_config: PTInferenceConfig

        Returns:
            PtychoPINN_Lightning or compatible Lightning module
        """
        from ptycho_torch.model import PtychoPINN_Lightning

        # Build your core generator module
        core = MyArchModule(...)

        # Wrap in Lightning with physics pipeline
        return PtychoPINN_Lightning(
            model_config=pt_configs['model_config'],
            data_config=pt_configs['data_config'],
            training_config=pt_configs['training_config'],
            inference_config=pt_configs['inference_config'],
            generator_module=core,
            generator_output="real_imag",  # or "amp_phase"
        )
```

2. **Register the generator** in `ptycho_torch/generators/registry.py`:

```python
from ptycho_torch.generators.my_arch import MyArchGenerator

_REGISTRY = {
    'cnn': CnnGenerator,
    'fno': FnoGenerator,
    'my_arch': MyArchGenerator,  # Add your generator
}
```

3. **Add validation** in `ptycho/config/config.py`:

Update the `ModelConfig.architecture` type hint and `validate_model_config()`:

```python
architecture: Literal['cnn', 'fno', 'my_arch'] = 'cnn'
```

4. **Add tests** in `tests/torch/test_generator_registry.py`:

```python
def test_resolve_generator_my_arch():
    cfg = TrainingConfig(model=ModelConfig(architecture='my_arch'))
    gen = resolve_generator(cfg)
    assert gen.name == 'my_arch'
```

5. **Update documentation**:

- Add entry to this README
- Document architecture-specific parameters in `docs/CONFIGURATION.md`
- Update `docs/workflows/pytorch.md`

## API Contract

All generators must:

1. Have a `name` class attribute matching the registry key
2. Accept a config object in `__init__`
3. Implement `build_model(pt_configs)` returning a Lightning module
4. The Lightning module should be compatible with the trainer in `_train_with_lightning`

## Output Format Options

Generators can use these output formats:

| Format | Shape | Description |
|--------|-------|-------------|
| `amp_phase` | Two tensors: `(B, C, H, W)` each | Amplitude and phase channels (CNN default; also the only Supervised-mode contract) |
| `real_imag` (tensor) | Single tensor: `(B, H, W, C, 2)` | Real and imaginary parts in last dimension (FNO) |
| `real_imag` (tuple) | Two tensors: `(B, C, H, W)` each, `(real, imag)` | CNN opt-in (`cnn_output_mode='real_imag'`, Unsupervised-only, Task 2.3 / backlog B1) |

The `generator_output` parameter in `PtychoPINN_Lightning` controls which adapter path is
used; `_predict_complex_patches()` dispatches `real_imag` to the tuple or tensor branch
based on the generator's actual return type (`isinstance(patches, (tuple, list))`).

## PyTorch-Specific Considerations

- Generators should return Lightning modules compatible with `L.Trainer.fit()`
- The returned model should support `save_hyperparameters()` for checkpoint compatibility
- Models should handle channel ordering (PyTorch uses NCHW, TensorFlow uses NHWC)
- See `ptycho_torch/model.py` for the reference `PtychoPINN_Lightning` implementation

## See Also

- `ptycho/config/config.py`: ModelConfig with architecture field
- `ptycho_torch/workflows/components.py`: Workflow integration via `resolve_generator`
- `ptycho_torch/model.py`: PtychoPINN_Lightning implementation
- `ptycho_torch/generators/fno.py`: FNO implementation
- `docs/workflows/pytorch.md`: PyTorch workflow documentation
