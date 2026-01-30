# PyTorch Generator Registry

This module provides a central registry for generator architectures used in PyTorch Lightning PINN models.

## Overview

The generator registry enables architecture selection via the `config.model.architecture` field. Currently supported architectures:

| Architecture | Description | Status |
|--------------|-------------|--------|
| `cnn` | U-Net based CNN generator (default) | ✅ Integrated |
| `fno` | Cascaded FNO + CNN refiner (Arch A) | ✅ Integrated |
| `hybrid` | Hybrid U-NO (Arch B) | ✅ Integrated |

All architectures train through `PtychoPINN_Lightning` with the same physics loss and stitching behavior.

## Architecture Details

### CNN (default)
The default CNN architecture uses a U-Net encoder-decoder with physics-informed forward model. See `ptycho_torch/model.py` for implementation.

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

### Hybrid (U-NO)
The Hybrid architecture (`architecture='hybrid'`) combines U-Net with FNO:
1. Encoder path with downsampling + FNO blocks
2. Bottleneck with spectral convolution
3. Decoder path with upsampling + skip connections

**Key parameters:**
- `fno_blocks`: Number of FNO blocks per level (default: 4)
- `fno_modes`: Spectral modes (default: min(12, N//4))

## Integration Contract

All FNO/Hybrid generators integrate with `PtychoPINN_Lightning` via:

1. **Output format**: Generators output `(B, H, W, C, 2)` real/imag tensor
2. **Adapter function**: `_real_imag_to_complex_channel_first()` converts to `(B, C, H, W)` complex
3. **Physics pipeline**: The complex patches flow through `ForwardModel` for physics loss
4. **Stitching**: Same TF reassembly helper as CNN (no stitching changes)

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
    'hybrid': HybridGenerator,
    'my_arch': MyArchGenerator,  # Add your generator
}
```

3. **Add validation** in `ptycho/config/config.py`:

Update the `ModelConfig.architecture` type hint and `validate_model_config()`:

```python
architecture: Literal['cnn', 'fno', 'hybrid', 'my_arch'] = 'cnn'
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

Generators can use two output formats:

| Format | Shape | Description |
|--------|-------|-------------|
| `amp_phase` | Two tensors: `(B, C, H, W)` each | Amplitude and phase channels (CNN default) |
| `real_imag` | Single tensor: `(B, H, W, C, 2)` | Real and imaginary parts in last dimension (FNO/Hybrid) |

The `generator_output` parameter in `PtychoPINN_Lightning` controls which adapter path is used.

## PyTorch-Specific Considerations

- Generators should return Lightning modules compatible with `L.Trainer.fit()`
- The returned model should support `save_hyperparameters()` for checkpoint compatibility
- Models should handle channel ordering (PyTorch uses NCHW, TensorFlow uses NHWC)
- See `ptycho_torch/model.py` for the reference `PtychoPINN_Lightning` implementation

## See Also

- `ptycho/config/config.py`: ModelConfig with architecture field
- `ptycho_torch/workflows/components.py`: Workflow integration via `resolve_generator`
- `ptycho_torch/model.py`: PtychoPINN_Lightning implementation
- `ptycho_torch/generators/fno.py`: FNO and Hybrid implementations
- `docs/workflows/pytorch.md`: PyTorch workflow documentation
- `docs/backlog/FNO_HYBRID_FULL_INTEGRATION.md`: Integration plan and status
