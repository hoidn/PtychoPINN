# PyTorch Generator Registry

This module provides a central registry for generator architectures used in PyTorch Lightning PINN models.

## Overview

The generator registry enables architecture selection via the `config.model.architecture` field. Currently supported architectures:

- `cnn` (default): U-Net based CNN generator from `ptycho_torch/model.py`
- `fno`: Reserved for Fourier Neural Operator (not yet implemented)
- `hybrid`: Reserved for hybrid CNN/FNO architecture (not yet implemented)

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
        # Your model creation logic here
        pass
```

2. **Register the generator** in `ptycho_torch/generators/registry.py`:

```python
from ptycho_torch.generators.my_arch import MyArchGenerator

_REGISTRY = {
    'cnn': CnnGenerator,
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

## PyTorch-Specific Considerations

- Generators should return Lightning modules compatible with `L.Trainer.fit()`
- The returned model should support `save_hyperparameters()` for checkpoint compatibility
- Models should handle channel ordering (PyTorch uses NCHW, TensorFlow uses NHWC)
- See `ptycho_torch/model.py` for the reference `PtychoPINN_Lightning` implementation

## See Also

- `ptycho/config/config.py`: ModelConfig with architecture field
- `ptycho_torch/workflows/components.py`: Workflow integration via `resolve_generator`
- `ptycho_torch/model.py`: PtychoPINN_Lightning implementation
- `docs/workflows/pytorch.md`: PyTorch workflow documentation
