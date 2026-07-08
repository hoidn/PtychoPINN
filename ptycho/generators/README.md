# TensorFlow Generator Registry

This module provides a central registry for generator architectures used in PINN models.

## Overview

The generator registry enables architecture selection via the `config.model.architecture` field. Currently supported TensorFlow architectures:

- `cnn` (default): U-Net based CNN generator from `ptycho/model.py`

This README covers the TensorFlow generator registry. Other architecture
strings (e.g., `fno`, `hybrid`, `hybrid_resnet`) are handled by the
PyTorch stack under `ptycho_torch/` and are not registered here—attempting
to resolve them in the TensorFlow registry will raise a ValueError.

## Adding a New Generator

1. **Create the generator module** in `ptycho/generators/`:

```python
# ptycho/generators/my_arch.py
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

    def build_models(self):
        """
        Build the model for training.

        Returns:
            Tuple of (model_instance, diffraction_to_obj) where:
            - model_instance: The trainable Keras model
            - diffraction_to_obj: The inference model for reconstruction

        Note:
            Requires update_legacy_dict(params.cfg, config) to have been
            called before invocation to ensure params.cfg is populated.
        """
        # Your model creation logic here
        pass
```

2. **Register the generator** in `ptycho/generators/registry.py`:

```python
from ptycho.generators.my_arch import MyArchGenerator

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

4. **Add tests** in `tests/test_generator_registry.py`:

```python
def test_resolve_generator_my_arch():
    cfg = TrainingConfig(model=ModelConfig(architecture='my_arch'))
    gen = resolve_generator(cfg)
    assert gen.name == 'my_arch'
```

5. **Update documentation**:

- Add entry to this README
- Document architecture-specific parameters in `docs/CONFIGURATION.md`
- Update `docs/workflows/pytorch.md` only if you also add a PyTorch generator

## Naming Conventions

- Generator runs are labeled `pinn_<arch>` in grid-lines outputs (e.g., `pinn_cnn` for TensorFlow).
- The supervised baseline is labeled `baseline` (never aliased with `cnn`).

## API Contract

All generators must:

1. Have a `name` class attribute matching the registry key
2. Accept a config object in `__init__`
3. Implement `build_models()` returning `(model_instance, diffraction_to_obj)`
4. Work with params.cfg already populated (CONFIG-001 compliance)

## See Also

- `ptycho/config/config.py`: ModelConfig with architecture field
- `ptycho/workflows/components.py`: Workflow integration via `resolve_generator`
- `ptycho/model.py`: CNN model implementation
- `docs/CONFIGURATION.md`: Configuration documentation
- `docs/workflows/pytorch.md`: PyTorch workflow documentation
