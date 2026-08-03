# Adding a PyTorch CDI Generator Architecture

Use this guide for a learned inverse network that consumes conditioned CDI
diffraction and returns complex object patches while reusing PtychoPINN's probe,
forward physics, losses, persistence, and reassembly. Generic supervised, PDE,
or image-to-image models belong behind task-specific adapters.

Direct `nn.Module` injection into `PtychoPINN` is useful only for a disposable
spike: the artifact cannot reconstruct an injected module. A saved architecture
must be registered, represented in configuration, and sealed in `ModelSpec`.
Arbitrary import-path plugins are not supported.

## 1. Contract and Ownership

The construction path is:

```text
public config + ExecutionRequest
  -> strict resolution
  -> Torch configs + ModelSpec
  -> application factory
  -> PtychoPINN_Lightning
  -> registered generator
  -> shared physics, loss, and reassembly
```

The generator owns only the learned diffraction-to-object map. Do not add a
private probe model, diffraction operator, loss, optimizer, reassembly policy,
or unsealed input normalization. Changes to `ptycho/model.py`, `ptycho/diffsim.py`, or
`ptycho/tf_helper.py` are separate physics-contract changes.

| Concern | Location |
|---|---|
| Public architecture and fields | `ptycho/config/config.py::ModelConfig` |
| Resolved Torch architecture and fields | `ptycho_torch/config_params.py::ModelConfig` |
| Public/Torch translation | `ptycho_torch/config_bridge.py`, `ptycho_torch/config_factory.py` |
| Strict architecture domain and patch fields | `ptycho_torch/config_resolution.py::SUPPORTED_TORCH_ARCHITECTURES`, `_TRAINING_INPUTS_BY_OWNER` |
| Registry | `ptycho_torch/generators/registry.py` |
| Application composition | `ptycho_torch/application_factory.py` |
| Core module construction | `ptycho_torch/model.py::_build_generator_module_from_config` |
| Complex output adaptation | `ptycho_torch/model.py::_predict_complex_patches` |
| Persisted structural identity | `ptycho_torch/model_spec.py` |
| Training and bundle loading | `ptycho_torch/workflows/components.py` |
| Inference | `ptycho_torch/inference.py` |

The registry is a name catalog, not a second constructor. Registry wrappers
delegate to `build_ptychopinn_from_configs()`, which derives `ModelSpec` and
enters the shared application factory. Resolved training enters
`build_ptychopinn_application()` directly. Both routes reach the same core
builder.

### Tensor contract

Input is a real floating tensor:

```text
(B, input_channels, H, W)
```

For the ordinary path, `H = W = N`. The adapter may fold the semantic
`C = gridsize^2` component axis and configured conditioning into
`input_channels`; do not squeeze or infer it away. Probe, positions, and scale
state remain outside the learned forward.

Supported outputs are:

| `generator_output_mode` | Return value |
|---|---|
| `real_imag` | Tensor `(B,H,W,C,2)`. The separate `(real, imag)` tuple is a CNN compatibility form. |
| `amp_phase` | Tuple `(amplitude, phase)`, each `(B,C,H,W)`. |
| `amp_phase_logits` | Tensor `(B,H,W,C,2)`; the shared adapter applies the amplitude and phase activations. |

Prefer `real_imag` for a new unsupervised architecture. It is required by the
`rectangular_scaled` CI forward path. `_predict_complex_patches()` converts it
to complex `(B,C,H,W)`. Do not return `(B,2*C,H,W)` and rely on downstream shape
guessing.

## 2. Implement the Module and Wrapper

Use architecture-specific field names. This minimal example adds
`tiny_residual_width` and `tiny_residual_blocks`:

```python
# ptycho_torch/generators/tiny_residual.py
from typing import Any

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class TinyResidualGeneratorModule(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        component_channels: int,
        width: int,
        blocks: int,
        output_mode: str,
    ):
        super().__init__()
        if width <= 0 or blocks <= 0:
            raise ValueError("tiny_residual width and blocks must be positive")
        if output_mode != "real_imag":
            raise ValueError("tiny_residual requires real_imag output")
        self.input_channels = int(input_channels)
        self.component_channels = int(component_channels)
        self.stem = nn.Conv2d(self.input_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.head = nn.Conv2d(width, 2 * self.component_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f"unexpected input shape {tuple(x.shape)}")
        batch, _, height, width = x.shape
        raw = self.head(self.blocks(self.stem(x)))
        return (
            raw.view(batch, 2, self.component_channels, height, width)
            .permute(0, 3, 4, 2, 1)
            .contiguous()
        )


class TinyResidualGenerator:
    name = "tiny_residual"

    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs: dict[str, Any]) -> nn.Module:
        from ptycho_torch.application_factory import build_ptychopinn_from_configs

        return build_ptychopinn_from_configs(pt_configs)
```

`TinyResidualGeneratorModule` is the trainable network.
`TinyResidualGenerator` is only the registry adapter: `resolve_generator()`
selects it from `config.model.architecture`, then its `build_model()` delegates
the complete Torch config bundle to the application factory. The wrapper must
not construct the module or `PtychoPINN_Lightning` directly; training and reload
must converge on the same application/core builder.

## 3. Register and Persist the Architecture

### 3.1 Configuration and strict resolution

Update all of these surfaces:

1. In `ptycho/config/config.py::ModelConfig`, add the architecture literal and
   topology fields; update `validate_model_config()`.
2. Add the same literal and fields to
   `ptycho_torch/config_params.py::ModelConfig`; validate their domains in its
   `__post_init__()`.
3. Map shared fields in `ptycho_torch/config_bridge.py` and forward them through
   `ptycho_torch/workflows/components.py::_train_with_lightning`.
4. Add the architecture value to
   `ptycho_torch/config_resolution.py::SUPPORTED_TORCH_ARCHITECTURES`.
5. Add each new topology patch name to the `model` owner in
   `_TRAINING_INPUTS_BY_OWNER`; `TRAINING_INPUT_RULES` is declared from this
   explicit allowlist.
6. Add public/Torch joins to
   `ptycho_torch/model_spec.py::_CANONICAL_TO_TORCH`.
7. Verify that supported explicit overrides accept the new fields.

Architecture values and patch names are separate explicit resolver domains;
neither is derived from the dataclasses. Topology belongs to `ModelConfig`, not
`TrainingConfig`, `ExecutionRequest`, or `PyTorchExecutionConfig`.

Update the exact architecture domain in the public/Torch literals and
`specs/ptychodus_api_spec.md`. Search maintained docs, designs, catalogs, and
tests for duplicated architecture literals or counts; update current
restatements and leave explicitly historical records unchanged.

### 3.2 Registry and core builder

Register the wrapper in `ptycho_torch/generators/registry.py`:

```python
from ptycho_torch.generators.tiny_residual import TinyResidualGenerator

_REGISTRY = {
    # existing entries...
    "tiny_residual": TinyResidualGenerator,
}
```

The registry key, wrapper `name`, public literal, Torch literal, and strict
resolver value must match.

Add the module branch to
`ptycho_torch/model.py::_build_generator_module_from_config`:

```python
if architecture == "tiny_residual":
    from ptycho_torch.generators.tiny_residual import (
        TinyResidualGeneratorModule,
    )

    if generator_mode != "real_imag":
        raise ValueError("tiny_residual requires generator_output_mode='real_imag'")
    return TinyResidualGeneratorModule(
        input_channels=(
            int(model_config.learned_input_channels) * int(data_config.C)
        ),
        component_channels=int(data_config.C),
        width=int(model_config.tiny_residual_width),
        blocks=int(model_config.tiny_residual_blocks),
        output_mode=generator_mode,
    )
```

Construction may use only persisted model fields and explicit data join keys.
Do not require caller injection, mutable globals, or study-local defaults.

### 3.3 `ModelSpec` and artifact migration

Every field that changes module type, parameter count, tensor shape, or forward
topology must be in `ModelSpec`.

Adding only an architecture value does not change the field set because
`architecture` is already sealed. Adding topology fields does require a schema
bump. For the current portable-v2-to-portable-v3 case:

1. leave `PORTABLE_V1_MODEL_FIELDS` and `PORTABLE_V2_MODEL_FIELDS` unchanged;
2. add a portable-v3 field set and make
   `torch-model-spec-portable-v3` current;
3. retain exact v1/v2 decoders and upgrade with literal historical values;
4. reject missing and unknown fields;
5. update `derive_model_spec()`;
6. update `PtychoPINN_Lightning.__init__` so its dual-written legacy
   `model_config` receives the same explicit migration before comparison;
7. update `ptycho_torch/artifact_schema.py` field-shape classification and
   compatibility paths.

Do not read migration values from current dataclass defaults. The enclosing
artifact schema changes only if its own envelope or section semantics change;
the nested `ModelSpec` version still changes when its structural field set
changes. Use a new architecture ID or explicit migration when an existing ID's
state-dict topology becomes incompatible.

## 4. Required Verification

### Module and adapter

```python
x = torch.randn(2, input_channels, 64, 64)
y = module(x)
assert y.shape == (2, 64, 64, C, 2)
assert y.dtype == x.dtype
assert torch.isfinite(y).all()
y.square().mean().backward()
assert any(parameter.grad is not None for parameter in module.parameters())
```

Also test invalid channel counts, invalid topology values at Torch config,
resolution, and reload boundaries, unsupported output modes, and
`_predict_complex_patches()` returning finite complex `(B,C,N,N)`.

### Integration and reload

Required coverage:

- `tests/torch/test_config_resolution_transaction.py`: exact
  architecture domain and model-owned patch fields; keep test names
  count-neutral.
- `tests/torch/test_generator_registry.py`: focused name resolution.
- `tests/torch/test_construction_consolidation.py`: every `_REGISTRY` entry
  delegates to the application factory and matches sealed construction.
- `tests/torch/test_generator_adapter.py`: only when adaptation changes.
- `tests/torch/test_config_bridge.py`: public/Torch agreement.
- `tests/torch/test_model_spec.py`: structural identity.
- `tests/torch/test_lightning_checkpoint.py`: strict checkpoint reload with no
  manual kwargs.

Run the core selectors:

```bash
pytest \
  tests/torch/test_config_resolution_transaction.py \
  tests/torch/test_generator_registry.py \
  tests/torch/test_construction_consolidation.py \
  tests/torch/test_config_bridge.py \
  tests/torch/test_model_spec.py \
  tests/torch/test_lightning_checkpoint.py -q
```

When the `ModelSpec` field set changes, also update and run:

```bash
pytest \
  tests/torch/test_model_spec_v2.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_artifact_schema_v2.py \
  tests/torch/test_config_pydantic_artifacts.py \
  tests/torch/test_absolute_scaling_entrypoints.py -q
```

Search maintained code, tests, and fixtures for
`torch-model-spec-portable-v2`, `PORTABLE_V2_MODEL_FIELDS`,
`MODEL_SPEC_V2_MODEL_FIELDS`, and `CURRENT_MODEL_SPEC_VERSION`. Preserve frozen
v2 input checks; update assertions that treat v2 as the current produced schema.

## 5. Train, Reload, and Infer

Use the existing data loader, scale contract, `TrainingConfig`, and execution
entry point documented in the [PyTorch Workflow](pytorch.md). The architecture
does not choose data, loss, scaling, optimizer, or reassembly policy. Do not
inject the module into a run that is intended to prove persistence.

`run_cdi_example_torch()` consumes resolved configuration explicitly and does
not project the full configuration into `params.cfg`. Any surviving legacy leaf
must own a narrow `legacy_params_scope()` / `configured_params_scope()` bridge.

After a short run, require a checkpoint, `wts.h5.zip`, and persisted effective
architecture/topology values. Reload in a fresh process:

```python
from pathlib import Path

from ptycho_torch.workflows.components import load_inference_bundle_torch

models, _ = load_inference_bundle_torch(
    Path("outputs/tiny_residual_run_001")
)
model = models["diffraction_to_obj"]

assert model.model_config.architecture == "tiny_residual"
assert type(model.model.autoencoder).__name__ == "TinyResidualGeneratorModule"
assert model.model_config.tiny_residual_width == 32
assert model.model_config.tiny_residual_blocks == 4
```

No module construction or configuration kwargs may be supplied by the reload
caller.

For an artifact trained with the CI count-intensity profile, run inference with
the required full-scan VarPro route:

```bash
python -m ptycho_torch.inference \
  --model_path outputs/tiny_residual_run_001 \
  --test_data datasets/my_test.npz \
  --output_dir outputs/tiny_residual_run_001/inference \
  --patch-weighting probe \
  --varpro-scaling \
  --accelerator cuda \
  --quiet
```

Do not combine that route with `--n_images`. For non-CI artifacts, select
reassembly and scaling from the artifact's contract.

Add one fixed-input lifecycle regression covering short training, checkpoint
and bundle save, fresh reload, inference, and fresh-versus-reloaded output
agreement. If DDP support is claimed, add a two-process smoke test through the
established mmap/Lightning data path.

## 6. Completion Checklist

- [ ] Input/output shapes and complex adaptation are exact for every supported
  `C`.
- [ ] Public config, Torch config, strict resolver, bridge, registry, and core
  builder contain the architecture and topology fields.
- [ ] Maintained specs, docs, catalogs, and exact-domain tests agree.
- [ ] Registry and sealed construction use one application path and have the
  same state-dict signature.
- [ ] `ModelSpec`, checkpoint compatibility, and artifact codecs preserve exact
  old/current schemas.
- [ ] Checkpoint-only and bundle-only reload need no injected module or config.
- [ ] A short train-save-reload-infer lifecycle passes.
- [ ] DDP has a two-process mmap-path smoke test if support is claimed.

## References

- [Configuration Guide](../CONFIGURATION.md)
- [PyTorch Workflow](pytorch.md)
- [Data Normalization Guide](../DATA_NORMALIZATION_GUIDE.md)
- [Ptychodus API Specification](../../specs/ptychodus_api_spec.md)
