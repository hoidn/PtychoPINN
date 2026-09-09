# Adding a PyTorch CDI Generator Architecture

Use this guide for a learned inverse network that consumes conditioned CDI
diffraction and returns complex object patches while reusing PtychoPINN's probe,
forward physics, losses, persistence, and reassembly. Generic supervised, PDE,
or image-to-image models belong behind task-specific adapters.

Direct `nn.Module` injection into `PtychoPINN` is useful only for a disposable
spike: the artifact cannot reconstruct an injected module. A saved architecture
must be selectable by the core builder, represented in configuration, and sealed in `ModelSpec`.
Arbitrary import-path plugins are not supported.

## Start here

Adding a generator means adding a network to the existing PyTorch backend,
not implementing another training pipeline. The shortest supported path is:

1. [Implement an `nn.Module`](#2-implement-the-module) with the existing
   input/output contract.
2. [Declare its architecture name](#31-configuration-and-strict-resolution)
   once in the shared Literal; no wrapper class or registry entry is needed.
3. [Add a core-builder branch](#32-core-builder) so training and reload can
   construct the same module from configuration.
4. [Persist topology settings](#33-modelspec-and-artifact-migration).
   Reusing suitable existing fields avoids adding schema fields; new topology
   fields need validation, configuration wiring, and a versioned upgrade.
5. [Verify the lifecycle](#4-required-verification): construction, gradients,
   short training, save, fresh-process reload, and inference.

Configuration and checkpoint compatibility are the main integration work
beyond the network itself. Do not reuse an unrelated field merely to avoid
adding a correctly owned topology setting.

### Generator or framework backend?

A new CNN, operator network, or transformer implemented in PyTorch follows
this guide. A genuinely new framework backend (for example, JAX) does not:
it needs a separate design for training, physics, data integration, persistence,
and inference. Define those framework boundaries before implementation. A task-specific supervised or PDE model is likewise not a CDI
generator merely because it reuses one of these network modules.

## 1. Contract and Ownership

The construction path is:

```text
public config + ExecutionRequest
  -> strict resolution
  -> Torch configs + ModelSpec
  -> application factory
  -> PtychoPINN_Lightning
  -> core builder -> generator module
  -> shared physics, loss, and reassembly
```

The generator owns only the learned diffraction-to-object map. Do not add a
private probe model, diffraction operator, loss, optimizer, reassembly policy,
or unsealed input normalization. Changes to `ptycho/model.py`, `ptycho/diffsim.py`, or
`ptycho/tf_helper.py` are separate physics-contract changes.

| Concern | Location |
|---|---|
| Shared architecture names | `ptycho/_architecture_names.py::_Architecture` |
| Public architecture and fields | `ptycho/config/config.py::ModelConfig` |
| Resolved Torch architecture and fields | `ptycho_torch/config_params.py::ModelConfig` |
| Public/Torch translation | `ptycho_torch/config_bridge.py`, `ptycho_torch/config_factory.py` |
| Strict architecture domain and patch fields | `ptycho_torch/config_resolution.py::SUPPORTED_TORCH_ARCHITECTURES`, `_TRAINING_INPUTS_BY_OWNER` |
| Application composition | `ptycho_torch/application_factory.py` |
| Core module construction | `ptycho_torch/model.py::_build_generator_module_from_config` |
| Complex output adaptation | `ptycho_torch/model.py::_predict_complex_patches` |
| Persisted structural identity | `ptycho_torch/model_spec.py` |
| Training and bundle loading | `ptycho_torch/workflows/components.py` |
| Inference | `ptycho_torch/inference.py` |

Advanced callers with four Torch config sections use
`build_ptychopinn_from_configs()`, which derives `ModelSpec` and enters the
shared application factory. Resolved training enters
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

## 2. Implement the Module

Use architecture-specific names for new fields. This illustrative architecture
adds `tiny_residual_width` and `tiny_residual_blocks`; it is not an installed
model. The examples require the configuration and persistence wiring below
before training or reload will work.

```python
# ptycho_torch/generators/tiny_residual.py
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
```

`TinyResidualGeneratorModule` is the trainable network. No wrapper class or
registry entry is needed; training and reload use the same application/core builder.

## 3. Select and Persist the Architecture

### 3.1 Configuration and strict resolution

Add the architecture name in step 1. Steps 2–6 apply when introducing new
topology settings; touch public/Torch joins only for shared fields:

1. Add the name once to `ptycho/_architecture_names.py::_Architecture`.
   Both ModelConfigs and the resolver use that domain.
2. Add new topology fields to `ptycho_torch/config_params.py::ModelConfig`
   and validate their domains. Add shared fields to the public ModelConfig only
   when the approved field ownership requires them.
3. Map shared fields in `ptycho_torch/config_bridge.py`; keep Torch-only fields
   on their Torch owner. Do not add architecture branches to the trainer.
4. Add each new topology patch name to the `model` owner in
   `_TRAINING_INPUTS_BY_OWNER`; `TRAINING_INPUT_RULES` is declared from this
   explicit allowlist.
5. Add public/Torch joins to
   `ptycho_torch/model_spec.py::_CANONICAL_TO_TORCH`.
6. Verify that supported explicit overrides accept the new fields.

The resolver derives architecture values from the shared Literal, while patch
names remain an explicit allowlist. Topology belongs to `ModelConfig`, not
`TrainingConfig`, `ExecutionRequest`, or `PyTorchExecutionConfig`.

Update the exact architecture domain in
`specs/ptychodus_api_spec.md`. Search maintained docs, designs, catalogs, and
tests for duplicated architecture literals or counts; update current
restatements and leave explicitly historical records unchanged.

### 3.2 Core builder

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
            model_config.learned_input_channels * data_config.gridsize**2
        ),
        component_channels=data_config.gridsize**2,
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
bump. Read `CURRENT_MODEL_SPEC_VERSION` and its explicit field sets in
`ptycho_torch/model_spec.py`; do not infer the current era from old examples.
Preserve frozen historical schemas and add explicit upgrades through the
existing model-spec/artifact/checkpoint codecs. Keep missing/unknown-field
rejection and shared-field agreement checks. The current checkpoint path seals
identity; do not reintroduce dual-written config dictionaries.

Do not read migration values from current dataclass defaults. The enclosing
artifact schema changes only if its own envelope or section semantics change;
the nested `ModelSpec` version still changes when its structural field set
changes. Use a new architecture ID or explicit migration when an existing ID's
state-dict topology becomes incompatible.

## 4. Required Verification

### Module and adapter

```python
module = module.to("cuda")
x = torch.randn(2, input_channels, 64, 64, device="cuda")
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

- `tests/torch/test_config_resolution_internal_transaction.py`: exact
  architecture domain and model-owned patch fields; keep test names
  count-neutral.
- `tests/torch/test_construction_consolidation.py`: every supported architecture
  constructs through the config and sealed-identity application paths.
- `tests/torch/test_generator_adapter.py`: only when adaptation changes.
- `tests/torch/test_config_bridge.py`: public/Torch agreement.
- `tests/torch/test_model_spec_v2.py`: structural identity.
- `tests/torch/test_lightning_checkpoint.py`: strict checkpoint reload with no
  manual kwargs.

Select claim-matched checks from these modules. Run CPU-only checks with
`python -m pytest -n 8 --dist loadfile`; model forwards/training use CUDA serially.
When identity changes, test exact old/current schemas and strict bundle reload.

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

### Integration success versus reconstruction quality

A passing lifecycle test proves the new architecture is usable and reloadable,
not that it reconstructs well. For a quality claim, reconstruct a complete
same-object acquisition and use the established SSIM/MAE evaluator.
Do not calibrate the new architecture against another model's quality fixture
or adjust existing models' thresholds to make an architecture addition pass.
Recalibration requires a separately approved protocol.

## 6. Completion Checklist

- [ ] Input/output shapes and complex adaptation are exact for every supported
  `C`.
- [ ] Shared name declaration, config, strict resolver, bridge, and core
  builder contain the architecture and topology fields.
- [ ] Maintained specs, docs, catalogs, and exact-domain tests agree.
- [ ] Config and sealed construction use one application path and have the
  same state-dict signature.
- [ ] `ModelSpec`, checkpoint compatibility, and artifact codecs preserve exact
  old/current schemas.
- [ ] Checkpoint-only and bundle-only reload need no injected module or config.
- [ ] A short train-save-reload-infer lifecycle passes.
- [ ] DDP has a two-process mmap-path smoke test if support is claimed.

## References

- [Configuration Guide](../CONFIGURATION.md)
- [PyTorch Workflow](pytorch.md)
- [Ptychodus API Specification](../../specs/ptychodus_api_spec.md)
