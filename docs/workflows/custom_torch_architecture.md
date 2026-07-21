# Building a Custom PyTorch CDI Architecture

This guide shows how to add, configure, train, save, reload, and run inference
with a new learned reconstruction architecture. It is for a model that replaces
the usual CNN or Fourier-family inverse network while retaining PtychoPINN's
data handling, differentiable ptychographic forward model, losses, checkpoint
format, and image reassembly.

In this guide, **generator** means the learned inverse network

```text
conditioned diffraction  ->  predicted complex object patches
```

It does not mean the synthetic-data generator. Synthetic probe, object, scan,
and detector choices belong to `SimulationConfig`; see the
[Configuration Guide](../CONFIGURATION.md).

## 1. Decide Whether This Is the Right Extension Point

Use the registered CDI generator path when the new network:

- consumes a PtychoPINN diffraction batch;
- predicts one complex object patch per semantic component;
- should use the existing probe, differentiable diffraction physics, and
  Poisson or MAE training path;
- must be selectable from `ModelConfig` and reconstructible from a saved
  checkpoint or `wts.h5.zip` bundle.

Do not add a generic image-to-image, PDE, or supervised benchmark model to the
CDI registry merely to reuse its training loop. Such a model should have a
task-local `model(x) -> y` adapter unless it satisfies the CDI contract above.

There are two useful development levels:

| Level | Good for | Limitation |
|---|---|---|
| Inject an `nn.Module` directly into `PtychoPINN` | A local shape or gradient experiment | The saved artifact cannot reconstruct the architecture by itself. |
| Register and seal the architecture | Training, bundles, strict reload, inference, comparisons | Requires complete configuration and `ModelSpec` wiring. |

Use direct injection only as a disposable spike. Use the registered path for
any result that you intend to save, compare, or reproduce.

The current system does not accept an arbitrary module import path as a
drop-in plugin. A reloadable custom architecture is a source-level extension
to the package and must remain importable anywhere its artifacts are loaded.

## 2. Understand the End-to-End Boundary

The learned module is only one component of the application:

```text
public ModelConfig + TrainingConfig + PyTorchExecutionConfig
                              |
                              v
                    configuration factory
                              |
          Torch data/model/training/inference configs
                    + sealed ModelSpec
                              |
                              v
                  application factory
                              |
                              v
                  PtychoPINN_Lightning
                              |
       diffraction -> your generator -> complex patches
                              |
       training assembly -> probe/object multiplication -> FFT
                              |
                    predicted measurement
                              |
                         loss and update
```

The generator owns the learned map from conditioned diffraction to object
patches. It must not implement a second probe model, diffraction operator,
loss, optimizer, or stitching algorithm. Those remain shared so architecture
comparisons change the network rather than silently changing the experiment.
Adding a generator should not require changes to the stable core physics files
`ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`; such a need is
a separate physics-contract change.

The main implementation locations are:

| Concern | Location |
|---|---|
| Public user-facing model choices | `ptycho/config/config.py::ModelConfig` |
| Resolved Torch structural choices | `ptycho_torch/config_params.py::ModelConfig` |
| Public/Torch translation | `ptycho_torch/config_bridge.py` and `ptycho_torch/config_factory.py` |
| Architecture-name registry | `ptycho_torch/generators/registry.py` |
| Core module construction and checkpoint rebuild | `ptycho_torch/model.py::_build_generator_module_from_config` |
| Output adaptation to complex channel-first patches | `ptycho_torch/model.py::_predict_complex_patches` |
| Sealed structural identity | `ptycho_torch/model_spec.py` |
| Training orchestration | `ptycho_torch/workflows/components.py` |
| Bundle loading and inference | `ptycho_torch/workflows/components.py` and `ptycho_torch/inference.py` |

The registry wrapper and the core module builder are intentionally separate.
The wrapper selects an architecture at a public workflow boundary. The builder
constructs the actual `nn.Module` from saved structural state. Registering a
name without adding the builder branch can make a fresh run appear to work
while checkpoint-only reload fails.

## 3. Freeze the Input and Output Contract First

### Input

Your module receives a real floating tensor in Torch channel-first layout:

```text
(B, input_channels, H, W)
```

Its callable contract is `forward(x)`. Scan positions, probe tensors, and scale
factors are supplied to the shared application and physics path, not to the
generator. A proposed network that requires those values inside its learned
forward is a change to the generator/application interface, not an ordinary
new registry entry.

For the ordinary CDI path, `H = W = N`. The effective input channel count is
formed by the data adapter and may include the `C = gridsize^2` grouped-patch
axis and explicitly configured conditioning channels. Do not silently squeeze,
broadcast, or infer away this axis. Validate it in the constructor or first
forward call.

The tensor has already been conditioned according to the selected scaling
contract. Do not add an architecture-private count normalization unless it is
an explicit, sealed model transform; otherwise training and reloaded inference
will no longer share one input contract.

Start with `N=64`, `gridsize=1`, and one input channel while bringing up a new
architecture. Expand to grouped patches or extra conditioning only after the
single-patch contract passes.

### Recommended output

For a new unsupervised architecture, prefer the existing real/imaginary tensor
contract:

```text
(B, H, W, C, 2)
                  ^ real/imaginary axis
```

`_predict_complex_patches()` converts it to complex channel-first patches of
shape `(B, C, H, W)`. This representation is also required by the
`rectangular_scaled` CI forward model.

The other supported output contracts are:

| `generator_output_mode` | Module return value |
|---|---|
| `real_imag` | One tensor `(B,H,W,C,2)`; the separate `(real, imag)` tuple form is the CNN compatibility path. |
| `amp_phase` | A tuple `(amplitude, phase)`, both `(B,C,H,W)`. The module is responsible for valid amplitude and wrapped phase parameterization. |
| `amp_phase_logits` | One tensor `(B,H,W,C,2)` whose last axis is amplitude and phase logits; the shared adapter applies `sigmoid` and `pi*tanh`. |

Do not return `(B,2*C,H,W)` and rely on downstream guessing. Reshape and permute
inside the generator so the declared contract is exact.

Most non-CNN generators preserve the `N x N` spatial size. An architecture that
emits a larger support must prove that the selected training assembly and
forward extraction policy expect that exact support. `probe_big` is a historical
CNN support choice; it is not a general permission to change generator output
size.

## 4. Implement the Core Module and Thin Wrapper

The following example is deliberately small. Its architecture-specific settings
are named `tiny_residual_width` and `tiny_residual_blocks` so they do not borrow
the meaning of an unrelated FNO field.

```python
# ptycho_torch/generators/tiny_residual.py
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
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
        if width <= 0:
            raise ValueError("tiny_residual_width must be positive")
        if blocks <= 0:
            raise ValueError("tiny_residual_blocks must be positive")
        if output_mode != "real_imag":
            raise ValueError("tiny_residual supports only real_imag output")

        self.input_channels = int(input_channels)
        self.component_channels = int(component_channels)
        self.stem = nn.Conv2d(self.input_channels, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.head = nn.Conv2d(
            width,
            2 * self.component_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                "tiny_residual expected "
                f"(B,{self.input_channels},H,W), got {tuple(x.shape)}"
            )
        batch, _, height, width = x.shape
        x = self.head(self.blocks(self.stem(x)))

        # Match the established non-CNN real/imag adapter exactly:
        # (B, 2*C, H, W) -> (B, H, W, C, 2).
        return (
            x.view(batch, 2, self.component_channels, height, width)
            .permute(0, 3, 4, 2, 1)
            .contiguous()
        )


class TinyResidualGenerator:
    """Public registry wrapper; it does not construct a second application."""

    name = "tiny_residual"

    def __init__(self, config):
        self.config = config

    def build_model(self, pt_configs: Dict[str, Any]) -> nn.Module:
        from ptycho_torch.application_factory import build_ptychopinn_from_configs

        return build_ptychopinn_from_configs(pt_configs)
```

The wrapper must delegate to the shared application factory. Do not construct
`PtychoPINN_Lightning` independently in the wrapper; doing so creates a second
construction path that can diverge from checkpoint reload.

If the module needs a third-party package, declare and pin the supported
dependency in the project packaging, and raise an actionable import error.
Never fall back silently to another architecture.

## 5. Wire the Architecture as Rebuildable Structural State

This is the part that turns a local module into a supported experiment.

### 5.1 Add the public configuration

In `ptycho/config/config.py`:

1. add `tiny_residual` to `ModelConfig.architecture`;
2. add `tiny_residual_width` and `tiny_residual_blocks` to `ModelConfig`;
3. add the architecture to `validate_model_config()`;
4. validate the new fields and any architecture-specific restrictions.

These are user-facing topology choices, so they belong to `ModelConfig`, not
`TrainingConfig` or `PyTorchExecutionConfig`.

### 5.2 Add the resolved Torch configuration

Add the same architecture literal and structural fields to
`ptycho_torch/config_params.py::ModelConfig`. Then carry them through all
configuration boundaries:

- map shared public fields in `ptycho_torch/config_bridge.py`;
- include them in the public-to-factory forwarding in
  `ptycho_torch/workflows/components.py::_train_with_lightning`;
- include them in `ptycho_torch/model_spec.py::_CANONICAL_TO_TORCH` so the
  factory rejects public/Torch disagreement;
- verify that explicit factory overrides accept them. The accepted override
  set is derived from the Torch dataclasses, so declared Torch fields are
  normally picked up automatically.

Users still configure one model. The duplicated public and Torch dataclasses
are internal representations joined and checked by the factory; they are not
two independent user choices.

If a setting is genuinely Torch-only and should not be public, omit the public
field and canonical mapping deliberately. It then needs an explicit supported
override path and cannot be presented as an ordinary cross-backend config
field.

### 5.3 Register the public name

In `ptycho_torch/generators/registry.py`:

```python
from ptycho_torch.generators.tiny_residual import TinyResidualGenerator

_REGISTRY = {
    # existing entries...
    "tiny_residual": TinyResidualGenerator,
}
```

The registry key, wrapper `name`, and both configuration literals must match
exactly.

### 5.4 Add the core builder branch

In `ptycho_torch/model.py::_build_generator_module_from_config`:

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

This branch must use only persisted config and data join keys. Do not close over
a study variable, inspect a global default, or require the caller to inject a
module during reload.

### 5.5 Version `ModelSpec`

`ModelSpec` is the structural identity used to rebuild the graph. Every value
that changes module type, parameter count, tensor shape, or forward topology
must be sealed there.

The current `torch-model-spec-portable-v2` field set is frozen. When adding new
structural fields:

1. leave the portable-v1 and portable-v2 tuples unchanged;
2. introduce a portable-v3 tuple containing the new fields;
3. make portable-v3 the current schema;
4. accept older exact schemas and upgrade them with explicit, frozen values;
5. reject missing and unknown fields rather than filling them from mutable
   current defaults;
6. update `derive_model_spec()` and exact-field tests;
7. update `PtychoPINN_Lightning.__init__` so a checkpoint's dual-written old-v2
   `model_config` is recognized exactly and receives the same explicit v3
   migration values before comparison with the decoded spec;
8. update the hard-coded portable-v2 imports and field-shape classification in
   `ptycho_torch/artifact_schema.py`, including its unversioned compatibility
   path and artifact/checkpoint compatibility tests.

For example, the v2-to-v3 migration may insert
`tiny_residual_width=32` and `tiny_residual_blocks=4` as literal historical
values. Do not read those migration values from the current dataclass defaults.

An enclosing artifact schema needs a revision only if its own frozen envelope
or section semantics change. The nested `ModelSpec` version must always change
when its structural field set changes.

Adding only a new architecture value, with no new structural fields, does not
by itself change the `ModelSpec` field set. The existing `architecture` field
already seals that value. The example in this guide does require a schema bump
because it adds two new topology fields.

Once artifacts exist, treat the meaning of an architecture ID as stable. If a
code change makes its state-dict structure incompatible without a selectable
config value, use a new architecture ID or an explicit artifact migration;
do not weaken strict loading.

## 6. Test the Integration Before Training

Bring up the architecture in this order.

The selectors below are focused evidence recipes, not permanent completion
gates created by this guide. Use the smallest fresh checks that can falsify the
actual architecture, reload, and workflow claims required by the applicable
contract or experiment plan.

### 6.1 Module contract

Test the core module directly with `B=2`, `N=64`, `C=1`:

```python
x = torch.randn(2, 1, 64, 64)
y = module(x)
assert y.shape == (2, 64, 64, 1, 2)
assert y.dtype == x.dtype
assert torch.isfinite(y).all()
y.square().mean().backward()
assert any(p.grad is not None for p in module.parameters())
```

Also test a wrong input channel count, an unsupported output mode, and each
architecture-specific configuration boundary.

### 6.2 Shared complex adapter

Pass the result through `_predict_complex_patches()` and assert a finite complex
tensor `(B,C,N,N)`. This catches a common error where the last two axes or real
and imaginary channel ordering are swapped.

### 6.3 Registry and factory

Add the new architecture to the parameterized coverage in:

- `tests/torch/test_generator_registry.py`;
- `tests/torch/test_construction_consolidation.py`;
- `tests/torch/test_generator_adapter.py` when the output contract is new.

The construction-consolidation test should prove that registry construction
and sealed-`ModelSpec` construction produce the same state-dict name/shape
signature.

### 6.4 Structural identity and reload

Extend:

- `tests/torch/test_config_bridge.py` for public/Torch agreement;
- `tests/torch/test_model_spec.py` and the current schema-version test for exact
  field sets and deterministic old-schema upgrade;
- `tests/torch/test_lightning_checkpoint.py` so
  `PtychoPINN_Lightning.load_from_checkpoint(path)` rebuilds the new module with
  no manual kwargs;
- artifact tests if the enclosing artifact schema changed.

The checkpoint test is essential. A successful training forward does not prove
that tomorrow's inference process can reconstruct the graph.

Run the smallest affected selectors first, for example:

```bash
pytest \
  tests/torch/test_generator_registry.py \
  tests/torch/test_construction_consolidation.py \
  tests/torch/test_generator_adapter.py \
  tests/torch/test_config_bridge.py \
  tests/torch/test_model_spec.py \
  tests/torch/test_lightning_checkpoint.py -q
```

## 7. Prepare Data and Configure a Real Experiment

The architecture does not select a loader. The chosen entry point and input
type do that. The ordinary file-oriented route uses a standalone NPZ readable
by `RawData.from_file()`, including diffraction, scan coordinates, and a probe
with the shapes defined by the [core contract](../specs/spec-ptycho-core.md).
Train and test data must agree with the configured `N`, grouping, measurement
domain, and probe semantics.

For generated data, use `SimulationConfig` to construct the dataset first and
then train from its saved or in-memory acquisition record. Do not copy probe,
object, scan, or dose-generation fields into the custom architecture config.

For architecture development, begin with a small fixed train/test split. Keep
the sample identities unchanged while comparing the custom network with a
known registered baseline.

For architecture development, the public Python dataclasses are the clearest
interface. Define one public `ModelConfig`, one public `TrainingConfig`, and a
separate execution config:

```python
from pathlib import Path

from ptycho.config.config import (
    ModelConfig,
    PyTorchExecutionConfig,
    TrainingConfig,
)

model_config = ModelConfig(
    N=64,
    gridsize=1,
    model_type="pinn",
    architecture="tiny_residual",
    tiny_residual_width=32,
    tiny_residual_blocks=4,
    generator_output_mode="real_imag",
    object_layout="single_patch",
    training_canvas="independent",
    training_patch_weighting="central_mask",
    probe_mask=False,
)

training_config = TrainingConfig(
    model=model_config,
    train_data_file=Path("datasets/my_train.npz"),
    test_data_file=Path("datasets/my_test.npz"),
    output_dir=Path("outputs/tiny_residual_run_001"),
    backend="pytorch",
    n_groups=512,
    batch_size=16,
    nepochs=20,
    torch_loss_mode="poisson",
    optimizer="adam",
    scheduler="Default",
    subsample_seed=3,
)

execution_config = PyTorchExecutionConfig(
    accelerator="cuda",
    devices=1,
    strategy="auto",
    precision="32-true",
    learning_rate=1e-3,
    num_workers=0,
    logger_backend="csv",
    enable_checkpointing=True,
)
```

Use `TrainingConfig` for the scientific optimization recipe and
`PyTorchExecutionConfig` for Lightning/runtime mechanics. Do not put network
width or block count in the execution config.

### Scaling profile

Choose the data/scaling contract independently of architecture:

- For count-intensity data and the rectangular Poisson forward, use the named
  CI profile through the factory or supported CLI and keep
  `generator_output_mode="real_imag"`.
- For a historical normalized-amplitude dataset, select both
  `scale_contract_version="legacy_v1"` and
  `measurement_domain="normalized_amplitude"` explicitly at the supported
  entry point.

Do not change the loss or scaling profile merely to make a new architecture
train. First show that its output scale, shape, gradients, and forward physics
are correct under the intended experiment contract. See the
[Data Normalization Guide](../DATA_NORMALIZATION_GUIDE.md).

## 8. Train and Save the Bundle

### Programmatic development path

```python
from ptycho import params
from ptycho.config.config import update_legacy_dict
from ptycho.raw_data import RawData
from ptycho_torch.config_factory import resolve_ci_profile
from ptycho_torch.workflows.components import run_cdi_example_torch

# CONFIG-001: bridge the resolved public config before legacy-backed data I/O.
update_legacy_dict(params.cfg, training_config)
train_data = RawData.from_file(str(training_config.train_data_file))
test_data = RawData.from_file(str(training_config.test_data_file))
ci_overrides = resolve_ci_profile()

_, _, results = run_cdi_example_torch(
    train_data=train_data,
    test_data=test_data,
    config=training_config,
    do_stitching=False,
    execution_config=execution_config,
    overrides=ci_overrides,
)
```

This example selects the coherent CI count-intensity profile. For the explicit
legacy normalized-amplitude contract, replace `ci_overrides` with its paired
scale fields through the same supported override boundary. Never pass only one
half of a scale-contract pair.

The explicit call above performs the required legacy bridge before data access;
the scoped workflow enforces the bridge again at its own boundary. It then
derives and seals the Torch configs, trains with Lightning, writes checkpoints,
and persists `outputs/tiny_residual_run_001/wts.h5.zip`.

For the first run, use a tiny data subset and one epoch. Its purpose is to prove
the lifecycle, not reconstruction quality. Then inspect:

- finite train and validation losses;
- nonzero finite gradients in the custom module;
- `checkpoints/last.ckpt` or the configured best checkpoint;
- `wts.h5.zip` with the current Torch artifact identity;
- the saved effective architecture and custom structural values.

### Public workflow script after registration

The repository training script exposes public dataclass fields once the
architecture literal and its fields are integrated. Run it from the repository
root:

```bash
python scripts/training/train.py \
  --backend pytorch \
  --architecture tiny_residual \
  --tiny_residual_width 32 \
  --tiny_residual_blocks 4 \
  --generator_output_mode real_imag \
  --N 64 \
  --gridsize 1 \
  --train_data_file datasets/my_train.npz \
  --test_data_file datasets/my_test.npz \
  --output_dir outputs/tiny_residual_run_001 \
  --n_groups 512 \
  --batch_size 16 \
  --nepochs 20 \
  --torch-accelerator cuda \
  --torch-learning-rate 1e-3 \
  --torch-logger csv
```

An installed `ptycho_train` console command is intended to invoke the same
script, but use the repository form above while developing source extensions;
it does not depend on the packaging environment exposing `scripts` as an
importable top-level namespace.

`python -m ptycho_torch.train` is also supported for its documented flags, but
its native new-style CLI does not expose every public architecture field. Use
the public workflow script or the programmatic path for a custom topology.

The public workflow script also does not expose the named CI profile. For a
custom architecture on the CI count-intensity contract, use the programmatic
`resolve_ci_profile()` path shown above until that profile and the custom
architecture can be selected together by one CLI.

## 9. Prove Reload Before Evaluating Quality

Reload the saved bundle in a fresh process:

```python
from pathlib import Path

from ptycho_torch.workflows.components import load_inference_bundle_torch

models, saved_params = load_inference_bundle_torch(
    Path("outputs/tiny_residual_run_001")
)
model = models["diffraction_to_obj"]

assert model.model_config.architecture == "tiny_residual"
assert type(model.model.autoencoder).__name__ == "TinyResidualGeneratorModule"
assert model.model_config.tiny_residual_width == 32
assert model.model_config.tiny_residual_blocks == 4
```

This must succeed without constructing `TinyResidualGeneratorModule` in the
calling script and without passing config kwargs to the loader. If it does not,
the architecture is not integrated end to end.

The custom module and any third-party packages it imports must, of course, be
installed and importable in the inference environment.

## 10. Run Inference and Reassembly

The inference CLI loads the sealed architecture from the bundle; do not repeat
or override its architecture settings:

```bash
python -m ptycho_torch.inference \
  --model_path outputs/tiny_residual_run_001 \
  --test_data datasets/my_test.npz \
  --output_dir outputs/tiny_residual_run_001/inference \
  --n_images 64 \
  --patch-weighting uniform \
  --accelerator cuda \
  --quiet
```

The output directory contains the reconstructed amplitude and phase images.
For grouped patches, choose inference `patch_weighting` and optional VarPro
scaling as an inference policy; these do not retroactively change the
training-stage assembly or gradients. See the Configuration Guide's
"Training Assembly Versus Inference Reconstruction" section.

For CI artifacts, inference reuses the frozen training statistics and the
persisted probe/scaling identity. A custom architecture must not recompute or
replace them.

## 11. Add One End-to-End Regression

After the focused unit tests pass, add a small lifecycle test that performs:

```text
tiny dataset
  -> one short train
  -> checkpoint and bundle save
  -> fresh strict reload
  -> inference on a fixed batch
  -> finite complex patches and reconstructed amplitude/phase
```

Compare the fresh and reloaded model on the same fixed input. Their outputs
must agree to the tolerance appropriate for the selected device and precision.
This catches unsealed constructor state even when both models appear to train.

If the architecture is claimed to support DDP, add a separate two-process
smoke test through the established mmap/Lightning data rail. A single-device
run does not establish DDP compatibility.

If the applicable change gate calls for the repository-wide suite, run it only
after the focused selectors and lifecycle test pass, following the
[Testing Guide](../TESTING_GUIDE.md).

## 12. Evaluate the Architecture Fairly

For a useful architecture comparison, keep fixed:

- training and validation sample identities;
- `N`, `gridsize`, object layout, and training assembly;
- selected probe and probe mask semantics;
- scaling profile and measurement domain;
- loss, optimizer, learning-rate schedule, epochs, and seed;
- inference reassembly and any VarPro setting.

Vary only the architecture and its declared topology fields. Report parameter
count, compute or memory cost, loss curves, reconstruction metrics, and the
same amplitude/phase visual grid. A lower loss without a valid reload or a
physically coherent reconstruction is not a successful architecture result.

## 13. Common Failure Modes

| Symptom | Likely cause |
|---|---|
| `Unknown architecture` | The public literal, validator, Torch literal, or registry entry is missing. |
| Fresh construction works; checkpoint reload says unsupported architecture | `_build_generator_module_from_config()` lacks the branch or needs non-persisted state. |
| Reload has state-dict size mismatches | A topology field was omitted from `ModelSpec`, or the builder ignored the saved value. |
| Public and Torch configs disagree | The bridge or `_CANONICAL_TO_TORCH` mapping is incomplete. |
| Complex patches have the wrong channel order | `(B,2*C,H,W)` was reshaped with `(C,2)` instead of the established `(2,C)` ordering. |
| CI construction rejects the model | CI requires the coherent count-intensity, Poisson, rectangular-scaled, real/imag contract. |
| Training runs but inference quality changes after reload | The module read a mutable global/default, random constructor choice, or caller injection not represented in the sealed spec. |
| `gridsize=1` works but grouped patches fail | The module collapsed the semantic `C` axis or the study runner does not support that architecture for grouped data. |
| Architecture comparison changes unexpectedly | A loss, probe, scaling, assembly, or inference policy changed with the model. |

## 14. Completion Checklist

- [ ] The core module accepts the exact Torch input layout.
- [ ] The core module returns one declared output contract with no broadcasting.
- [ ] The public and Torch model configs contain the architecture and its user-visible topology fields.
- [ ] Configuration validation rejects unsupported combinations.
- [ ] The workflow carries the fields into the closed factory.
- [ ] The registry wrapper delegates to the one application factory.
- [ ] The core builder reconstructs the module entirely from saved state.
- [ ] `ModelSpec` seals every topology field, with a schema bump when the field set changes.
- [ ] Old schemas upgrade with explicit frozen values and unknown fields fail.
- [ ] Dual-written Lightning checkpoint configs and artifact classifiers recognize the same old/current schemas.
- [ ] Registry and sealed construction have the same state signature.
- [ ] Checkpoint-only and bundle-only reload require no manual injection.
- [ ] A short train-save-load-infer test passes.
- [ ] The intended scaling, physics, and reconstruction policies remain fixed.
- [ ] Claimed multi-device support has a DDP-specific smoke test.
- [ ] Focused tests pass before the repository-wide regression suite.

## Related Contracts and Guides

- [Configuration Guide](../CONFIGURATION.md) — ownership, config factory,
  `ModelSpec`, data routing, and probe lifecycle.
- [PyTorch Workflow](pytorch.md) — supported training and inference entry
  points, execution settings, and persistence.
- [Torch Loader and Batch Contract](../specs/spec-ptycho-interfaces.md) —
  normative batch fields and shapes.
- [PtychoPINN Core Contract](../specs/spec-ptycho-core.md) — normative physics,
  scaling, loss, and output contracts.
- [Ptychodus API Specification](../../specs/ptychodus_api_spec.md) — backend and
  portable bundle behavior.
- [Testing Guide](../TESTING_GUIDE.md) — selector and suite guidance.
