# PyTorch Workflow Guide

This guide is the authority for configuring and running the PyTorch backend of
PtychoPINN: the Lightning-based training stack under `ptycho_torch/` with a generator
registry for architecture selection. PyTorch (torch ≥ 2.2) is a mandatory dependency.

## 1. Overview

There are three ways to run the backend, from highest-level to lowest:

| Entry point | Use when |
| --- | --- |
| Unified CLIs: `ptycho_train` / `ptycho_inference` with `--backend pytorch` | You want the backend-agnostic workflow (same flags as TensorFlow, plus `--torch-*` execution flags) |
| Native CLIs: `python -m ptycho_torch.train` / `python -m ptycho_torch.inference` | You want direct control of torch execution flags |
| Programmatic: `ptycho_torch.workflows.components.run_cdi_example_torch` | You are composing a custom workflow or study runner |

Key properties:

- **Configuration** uses the same canonical public records as TensorFlow:
  nested Pydantic `TrainingConfig`, plus dataclass `ModelConfig` and
  `InferenceConfig`. They are bridged to the Torch-side dataclasses in
  `ptycho_torch/config_params.py` through `ptycho_torch/config_bridge.py`.
- **Training** runs through `PtychoPINN_Lightning` (`ptycho_torch/model.py`) with
  deterministic settings, Lightning checkpointing, and the full physics loss for every
  architecture.
- **Data contract** uses the shared standalone NPZ keys and shapes while
  supporting both legacy normalized-amplitude measurements and CI
  count-intensity measurements (see §2).

### Training entry-point convergence

Entry points own source-specific validation and translate their inputs into a
neutral, resolved training request. They do not invoke one another. Training
behavior converges on `ptycho.workflows.training.run_training_workflow`:

```text
ptycho_synthetic ── simulate + verify manifest ─┐
                                                │
ptycho_train ───── validate supplied NPZs ─────┼─> Shared training service
                                                │      ├─ group data once
study adapter ──── validate cached dataset ─────┘      ├─ select memory/mmap rail
                                                       ├─ resolve model/runtime
                                                       └─ train + save bundle
```

At this boundary, `ptycho_synthetic` owns simulation, manifest identity, and
stage lifecycle; `ptycho_train` owns caller-supplied standalone NPZ inputs; and
study adapters own any specialized cached-data translation. The shared service
owns grouping, data-residency selection, model/runtime resolution, training,
and bundle persistence. A compatibility entry point that still performs any of
those shared steps locally is transitional and should converge by delegating to
the service, not by calling another CLI or passing CLI-specific arguments into
the shared core.

## 2. Prerequisites

- `pip install .` installs the declared `torch`, `lightning`, and `tensordict`
  dependencies. Supported workflows require torch ≥ 2.2.
  For a specific CUDA build, install PyTorch manually first
  ([instructions](https://pytorch.org/get-started/locally/)), then `pip install .`
- Public training configuration and CLI generation use the declared
  `pydantic-settings` dependency.
- Input NPZ files with `diff3d`, `xcoords`/`ycoords` scan positions, a complex
  `probeGuess`, and `objectGuess`. Under the legacy contract, `diff3d` is
  normalized amplitude (square root of intensity). Under `ci_intensity_v2`, it
  is Poisson-realized detector counts and `probeGuess` is the CI-scaled physical
  forward probe. The Torch mmap loader also accepts `diffraction` as a
  compatibility alias; the ordinary `RawData.from_file()` route requires
  `diff3d`. The resolved configuration must match the NPZ measurement domain.

## 3. Configuration

### 3.1. Configuration Owners and Runtime Resolution

1. **Canonical configs** (`TrainingConfig`, `InferenceConfig`, `ModelConfig`) describe
   the model and data. Public `TrainingConfig` is a nested Pydantic settings
   model; the other two are validated dataclasses. They bridge to `params.cfg`
   and to the Torch config records.
   `update_legacy_dict(params.cfg, config)` MUST run before any data
   loading or legacy-module import. The CLIs do this automatically via
   `ptycho_torch/config_factory.py`; programmatic callers must do it themselves.
2. **`ExecutionRequest`** (`ptycho_torch.execution_request`) carries unresolved
   runtime values such as accelerator, workers, checkpointing, and logging,
   together with the set of fields the caller explicitly supplied. It does not
   own optimization or model topology and must NEVER populate `params.cfg`.
3. The factory validates canonical owners, observes capabilities only when
   required, and returns **`PyTorchExecutionConfig`** as a runtime-only output
   for Lightning and DataLoader construction. A bare resolved carrier is not a
   supported factory input. Full field catalog and validation rules:
   `specs/ptychodus_api_spec.md` §4.9.

Learning rate, optimizer, scheduler, gradient clipping, and accumulation enter
through Torch `TrainingConfig` or an explicit canonical factory training patch.
Topology enters through Torch `ModelConfig`.

```python
from pathlib import Path
from ptycho.config.config import (
    DataConfig,
    ModelConfig,
    SamplingConfig,
    TrainingConfig,
    update_legacy_dict,
)
from ptycho import params

config = TrainingConfig(
    model=ModelConfig(N=64, gridsize=2, model_type='pinn', architecture='cnn'),
    data=DataConfig(
        train_data_file=Path('datasets/my_train.npz'),
        test_data_file=Path('datasets/my_test.npz'),   # optional
    ),
    sampling=SamplingConfig(n_groups=512),
    batch_size=4,
    nepochs=10,
    output_dir=Path('outputs/my_experiment'),
)
update_legacy_dict(params.cfg, config)   # MANDATORY before data loading
```

#### Training-only CI profile

The native Torch CLI accepts `--profile ci`; the training factories accept
`profile="ci"`. Both select the same named starting bundle for existing
count-intensity NPZs. It is expanded before normal configuration construction
and validation. It locks exactly five fields:
`scale_contract_version=ci_intensity_v2`,
`measurement_domain=count_intensity`,
`physics_forward_mode=rectangular_scaled`, `torch_loss_mode=poisson`, and
`loss_function=Poisson`. Contradictions fail; non-contract defaults remain
overrideable. With `profile=None`, ordinary resolution applies without a named
bundle.

The profile's overrideable `rect_s1s2_init` default is `dose_closure`. A bare
`ModelConfig` remains `ones`, and an explicit `ones` override wins over the
profile. On the native CLI, omitting `--rect-s1s2-init` authors no override, so
`--profile ci` retains `dose_closure`.

Fresh dose-closure runs write `rect-s1s2-initialization-v2`; strict historical
v1 records remain readable without being rewritten. See the
[configuration guide](../CONFIGURATION.md#dose-closure-initialization) for the
selection identity, record method, and failure semantics.

This training-only profile is distinct from the synthetic runner's
`--profile cnn-lines-ci`. The persisted resolved model, data, training, and
bundle identity controls inference, so a profile name is not reselected when a
bundle is loaded. See the [configuration guide](../CONFIGURATION.md#torch-training-only-ci-profile)
for its overrideable defaults.

### 3.2. Architecture Selection (Generator Registry)

`config.model.architecture` routes through the generator registry
(`ptycho_torch/generators/registry.py`, `resolve_generator`). Every architecture
trains through `PtychoPINN_Lightning` with the same physics pipeline. Registered
architectures:

- `cnn` (default) — U-Net-style CNN encoder/decoder pair
- `fno`, `fno_vanilla`, `ffno` — Fourier-operator stacks (see `fno_modes`,
  `fno_width`, `fno_blocks`, `fno_cnn_blocks`, `fno_input_transform`)
- `neuralop_uno` — wraps external `neuraloperator==2.0.0` U-NO (locked to the
  Lines128 CDI path: `N=128`, `gridsize=1`, `C=1`, `real_imag`)

To implement, configure, train, save, and reload a new architecture, follow the
[Custom PyTorch CDI Architecture Guide](custom_torch_architecture.md). The
generator-package README is a lower-level reference for the existing modules.

**Reliability caveat:** the stock `cnn` under the count-Poisson recipe at
`N=128` is collapse-prone and can produce a flat-amplitude output unless the
complete TF-parity preset below is applied (§3.6).

### 3.3. Loss, Scheduler, and Sampling

- Public `TrainingConfig.loss.torch_loss_mode` (resolved Torch
  `TrainingConfig.torch_loss_mode`): `'poisson'` (physics-weighted Poisson NLL,
  default) or `'mae'` (amplitude-only MAE, `physics_weight=0`). Native-CLI flag:
  `--torch-loss-mode`.
- Public `TrainingConfig.scheduler.kind` (resolved Torch
  `TrainingConfig.scheduler`): `'Default'` (constant LR), `'Exponential'`,
  `'WarmupCosine'` (with `lr_warmup_epochs`, `lr_min_ratio`), or
  `'ReduceLROnPlateau'`. The native `ptycho_torch.train` CLI also retains the
  legacy `MultiStage` and `Adaptive` spellings; its `--help` output is the
  authority for that interface.
- Public `TrainingConfig.sampling.subsample_seed` seeds
  `lightning.pytorch.seed_everything`; `sampling.sequential_sampling=True`
  uses deterministic first-N grouping anchors after raw-row subsampling.
  Subsampled indices
  are persisted (`raw.sample_indices`, `tmp/subsample_seed{X}_indices.txt`) and
  asserted equal across backends.

### 3.4. Probe Masking

`config.model.probe_mask` (default `False`) enables a centered soft disk mask
(diameter `N/2`, Gaussian edge `sigma=1 px`) on the probe. Overrides:
`probe_mask_tensor` (explicit `(N, N)` mask; enables masking even when
`probe_mask=False`), `probe_mask_sigma`, `probe_mask_diameter`. CLI:
`--probe-mask/--no-probe-mask`, `--probe-mask-sigma`, `--probe-mask-diameter` on both
native CLIs.

### 3.5. CNN Output and Physics-Forward Knobs

Four torch-`ModelConfig` knobs port the legacy-main CNN representation and physics as
opt-in modes. All default to the values that keep existing CNN and FNO behavior
unchanged:

| Knob | Default | Opt-in value | Effect |
|---|---|---|---|
| `cnn_output_mode` | `'amp_phase'` | `'real_imag'` (Unsupervised-only) | CNN emits `(real, imag)` via `ScaledTanh` boxes (real ∈ (−0.8, 1.2), imag ∈ (−1.2, 1.2)); prerequisite for `rectangular_scaled`. Representability limit: unit-amplitude objects near `|phase| → π` are unreconstructable in this mode. |
| `use_shared_decoder` | `False` | `True` | Single shared decoder emitting `2*C_out` channels, split per branch; architecture-only knob. |
| `training_patch_weighting` | `'central_mask'` | `'probe'` (or `'uniform'`) | Training-forward reassembly weighting: binary center mask vs `Σ|probe|²`-weighted (`'uniform'` isolates the code-path change without probe weighting). Distinct from the inference-only `InferenceConfig.patch_weighting`. |
| `physics_forward_mode` | `'amplitude'` | `'rectangular_scaled'` | Routes patches through `RectangularScaledDiffraction` (analytic real/imag intensity model with per-dataset trainable `s1`/`s2` unless `rect_s1s2_trainable=False`). Requires an effective `real_imag` generator output; for the CNN, select `cnn_output_mode='real_imag'`. Matching intensity-domain losses are selected automatically. |

`cnn_output_mode` and `physics_forward_mode` are coupled but are not aliases.
The first controls how the CNN decoder parameterizes the object. Other
architectures use `generator_output_mode` for the equivalent generator-output
contract. Either representation is normalized to one complex object before the
second control selects the differentiable diffraction calculation and detector
prediction domain:

```text
generator output representation       complex object       physics forward
amp/phase or real/imaginary  ────────► x              ────► amplitude or intensity
```

The supported combinations are:

| Effective generator output | `physics_forward_mode='amplitude'` | `physics_forward_mode='rectangular_scaled'` |
|---|---|---|
| `amp_phase` | Supported legacy amplitude-domain path | Rejected: independent real/imaginary scaling would not match the generator heads |
| `real_imag` | Supported representation ablation using the amplitude-domain forward | Supported rectangular intensity path used by CI |

The CI profiles select `real_imag` and `rectangular_scaled` together because
that is their coherent scientific contract, not because the two fields have
the same meaning.

Two further knobs are **inference-only** (`InferenceConfig.patch_weighting`,
`InferenceConfig.varpro_scaling`): they affect only
`ptycho_torch.reassembly.reconstruct_image_barycentric` (the in-process reconstruction
path) and never touch training numerics. The `python -m ptycho_torch.inference` CLI
does not consume them — it always uses uniform
`helper.reassemble_patches_position_real`. Call `reconstruct_image_barycentric`
directly when these knobs must take effect.

### 3.6. TF-Parity Preset for the Torch CNN (N=128 reliability)

Three knobs on the standard `cnn` close the collapse gap against the TensorFlow
reference. Here "preset" describes a bundle of controls, not a named
configuration profile or registry entry:

| Knob | Where | Parity value | Default |
|---|---|---|---|
| `cbam_encoder` | torch `ModelConfig` | `False` | `True` (stock `cnn` remains collapse-prone) |
| `parity_init_scheme` | `PtychoPINN_Lightning` kwarg (`"default"` \| `"tf_glorot"`) | `"tf_glorot"` | `"default"` (kaiming) |
| `scheduler` | `TrainingConfig` | `"ReduceLROnPlateau"` | `"Default"` |

An additional default-off mechanism, `parity_scale_mode`
(`PtychoPINN_Lightning` kwarg; `"off"` \| `"tied"` \| `"input"` \| `"output"` \|
`"fixed"`), controls the TF-parity global intensity scale; it is forwarded by
`ptycho_torch/train_lightning_only.py` and driven from
`scripts/studies/varpro_probe_ablation_runner.py`
(`--cbam-encoder`, `--parity-init-scheme`, `--parity-scale-mode`, `--scheduler`).

These controls, and evidence produced by a hybrid architecture, do not establish
a quality threshold or baseline for `cnn-lines-ci`; architecture-specific
claims require evidence from a run under that exact contract.

Cautions:

- Do NOT set `intensity_scale_trainable=True` alongside the parity kwargs — the dead
  `IntensityScalerModule` machinery silently overwrites the input-side parity scale.

## 4. Training

### 4.1. Unified CLI (backend selection)

```bash
ptycho_train --data.train_data_file datasets/my_train.npz \
  --output_dir outputs/my_run \
  --backend pytorch \
  --torch-accelerator auto --torch-logger csv
```

Torch runtime flags on the unified scripts include `--torch-accelerator`,
`--torch-logger`, `--torch-num-workers`, `--torch-deterministic`,
`--torch-enable-checkpointing`, and `--torch-checkpoint-save-top-k`.
Optimization flags such as `--torch-learning-rate`, `--torch-scheduler`, and
`--torch-accumulate-grad-batches` form an explicit Torch `TrainingConfig`
patch. Dispatch happens in `ptycho/workflows/backend_selector.py` (see §7).

### 4.2. Native CLI

```bash
CUDA_VISIBLE_DEVICES="0" python -m ptycho_torch.train \
  --train_data_file datasets/my_train.npz \
  --test_data_file datasets/my_test.npz \
  --output_dir outputs/my_run \
  --n_images 512 --gridsize 2 --batch_size 16 --max_epochs 50 \
  --accelerator cuda --logger csv --quiet
```

For a count-intensity run, omission keeps the `ci` profile's dose-closure
default; pass `ones` only when unit initialization is intentional:

```bash
python -m ptycho_torch.train \
  --train_data_file datasets/counts_train.npz \
  --output_dir outputs/ci_run \
  --profile ci

python -m ptycho_torch.train \
  --train_data_file datasets/counts_train.npz \
  --output_dir outputs/ci_unit_init \
  --profile ci --rect-s1s2-init ones
```

These two flags belong to the native Torch CLI; the unified `ptycho_train`
command does not expose `--profile ci` or `--rect-s1s2-init`.

Flags (`python -m ptycho_torch.train --help` is authoritative):

| Group | Flags |
|---|---|
| Data/model | `--train_data_file`, `--test_data_file`, `--output_dir`, `--n_images` (number of groups), `--gridsize`, `--batch_size`, `--max_epochs`, `--profile {ci}`, `--rect-s1s2-init {ones,dose_closure}` |
| Runtime | `--accelerator {auto,cuda,cpu,tpu,mps}`, `--deterministic/--no-deterministic`, `--num-workers`, `--quiet` |
| Optimization | `--learning-rate`, `--scheduler {Default,Exponential,MultiStage,Adaptive,WarmupCosine,ReduceLROnPlateau}`, `--accumulate-grad-batches` |
| Checkpointing | `--enable-checkpointing/--disable-checkpointing`, `--checkpoint-save-top-k`, `--checkpoint-monitor` (default `val_loss`, auto-aliased to the model's actual metric, e.g. `poisson_val_loss`), `--checkpoint-mode`, `--early-stop-patience` |
| Loss/probe | `--torch-loss-mode {poisson,mae}`, `--probe-mask/--no-probe-mask`, `--probe-mask-sigma`, `--probe-mask-diameter` |
| Logging | `--logger {csv,tensorboard,mlflow,none}`, `--log-patch-stats`, `--patch-stats-limit` |
| Deprecated | `--device` (→ `--accelerator`), `--disable_mlflow` (→ `--logger none` + `--quiet`) |

The native parser retains `tpu` in its accepted `--accelerator` spellings for
CLI compatibility, but execution-request resolution rejects it explicitly:
Torch-XLA TPU execution is unsupported.

The CLI builds `ExecutionRequest` with
`build_execution_request_from_args()` and a canonical optimization patch with
`build_training_config_patch_from_args()`. The config factory validates owner
configuration, performs the mandatory `params.cfg` bridge, resolves runtime
capabilities, and returns `PyTorchExecutionConfig` in the payload.

### 4.3. Programmatic

```python
from ptycho.raw_data import RawData
from ptycho_torch.execution_request import ExecutionRequest
from ptycho_torch.workflows.components import run_cdi_example_torch

train_data = RawData.from_file(str(config.data.train_data_file))
test_data = (
    RawData.from_file(str(config.data.test_data_file))
    if config.data.test_data_file
    else None
)
execution_request = ExecutionRequest(
    values={
        "accelerator": "auto",
        "num_workers": 0,
        "logger_backend": "csv",
    },
    explicit_fields=frozenset(
        {"accelerator", "num_workers", "logger_backend"}
    ),
)

amplitude, phase, results = run_cdi_example_torch(
    train_data, test_data, config,
    do_stitching=True,            # False → (None, None, results) training-only
    execution_config=execution_request,  # optional unresolved request
)
```

`run_cdi_example_torch` normalizes the data into `PtychoDataContainerTorch`, seeds and
passes the request through the factory exactly once, instantiates
`PtychoPINN_Lightning`, runs `Trainer.fit()` (deterministic, checkpoints under
`{output_dir}/checkpoints/`), persists the bundle, and — when
`do_stitching=True` — runs Lightning prediction and reassembles the image
(`flip_x`/`flip_y`/`transpose` args control coordinate transforms, `M` the stitch
window).

## 5. Checkpoints, Persistence, Reproducibility

- **Determinism:** `deterministic=True` plus the resolved sampling seed
  (`config.sampling.subsample_seed` on the public record).
- **Checkpoints:** `{output_dir}/checkpoints/last.ckpt` (Lightning), with
  hyperparameters embedded via `save_hyperparameters()` — checkpoints reload without
  manual config kwargs.
- **Bundle:** the final model persists as `{output_dir}/wts.h5.zip`. The
  `intensity_scale` is captured (learned value if trainable, else the spec fallback
  `sqrt(nphotons)/(N/2)`) and stored in the bundle's `params.dill`, so inference uses
  the same normalization as training.
- **Resolved identity:** the bundle's persisted model, data, training, and
  enclosing artifact identity—including `measurement_domain` and
  `scale_contract_version`—is authoritative at inference and prevents domain
  drift. Loading does not rerun or require a named training profile.
- **Loading:**

```python
from ptycho_torch.workflows.components import load_inference_bundle_torch
models_dict, loaded_config = load_inference_bundle_torch(Path('outputs/my_run'))
lightning_module = models_dict['lightning_module']
```

## 6. Inference

CLI (loads the bundle, runs Lightning prediction, saves
`reconstructed_amplitude.png` / `reconstructed_phase.png`):

```bash
CUDA_VISIBLE_DEVICES="0" python -m ptycho_torch.inference \
  --model_path outputs/my_run \
  --test_data datasets/my_test.npz \
  --output_dir outputs/inference_results \
  --n_images 64 --accelerator cuda --quiet
```

Additional flags: `--num-workers`, `--inference-batch-size` (default: reuse training
batch size), probe-mask flags, `--log-patch-stats`. A legacy MLflow-run mode
(`--run_id`, `--infer_dir`, `--file_index`) still exists but is not the default path.
Reminder: this CLI stitches with uniform weighting only — use
`ptycho_torch.reassembly.reconstruct_image_barycentric` in-process for
`patch_weighting`/`varpro_scaling` (§3.5).

**Device handoff:** when chaining training and custom inference
in one process, do not assume the post-`fit()` module is still on the training
accelerator — resolve the target device explicitly and call `model.to(device)` before
the forward loop.

## 7. Backend Selection (Unified Workflows / Ptychodus)

`TrainingConfig.backend` / `InferenceConfig.backend` (`'tensorflow'` default,
`'pytorch'`) select the implementation. The dispatcher
(`ptycho/workflows/backend_selector.py`; contract in `specs/ptychodus_api_spec.md`
§4.8) guarantees:

- `'tensorflow'` routes to `ptycho.workflows.components` without importing torch;
  `'pytorch'` routes to `ptycho_torch.workflows.components` with the same
  `(amplitude, phase, results)` return shape (plus `results['backend']`).
- The legacy `params.cfg` bridge (`update_legacy_dict`) runs before backend
  inspection.
- Fail-fast: missing torch raises an actionable `RuntimeError` (no silent TensorFlow
  fallback — PyTorch is a hard dependency); invalid backend values raise `ValueError`; loading a
  checkpoint with the wrong backend raises a descriptive error (TF bundles are Keras
  `.h5.zip`; torch bundles are Lightning `.ckpt` + `.h5.zip`).

Validated by `pytest tests/torch/test_backend_selection.py -vv`.

## 8. Experiment Tracking and Logging

- `--logger csv` (default): metrics from `self.log()` land in
  `{output_dir}/lightning_logs/version_N/metrics.csv`; no extra dependencies.
- `--logger tensorboard`: view with `tensorboard --logdir {output_dir}/lightning_logs/`.
- `--logger mlflow`: requires an MLflow server/URI. With MLflow, intermediate
  reconstruction logging is available through the execution-config fields
  `recon_log_every_n_epochs`, `recon_log_num_patches`, `recon_log_fixed_indices`,
  `recon_log_stitch` (opt-in, expensive), `recon_log_max_stitch_samples`
  (`ptycho_torch/workflows/recon_logging.py`; artifacts under
  `epoch_NNNN/patch_NN/*.png`, DDP-safe via `trainer.is_global_zero`).
- `--logger none` + `--quiet`: fully silent smoke runs.
- Loss/metric parity: training logs `amp_inv_mae_epoch` (measurement domain) and
  `amp_mae_tf_scale_epoch` (TF-normalized domain) so Poisson-vs-MAE curves compare
  directly against TensorFlow amplitude MAE.

## 9. Study Runners

Deep experiment knobs (grid-lines dataset generation, position-reassembly backend
selection, count-scale modes, structural-search sweeps, parity controls) are owned by
the study CLIs, not this guide:

- `scripts/studies/grid_lines_torch_runner.py` — grid-lines training/eval for the
  registry architectures (`--architecture`, plateau/warmup schedulers, probe-source
  validation, reassembly strategy knobs).
- `scripts/studies/grid_lines_compare_wrapper.py` — TF-vs-torch multi-model
  comparisons (`--architectures`, `--dataset-source`, probe scaling/masking).
- `scripts/studies/varpro_probe_ablation_runner.py` — parity-control ablations
  (§3.6 flags).

Consult each runner's `--help` and `scripts/studies/README.md`.

## 10. Constraints and Known Pitfalls

- **Gradient accumulation:** `PtychoPINN_Lightning` uses manual optimization, which is
  incompatible with gradient accumulation — `--accumulate-grad-batches > 1` raises a
  `RuntimeError` before training. Keep the default (`1`).
- **Supervised mode** (`model_type='supervised'`) requires `label_amp` /
  `label_phase` keys in the NPZ; experimental datasets lack them and fail dataloader
  validation. Use PINN mode or generate labeled synthetic data.
- **Gridsize > 1 support** is architecture-gated in the study runners (only `cnn`
  is ported); other architectures currently reject `gridsize > 1` there.
- **N=128 count-Poisson CNN is collapse-prone** without the complete TF-parity
  preset (§3.6).
- **`intensity_scale_trainable=True`** conflicts with the parity scale path (§3.6).
- Shape mismatches at load time usually mean the `update_legacy_dict(params.cfg,
  config)` bridge was skipped (§3.1).

## 11. Testing

- Fast suite: `pytest tests/torch -m "not slow"`; the CI gate runs exactly this via
  `bash ci/run_ci_tests.sh`.
- End-to-end regression (train→save→load→infer, GPU-pinned):

```bash
CUDA_VISIBLE_DEVICES="0" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

- Visual parity evidence for pipeline changes:
  `python scripts/tools/patch_parity_helper.py --tf-npz ... --torch-npz ...`
  (aligns shared `sample_indices`, writes comparison grids under `tmp/patch_parity/`).

---

**Related Documentation:**
- <doc-ref type="guide">docs/CONFIGURATION.md</doc-ref> — configuration ownership and precedence
- <doc-ref type="guide">docs/DATA_NORMALIZATION_GUIDE.md</doc-ref> — scaling and measurement conventions
- <doc-ref type="spec">specs/ptychodus_api_spec.md</doc-ref> — backend dispatch and execution-config contracts (§4.8–4.9)
