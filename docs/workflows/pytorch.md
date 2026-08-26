# PyTorch Workflow Guide

This guide is the authority for configuring and running the PyTorch backend of
PtychoPINN: the Lightning-based training stack under `ptycho_torch/` with a generator
registry for architecture selection. PyTorch (torch ≥ 2.2) is a mandatory dependency.

## 1. Overview

There are four ways to run the backend, from highest-level to lowest:

| Entry point | Use when |
| --- | --- |
| `ptycho_synthetic` | You want the supported synthetic simulate → train → strict-reload → mmap barycentric reconstruct → evaluate workflow |
| Unified CLIs: `ptycho_train` / `ptycho_inference` with `--backend pytorch` | You want the backend-agnostic workflow (same flags as TensorFlow, plus `--torch-*` execution flags) |
| Native CLIs: `python -m ptycho_torch.train` / `python -m ptycho_torch.inference` | You want direct control of torch execution flags |
| Programmatic: `ptycho_torch.workflows.components.run_cdi_example_torch` | You are composing a custom workflow or study runner |

Key properties:

- **Configuration** uses the same canonical dataclasses as TensorFlow (`ptycho.config.config.TrainingConfig` / `InferenceConfig`). The config factory resolves the torch-side config singletons (`ptycho_torch/config_params.py`), and `ptycho_torch/config_bridge.py` translates those resolved Torch singletons into the TensorFlow dataclasses, which are then projected to the legacy `params.cfg` via `update_legacy_dict`. Normative field mapping: <doc-ref type="spec">docs/specs/spec-ptycho-config-bridge.md</doc-ref>.
- **Training** runs through `PtychoPINN_Lightning` (`ptycho_torch/model.py`) with
  deterministic settings, Lightning checkpointing, and the full physics loss for every
  architecture.
- **Data contract** uses the shared standalone NPZ keys and shapes while
  supporting both legacy normalized-amplitude measurements and
  `ci_intensity_v2` count-intensity measurements. The resolved configuration
  must match the stored measurement domain; normative schema:
  <doc-ref type="contract">docs/specs/spec-ptycho-core.md</doc-ref>.
- **Recommended parameter baselines** live in
  [docs/model_baselines.md](../model_baselines.md); this guide documents the available
  knobs, not the current best-practice combination.

### Configuration and identity lifecycle

Every Torch run passes through the same four stages:

```text
authored config (TrainingConfig / InferenceConfig)
  -> resolved payload (TrainingPayload / InferencePayload)   [create_training_payload / create_inference_payload]
  -> sealed identity (ModelSpec)                             [checkpoint / bundle write]
  -> restored identity (strict bundle/checkpoint decode)     [decode_checkpoint_hparams + load_inference_bundle_torch]
```

The resolved payload is the single configuration currency at the four
consumption points: `_train_with_lightning` (training service), loader
construction, `PtychoPINN_Lightning.__init__` (module construction), and the
inference kernel (decoded bundle identity + explicit runtime argument).

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

- `pip install .` installs torch ≥ 2.2, `lightning`, and `tensordict` automatically.
  For a specific CUDA build, install PyTorch manually first
  ([instructions](https://pytorch.org/get-started/locally/)), then `pip install .`
- Input NPZ files conforming to `docs/specs/spec-ptycho-core.md`. Legacy files
  store normalized amplitude; CI files store count-intensity measurements.
  In both cases the resolved measurement contract must agree with the NPZ.

## 3. Configuration

### 3.1. Scientific Configuration and Runtime Resolution

1. **Canonical and Torch scientific configs** describe data, model topology,
   optimization, and inference. Learning rate, scheduler, gradient clipping,
   and accumulation resolve into Torch `TrainingConfig`; topology resolves into
   Torch `ModelConfig`. Compatibility entry points bridge the canonical
   projection to `params.cfg` before a legacy consumer, while the modern Torch
   payload resolver leaves global configuration and filesystem state untouched
   and scopes any surviving legacy leaf separately.
2. **`ExecutionRequest`** carries unresolved runtime-only intent and explicit
   presence for accelerator, devices, workers, precision, logging,
   checkpointing, and Lightning mechanics. Capability resolution returns
   **`PyTorchExecutionConfig`** as an effective output carrier. Callers do not
   construct that resolved carrier as a request, and neither form owns topology
   or optimization.

Full execution field catalog and validation rules:
`specs/ptychodus_api_spec.md` §4.9.

```python
from pathlib import Path
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params

config = TrainingConfig(
    model=ModelConfig(N=64, gridsize=2, model_type='pinn', architecture='cnn'),
    train_data_file=Path('datasets/my_train.npz'),
    test_data_file=Path('datasets/my_test.npz'),   # optional
    training_groups=512,
    batch_size=4,
    nepochs=10,
    output_dir=Path('outputs/my_experiment'),
)
update_legacy_dict(params.cfg, config)   # MANDATORY before data loading
```

#### Training-only CI profile

The native Torch CLI accepts `--profile ci`; the training factories accept
`profile="ci"`. Both select the same named starting bundle for existing
count-intensity NPZs. It locks `scale_contract_version=ci_intensity_v2`,
`measurement_domain=count_intensity`,
`physics_forward_mode=rectangular_scaled`, `torch_loss_mode=poisson`, and
`loss_function=Poisson`. Contradictions fail closed; the overrideable
`rect_s1s2_init` profile default is `dose_closure`, and an explicit `ones`
selects unit initialization. With `profile=None`, ordinary resolution applies
without a named bundle and the bare model default remains `ones`.

This training-only profile is distinct from the synthetic runner's
`--profile hybrid-resnet-lines-ci`, which also chooses the simulation recipe
and defaults to `dose_closure`. Persisted model, data, training, inference,
and artifact identity controls inference, so loading does not require selecting either
profile name again. See the
[configuration guide](../CONFIGURATION.md#torch-training-only-ci-profile) for
the complete field tables.

### 3.2. Architecture Selection (Generator Registry)

`config.model.architecture` routes through the generator registry
(`ptycho_torch/generators/registry.py`, `resolve_generator`). Every architecture
trains through `PtychoPINN_Lightning` with the same physics pipeline. Registered
architectures:

- `cnn` (default) — U-Net-style CNN encoder/decoder pair
- `fno`, `fno_vanilla`, `ffno` — Fourier-operator stacks (see `fno_modes`,
  `fno_width`, `fno_blocks`, `fno_cnn_blocks`, `fno_input_transform`)
- `hybrid`, `stable_hybrid` — FNO/CNN hybrids (`stable_hybrid` adds
  InstanceNorm-stabilized Norm-Last residual blocks)
- `neuralop_uno` — wraps external `neuraloperator==2.0.0` U-NO (locked to the
  Lines128 CDI path: `N=128`, `gridsize=1`, `C=1`, `real_imag`)
- `hybrid_resnet` and variants (`hybrid_resnet_ffno_bottleneck`,
  `hybrid_resnet_convnext_bottleneck`, `hybrid_resnet_ffno_ptychoblock_encoder`,
  `hybrid_resnet_ptychoblock_ffno_encoder`) — FNO encoder + ResNet decoder family
- `spectral_resnet_bottleneck_net`, `spectral_resnet_bottleneck_linear_decoder` —
  Hybrid-ResNet shell with spectral ResNet bottleneck

To implement, configure, train, save, and reload a new architecture, follow the
[Custom PyTorch CDI Architecture Guide](custom_torch_architecture.md). The
generator-package README is a lower-level reference for existing modules.
Structural search knobs for the hybrid/ResNet family (`hybrid_downsample_steps`,
`hybrid_downsample_op`, `hybrid_resnet_blocks`, `hybrid_skip_style`,
`hybrid_encoder_*`, `spectral_bottleneck_*`, `ffno_encoder_blocks`) live on
Torch `ModelConfig` and are intentionally not owned by execution configuration.

**CNN geometry requirement:** use the canonical geometry/scaling matrix in
`docs/model_baselines.md`. In particular, a grouped `object_big=True` CNN must
keep `probe_big=True` so the decoder learns the full patch support.

### 3.3. Loss, Scheduler, and Sampling

- `TrainingConfig.torch_loss_mode`: `'poisson'` (physics-weighted Poisson NLL,
  default) or `'mae'` (amplitude-only MAE, `physics_weight=0`). Native-CLI flag:
  `--torch-loss-mode`.
- `TrainingConfig.scheduler`: `'Default'` (constant LR), `'Exponential'`,
  `'WarmupCosine'` (with `lr_warmup_epochs`, `lr_min_ratio`), or
  `'ReduceLROnPlateau'`. Note the native `ptycho_torch.train` CLI's `--scheduler`
  accepts a different, legacy choice set (`Default`, `Exponential`, `MultiStage`,
  `Adaptive`); the plateau/warmup schedulers are exposed by structured
  synthetic configuration and the unified `--torch-scheduler` flag.
- The resolved Torch training seed (`training.torch_training_seed`, derived
  via seed lineage for synthetic runs, with `subsample_seed`/`42` only as the
  legacy direct-call fallback) seeds `lightning.pytorch.seed_everything`;
  `subsample_seed` independently seeds data selection/grouping.
  `sequential_sampling=True` gives deterministic first-N grouping and preserves
  training order. With `False`, selection and per-epoch training shuffle are
  seeded; validation is never shuffled. The RAM and mmap rails use the
  same policy. Subsampled indices are retained on the in-memory
  `raw.sample_indices` attribute and asserted equal across backends; the
  retired `tmp/subsample_seed{X}_indices.txt` side-effect file is no longer
  written.

### 3.3.1. Data rails and batching

The maintained training workflow retains two storage choices but only one
batch conversion and native loader path:

```text
RawData -> grouped dict -> PtychoDataContainerTorch -> RAM dataset ┐
                                                                  ├─> shared batch emitter
standalone NPZ -> PtychoDataset TensorDict mmap -------------------┘   -> native DataLoader
```

The RAM and mmap datasets both use vectorized row fetching, then emit
`(tensor_dict, probe, probe_scaling)` with the same channel-first image and
coordinate layouts. The common emitter also selects per-experiment probes,
expands probe modes/channels, and attaches CI fields and frozen training
statistics. Plain grouped dictionaries from study adapters enter the RAM side;
they do not define another batching path.

`build_ptycho_loader` owns training batch size, seeded shuffle or explicit
sampler, worker/prefetch settings, pinning, and collation. The retained
`TensorDictDataLoader` name is a compatibility subclass of PyTorch's native
`DataLoader`, not a custom iterator. Under DDP, Lightning performs the sole
default sharding step. When a held-out mmap is supplied, it is loaded unchanged
as validation; only a run without one may split training data.

Legacy `ptycho_torch.api` and inference/reassembly loaders are not additional
maintained training rails. An already-built mmap enters the shared Lightning
service through `PrebuiltPtychoDataModule`; no second trainer-owned DataModule
or loader path remains.

### 3.4. Probe Masking

`config.model.probe_mask` (default `False`) enables a centered soft disk mask
(diameter `N/2`, Gaussian edge `sigma=1 px`) on the probe. Overrides:
`probe_mask_tensor` (explicit `(N, N)` mask; enables masking even when
`probe_mask=False`), `probe_mask_sigma`, `probe_mask_diameter`. CLI:
`--probe-mask/--no-probe-mask`, `--probe-mask-sigma`, `--probe-mask-diameter` on both
native CLIs.

### 3.4.1. Public object policy and legacy migration

New configuration uses three independent public fields:

```python
ModelConfig(
    object_layout="grouped_patches",
    training_canvas="relative_overlap",
    training_patch_weighting="central_mask",
)
```

The only supported layout/canvas pairs are
`single_patch`/`independent` and
`grouped_patches`/`relative_overlap`. PyTorch supports `central_mask`,
`uniform`, and `probe` weighting. TensorFlow supports `central_mask` only.
Unsupported pairs, partial pairs, contradictory dual old/new input, and
unsupported TensorFlow weighting fail before model construction.

`object_big` remains an optional deprecated input alias for external callers
and old configuration files. `False` maps to
`single_patch`/`independent`; `True` maps to
`grouped_patches`/`relative_overlap`. After resolution, the compatibility
Boolean is derived from `object_layout` and is written to legacy
`params.cfg['object.big']`. New code should not set `object_big`.

New Torch checkpoints and bundles use `torch-model-spec-v3` inside
`torch-artifact-v4`. The runtime load paths accept `torch-artifact-v3` and
`torch-artifact-v4`; pre-v3 bundles are recovered through
`python -m ptycho_torch.migrate_bundle`, which deterministically upgrades them
to the current era. The bundle version remains `2.0-pytorch` with exactly
`autoencoder` and `diffraction_to_obj`; TensorFlow bundle version `1.0` is
unchanged.

### 3.5. CNN Output and Physics-Forward Knobs

Four torch-`ModelConfig` knobs port the legacy-main CNN representation and physics as
opt-in modes. All default to the values that keep existing CNN/FNO/hybrid behavior
unchanged:

| Knob | Default | Opt-in value | Effect |
|---|---|---|---|
| `cnn_output_mode` | `'amp_phase'` | `'real_imag'` (Unsupervised-only) | CNN emits `(real, imag)` via `ScaledTanh` boxes (real ∈ (−0.8, 1.2), imag ∈ (−1.2, 1.2)); prerequisite for `rectangular_scaled`. Representability limit: unit-amplitude objects near `|phase| → π` are unreconstructable in this mode. |
| `use_shared_decoder` | `False` | `True` | Single shared decoder emitting `2*C_out` channels, split per branch; architecture-only knob. |
| `training_patch_weighting` | `'central_mask'` | `'probe'` (or `'uniform'`) | Public training-forward assembly policy for grouped patches: binary center mask vs `Σ|probe|²`-weighted (`'uniform'` isolates the code-path change without probe weighting). Distinct from the inference-only `InferenceConfig.patch_weighting`. |
| `physics_forward_mode` | `'amplitude'` | `'rectangular_scaled'` | Routes patches through `RectangularScaledDiffraction` (analytic real/imag intensity model with per-dataset trainable `s1`/`s2` unless `rect_s1s2_trainable=False`). Requires an effective `real_imag` generator output; for the CNN, select `cnn_output_mode='real_imag'`. Matching intensity-domain losses are selected automatically. |
| `rect_s1s2_init` | `'ones'` | `'dose_closure'` | Before fitting, either keep `s1=s2=1` or solve one shared startup gauge from the fixed representative 256-slot sample. `dose_closure` fails closed outside CI; the [core contract](../specs/spec-ptycho-core.md#ci-rectangular-gauge-initialization-normative) owns the sampling mechanics. |

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

Physical semantics of `s1`/`s2` and known residual differences: see the
rectangular-scaled diffraction entry in `docs/findings.md`.

`dose_closure` adopts a unit-object convention, so it is startup conditioning,
not physical probe calibration. Bare Torch `ModelConfig` defaults to `ones`;
the training-only `ci` and synthetic `hybrid-resnet-lines-ci` profiles default
to `dose_closure`. The field is sealed in `ModelSpec`. Its initialization
record is distinct from final learned `s1`/`s2` and inference VarPro.

One further amplitude-mode training knob (PROBE-RANK-001, 2026-07-12):
`ModelConfig.amplitude_physics_gain` (default `1.0`) multiplies the predicted
amplitude ONCE inside the amplitude-mode training forward. It is the explicit,
batch-size-independent replacement for the banned flat-probe layout's
accidental ×B gain (probe batches must follow the documented `(B, C, P, H, W)`
layout; sub-rank-5 probes raise `ProbeLayoutError`). The effective value is
recorded in the training-payload audit trail and Lightning hparams; it must be
finite and > 0, must be exactly `1.0` for `rectangular_scaled`/CI modes
(fail-closed), and is never applied at inference. Contract:
`docs/specs/spec-ptycho-torch-probe-layout.md`. Derive the legacy value once
from the exact sealed training input and forward normalization, using the
expression in `docs/model_baselines.md`, and share it across architectures and
legacy loss profiles. The historical value `16` is only the batch-16
broadcast-equivalent conditioner, not a physical normalization. This does not
change the `1.0` default or relax the required `1.0` value for rectangular/CI
scaling.

Two further knobs are **inference-only** (`InferenceConfig.patch_weighting`,
`InferenceConfig.varpro_scaling`): they affect only
`ptycho_torch.reassembly.reconstruct_image_barycentric` (the in-process reconstruction
path) and never touch training numerics. The native and unified inference CLIs
route to the public strict-load/mmap helper
`ptycho_torch.inference.reconstruct_npz_barycentric` when probe weighting or
VarPro is requested; otherwise they retain the uniform
`helper.reassemble_patches_position_real` compatibility route.

### 3.6. CNN Parity Diagnostic Knobs (Not A Baseline)

These knobs remain available for controlled parity diagnostics, but they do not
replace the canonical baseline in `docs/model_baselines.md`:

Here "preset" in older references describes an informal bundle of controls,
not a named configuration profile or registry entry.

| Knob | Where | Diagnostic value | Default |
|---|---|---|---|
| `cbam_encoder` | torch `ModelConfig` | `False` | `True` |
| `parity_init_scheme` | `PtychoPINN_Lightning` kwarg (`"default"` \| `"tf_glorot"`) | `"tf_glorot"` | `"default"` (kaiming) |
| `scheduler` | `TrainingConfig` | `"ReduceLROnPlateau"` | `"Default"` |

An additional default-off mechanism, `parity_scale_mode`
(`PtychoPINN_Lightning` kwarg; `"off"` \| `"tied"` \| `"input"` \| `"output"` \|
`"fixed"`), controls the TF-parity global intensity scale; resolved study
records carry it through the exact payload adapter into the shared service. It
is driven from
`scripts/studies/varpro_probe_ablation_runner.py`
(`--cbam-encoder on|off`, `--parity-init-scheme`, `--scheduler`).

These controls, and evidence produced by `hybrid_resnet`, do not establish a
quality threshold or baseline for a count-Poisson `cnn`; architecture-specific
claims require evidence from a run under that exact contract.

Cautions:
- CBAM-off did not fix the unrelated support-on Task 30 failure. In the N=128
  count-Poisson parity study, however, CBAM-off was the dominant control (3/5
  escapes alone); the complete controls reached 6/10 while retaining a 2/10
  flat-collapse tail.
- Do NOT set `intensity_scale_trainable=True` alongside the parity kwargs — the dead
  `IntensityScalerModule` machinery silently overwrites the input-side parity scale
  (see the dead intensity-scaler entry in `docs/findings.md`).
- Root-cause record: the N=128 flat-amplitude collapse entry in `docs/findings.md`;
  evidence: `docs/plans/2026-07-08-cnn-n128-tf-parity.md`.

## 4. User-Facing Workflows

### 4.1. Synthetic Generation Through Evaluation (Recommended)

`ptycho_synthetic` is the supported entry point for new synthetic PyTorch
work. With no stage selection it runs all four stages in order:

```text
simulate -> train -> strict bundle reload -> mmap barycentric reconstruct -> evaluate
```

The detector simulation leaf still uses TensorFlow, but it runs in a
CUDA-hidden child process. Object/probe/coordinate production and workflow
orchestration use NumPy/Python, while all model training, inference, and
reassembly are PyTorch.

The complete coherent default profile is a 50-epoch GS1 Hybrid ResNet run with
an ideal probe:

```bash
ptycho_synthetic \
  --profile hybrid-resnet-lines \
  --output-root outputs/synthetic_hybrid_resnet_gs1
```

Important default-profile values are:

| Area | Default |
| --- | --- |
| Geometry/model | `N=128`, `gridsize=1`, `C=1`, `hybrid_resnet`, real/imag output |
| Generated data | 4,096 train and 1,024 test raw patterns, lines object, seed 3, normalized-amplitude legacy contract |
| Training sampling | select all 4,096 train frames; 1,024 train groups; 1,024 validation groups |
| Training | 50 epochs, batch 16, Adam `2e-4`, `ReduceLROnPlateau`, MAE |
| Probe | ideal probe, `smooth:0.5|pad_preserve:128`; simulation and model masks off |
| Reconstruction | probe-weighted barycentric assembly, VarPro on, `groups_per_center=1` |
| Runtime | accelerator auto, one device, FP32, deterministic, zero workers, CSV logger |

For the sealed Hybrid ResNet GS1/C1 five-epoch recipe with the checked-in
Run1084 probe, use:

```bash
ptycho_synthetic \
  --profile hybrid-resnet-lines \
  --output-root outputs/synthetic_hybrid_resnet_gs1_5ep_quality \
  --gridsize 1 \
  --epochs 5 \
  --batch-size 16 \
  --seed 3 \
  --probe-source custom \
  --probe-path datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-transform 'pad_extrapolate:128|smooth:0.5' \
  --train-patterns 4489 \
  --test-patterns 729 \
  --train-raw-selection 4489 \
  --training-groups 4489 \
  --validation-groups 729 \
  --neighbor-count 1 \
  --neighbor-pool-size 1 \
  --groups-per-center 1 \
  --accelerator cuda \
  --devices 1 \
  --precision 32-true \
  --workers 0 \
  --logger csv \
  --deterministic
```

The GS2/C4 normalized-amplitude and C4-CI count-intensity sealed recipes use
the same workflow; their exact contracts and selectors are in
`docs/TESTING_GUIDE.md`.

The same run may be expressed as a structured JSON, TOML, or YAML file. For
example, save this as `configs/synthetic_gs1.yaml`:

```yaml
profile: hybrid-resnet-lines
simulation:
  gridsize: 1
  train_patterns: 4489
  test_patterns: 729
  probe:
    source: custom
    source_path: datasets/Run1084_recon3_postPC_shrunk_3.npz
    transform_pipeline: "pad_extrapolate:128|smooth:0.5"
training:
  epochs: 5
  train_raw_selection: 4489
  training_groups: 4489
  validation_groups: 729
  neighbor_count: 1
  neighbor_pool_size: 1
inference:
  groups_per_center: 1
workflow:
  output_root: outputs/synthetic_hybrid_resnet_gs1_5ep_quality
  accelerator: cuda
  devices: 1
  precision: 32-true
  num_workers: 0
  logger_backend: csv
  deterministic: true
```

CLI values override file values, and file values override the named profile.
Stage selection supports reproducible partial execution. For example, train
first and reconstruct later with exactly the same scientific configuration:

```bash
ptycho_synthetic --config configs/synthetic_gs1.yaml \
  --stages simulate,train

ptycho_synthetic --config configs/synthetic_gs1.yaml \
  --stages reconstruct,evaluate
```

Stages must be a duplicate-free ordered subsequence of `simulate`, `train`,
`reconstruct`, and `evaluate`. A skipped predecessor must already be marked
complete in `stage_manifest.json`; the configuration namespaces consumed by a
required reused stage must match `resolved_workflow.json`. A required identity
mismatch or partial selected-stage artifact fails closed and should be run
under a new output root rather than overwritten. An incompatible completed
downstream stage not required by the current selection is pruned from the
manifest.

The current `synthetic-stage-manifest-v2` training contract includes both the
bundle and `training/training_summary.json`. That summary is strictly parsed as
a v1 or v2 initialization record, compared with the backend result on fresh
completion, and checked against the resolved mode on reuse. Fresh runs write
v2; historical v1 records remain strict prefix-era history. Version-1 manifest
roots lack the summary artifact and require a new output root or retraining.

The output contract is:

```text
OUTPUT/
  invocation.json
  invocation.sh
  resolved_workflow.json
  stage_manifest.json
  stage_logs/
    simulate_request.json
    simulate.log
  datasets/
    source.npz
    train.npz
    test.npz
    manifest.json
  training/
    wts.h5.zip
    training_summary.json
    effective_runtime.json
    checkpoint_selection.json
    checkpoints/
      <monitored-best>.ckpt
      last.ckpt
    lightning_logs/
  reconstruction/
    reconstruction.npz
    metrics.json
    diagnostics.json
    comparison.png
```

#### Sampling and reconstruction identity

Synthetic NPZs use `flat_acquisition_v1`: each row is one raw scan position,
even when `gridsize > 1`. The shared generic loader is the sole owner that
forms `C = gridsize ** 2` channels. The four similarly named controls are
independent:

| Control | Meaning |
| --- | --- |
| `--train-raw-selection` | Raw train frames selected before grouping; persisted as the training `DataConfig.n_raw_frames_selected` |
| `--training-groups` | Exact grouped samples built for the train container |
| `--validation-groups` | Exact grouped samples built independently from the complete test acquisition |
| `--groups-per-center` | Reconstruction-only repeated neighbor groups per valid scan center |

`--training-groups` and `--validation-groups` need not be equal. The former is
always bounded by the selected train-frame count; the latter is bounded by the
test raw-pattern count. Reconstruction starts from the strictly loaded
persisted `DataConfig` and threads `groups_per_center` to the dataset
constructor as an explicit runtime argument (no dataclass field round-trip).
It never rewrites training identity.

The synthetic training request always calls the shared generic trainer with
`do_stitching=False`. Generic stitching reduces grouped predictions at their
centers, which loses the global all-C-channel evidence needed for a valid GS2
quality reconstruction. The separate reconstruction stage strictly reloads
`training/wts.h5.zip`, stages only the held-out NPZ into a fresh mmap
workspace, and performs probe-weighted barycentric/VarPro assembly across all
channels and coordinates.

For parameterized multi-arm execution, `ptycho_study` composes public synthetic
workflow configurations as described in §9.

### 4.2. Unified CLI (backend selection)

```bash
ptycho_train --train_data_file datasets/my_train.npz \
  --output_dir outputs/my_run \
  --backend pytorch \
  --torch-accelerator auto --torch-logger csv
```

Torch execution flags on the unified scripts: `--torch-accelerator`,
`--torch-logger`, `--torch-learning-rate`, `--torch-scheduler`, `--torch-num-workers`,
`--torch-deterministic`, `--torch-enable-checkpointing`,
`--torch-checkpoint-save-top-k`, `--torch-accumulate-grad-batches`. Dispatch happens
in `ptycho/workflows/backend_selector.py` (see §7).

The `--torch-*` prefix identifies the backend lane, not the configuration
owner. The CLI builder sends accelerator, workers, logging, and checkpoint
values to `ExecutionRequest`, while learning rate, scheduler, clipping, and
accumulation form a separate Torch `TrainingConfig` patch.

### 4.3. Native CLI

```bash
CUDA_VISIBLE_DEVICES="0" python -m ptycho_torch.train \
  --train_data_file datasets/my_train.npz \
  --test_data_file datasets/my_test.npz \
  --output_dir outputs/my_run \
  --n_images 512 --gridsize 2 --batch_size 16 --max_epochs 50 \
  --accelerator cuda --logger csv --quiet
```

Flags (`python -m ptycho_torch.train --help` is authoritative):

| Group | Flags |
|---|---|
| Data/model | `--train_data_file`, `--test_data_file`, `--output_dir`, `--n_images` (number of groups), `--gridsize`, `--batch_size`, `--max_epochs` |
| CI initialization | `--profile ci`, `--rect-s1s2-init {ones,dose_closure}` |
| Execution | `--accelerator {auto,cuda,cpu,tpu,mps}`, `--deterministic/--no-deterministic`, `--num-workers`, `--quiet` |
| Optimization | `--learning-rate`, `--scheduler {Default,Exponential,MultiStage,Adaptive}`, `--accumulate-grad-batches` |
| Checkpointing | `--enable-checkpointing/--disable-checkpointing`, `--checkpoint-save-top-k`, `--checkpoint-monitor` (default `val_loss`, auto-aliased to the model's actual metric, e.g. `poisson_val_loss`), `--checkpoint-mode`, `--early-stop-patience` |
| Loss/probe | `--torch-loss-mode {poisson,mae}`, `--probe-mask/--no-probe-mask`, `--probe-mask-sigma`, `--probe-mask-diameter` |
| Logging | `--logger {csv,tensorboard,mlflow,none}`, `--log-patch-stats`, `--patch-stats-limit` |
| Deprecated | `--device` (→ `--accelerator`), `--disable_mlflow` (→ `--logger none` + `--quiet`) |

The CLI builds an `ExecutionRequest` and a separate Torch training patch through
`ptycho_torch/cli/shared.py`
(`build_execution_request_from_args`, `build_training_config_patch_from_args`,
`validate_paths`). The config factory capability-resolves the request to
`PyTorchExecutionConfig` and performs the required compatibility projection for
this native entry point.

This native CLI does not accept `--config`. Its
`--rect-s1s2-init` argparse default is `None`: omission preserves the
training-only `ci` profile's `dose_closure` default or the bare `ones` default,
while an explicit spelling is forwarded as the caller override. Use
`ptycho_synthetic --config` for the structured synthetic workflow.

### 4.4. Programmatic

```python
from ptycho.raw_data import RawData
from ptycho_torch.execution_request import ExecutionRequest
from ptycho_torch.workflows.components import run_cdi_example_torch

train_data = RawData.from_file(str(config.train_data_file))
test_data = RawData.from_file(str(config.test_data_file)) if config.test_data_file else None
execution_request = ExecutionRequest(
    values={"accelerator": "cuda", "num_workers": 0},
    explicit_fields=frozenset({"accelerator", "num_workers"}),
)
torch_training_overrides = {
    "learning_rate": 1e-3,
    "scheduler": "Default",
    "gradient_clip_val": None,
    "accum_steps": 1,
}

amplitude, phase, results = run_cdi_example_torch(
    train_data, test_data, config,
    do_stitching=True,            # Synthetic workflow always sets this False
    execution_config=execution_request,
    overrides=torch_training_overrides,
)
```

`run_cdi_example_torch` normalizes the data into `PtychoDataContainerTorch`, seeds and
instantiates `PtychoPINN_Lightning`, runs `Trainer.fit()` (deterministic, checkpoints
under `{output_dir}/checkpoints/`), persists the bundle, and — when
`do_stitching=True` — runs Lightning prediction and reassembles the image
(`flip_x`/`flip_y`/`transpose` args control coordinate transforms, `M` the stitch
window). Component contract: `docs/architecture_torch.md` §6.

Study callers that already own a `PtychoDataset` mmap build it once, wrap it in
`PrebuiltPtychoDataModule`, and pass that DataModule with the resolved payload
to the same shared service. Held-out evaluation mmaps remain separate from the
training validation split.

## 5. Checkpoints, Persistence, Reproducibility

- **Determinism:** `deterministic=True` + `seed_everything(<resolved torch training seed>)` (the seed resolved in §3.3, not `subsample_seed`).
- **Checkpoints:** monitored runs keep both the selected best checkpoint and
  `{output_dir}/checkpoints/last.ckpt`; `last.ckpt` is the recovery state.
  `checkpoint_selection.json` records the selection metric, score, epoch,
  checkpoint digest, and recovery path. Checkpoint hyperparameters reload
  without manual config kwargs.
- **Bundle:** `{output_dir}/wts.h5.zip` persists the declared selected state.
  With `checkpoint_save_top_k > 0`, that is the monitored best checkpoint;
  with top-k disabled or checkpointing off, it is the final in-memory state. The
  `intensity_scale` is captured (learned value if trainable, else the spec fallback
  `sqrt(nphotons)/(N/2)`) and stored in the bundle's `params.json`, so inference uses
  the same normalization as training.
- **Resolved identity:** the bundle's persisted model, data, training, and
  enclosing artifact identity—including `measurement_domain` and
  `scale_contract_version`—is authoritative at inference and prevents domain
  drift. Loading does not rerun or require a named training profile.
- **Gauge initialization summary:** the shared Torch training path used by the
  supported public training entry points write
  `{output_dir}/training_summary.json`. Its exact fields are
  `schema_version`, `mode`, `solved_gauge`, `method`, and `sampled_patterns`.
  Fresh records use `rect-s1s2-initialization-v2`; `ones` records `1.0`,
  `unit_default_no_solve`, and zero patterns, while `dose_closure` records
  `dose_closure_seeded_uniform_unit_object` and exactly 256 detector slots.
  Strict v1 reading is retained for prefix-era records. The
  [core contract](../specs/spec-ptycho-core.md#ci-rectangular-gauge-initialization-normative)
  owns the full schema and sampling rules. Under DDP, only global rank zero publishes it, using atomic
  replacement, and every rank enters the live strategy barrier before fitting.
- **Synthetic strict reload:** the synthetic reconstruct stage requires a
  nonempty `training/wts.h5.zip` and validates the serialized `ModelSpec`,
  Data/Model/Training/Inference configs, scaling identity, architecture,
  geometry, and channel counts before creating its mmap workspace.
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
By default this CLI preserves uniform compatibility stitching. Passing
`--patch-weighting probe` or `--varpro-scaling` routes through
`reconstruct_npz_barycentric`, which strictly reloads the bundle, reconstructs
the full held-out scan through mmap, and calls the barycentric reassembler.
`--groups-per-center` controls only that runtime route and does not alter the
persisted training selection.

**Programmatic reconstruction seams** (embedder-facing, no CLI, no
`params.cfg`):

```python
from ptycho_torch.inference import (
    reconstruct_from_dataset,     # dataset-in kernel
    reconstruct_from_arrays,      # arrays-in seam
    ReconstructionRuntimeParams,
)

# Arrays-in: a loaded model + in-memory flat-acquisition NPZ arrays.
result = reconstruct_from_arrays(
    model,
    arrays,  # {"diff3d": ..., "xcoords": ..., "ycoords": ..., "probeGuess": ...}
    runtime_params=ReconstructionRuntimeParams(
        data_config=model.data_config,
        training_config=model.training_config,
        inference_config=inference_config,
        source_metadata={},
    ),
    workspace=Path("mmap_workspace"),
)
```

`reconstruct_from_arrays` stages the in-memory arrays into a caller-provided
mmap workspace (writing one NPZ bridge file), then delegates to
`reconstruct_from_dataset`. `runtime_params.data_config`,
`runtime_params.training_config`, and `runtime_params.source_metadata` are
derived during staging (device/num_workers are threaded into the training
config); the caller supplies `inference_config` and the
`precision`/`quiet`/`enforce_ci_varpro`/`compute_count_metrics` knobs.


**Device handoff:** when chaining training and custom inference
in one process, do not assume the post-`fit()` module is still on the training
accelerator — resolve the target device explicitly and call `model.to(device)` before
the forward loop (see `docs/DEVELOPER_GUIDE.md` §2.6).

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

Both backends keep a `workflows/components.py` facade by design: the TF and
torch packages mirror each other's public import surface
(`ptycho.workflows.components` / `ptycho_torch.workflows.components`), and the
shared basename is the parity signal — the two files are shim-pure re-export
facades over their respective implementation slabs, not an accidental
collision. They live in different packages, so there is no import ambiguity.


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
  Maintained whole-model MLflow load helpers immediately revalidate
  `model.model_config.rect_s1s2_init` after unpickling, so retired `data`
  artifacts fail at the same semantic boundary as current configs and
  checkpoints.
- `--logger none` + `--quiet`: fully silent smoke runs.
- Loss/metric parity: training logs `amp_inv_mae_epoch` (measurement domain) and
  `amp_mae_tf_scale_epoch` (TF-normalized domain) so Poisson-vs-MAE curves compare
  directly against TensorFlow amplitude MAE.

## 9. Study Runners

`ptycho_study` is the public multi-arm composer. It writes each resolved arm
configuration and delegates the arm to the configured public runner; it does
not provide a parallel training implementation. See
[`hydra_studies.md`](hydra_studies.md) for composition, resume, and provenance
semantics.

Retained study-specific CLIs have narrower roles:

- `scripts/studies/torch_ablation_driver.py` runs manifest-driven Torch
  ablations.
- `scripts/studies/varpro_probe_ablation_runner.py` runs the VarPro and probe
  weighting ablations described in §3.6.
- `scripts/studies/collate_study_metrics.py` collates completed arm metrics;
  `scripts/studies/render_study_comparison.py` renders completed-arm
  comparisons.

Use `ptycho_synthetic` for a single supported synthetic run.

## 10. Constraints and Known Pitfalls

- **Gradient accumulation:** `PtychoPINN_Lightning` uses manual optimization, which is
  incompatible with gradient accumulation — `--accumulate-grad-batches > 1` raises a
  `RuntimeError` before training. Keep the default (`1`).
- **Supervised mode** (`model_type='supervised'`) requires `label_amp` /
  `label_phase` keys in the NPZ; experimental datasets lack them and fail dataloader
  validation. Use PINN mode or generate labeled synthetic data.
- **Gridsize > 1 support** is architecture-gated in the public synthetic
  workflow (`cnn`, `hybrid_resnet`); `ptycho_study` arms that delegate to that
  workflow enforce the same restriction.
- **N=128 count-Poisson CNN is collapse-prone.** The complete TF-parity
  controls reduce but do not eliminate the risk; `hybrid_resnet` remains the
  recommended alternative (§3.6).
- **`intensity_scale_trainable=True`** conflicts with the parity scale path (§3.6).
- Shape mismatches at load time usually mean the `update_legacy_dict(params.cfg,
  config)` bridge was skipped — see
  `docs/debugging/TROUBLESHOOTING.md`.

## 11. Legacy-Bundle Migration

Pre-JSON or metadata-free PyTorch archives (`manifest.dill` + per-model
`params.dill`, or a JSON manifest without a sealed identity) are recovered
offline with the era-detecting migrator:

```bash
python -m ptycho_torch.migrate_bundle SOURCE_DIR OUT_DIR
```

Both arguments are directories holding a `wts.h5.zip`. The migrator detects
the source era (metadata-free legacy, ci-entrypoints-v1, v1, v2, v3), rebuilds
the model from the archived weights, seals a fresh current-era
(`torch-artifact-v4`) identity, and writes the migrated archive to `OUT_DIR`.
Migrating an already-current bundle is a no-op. Errors name the missing
bundle or the offending member; the module import itself is torch/dill-free.

## 12. Testing

- Fast suite: `pytest tests/torch -m "not slow"` (this is what the public-main CI
  gate runs).
- End-to-end regression (train→save→load→infer, GPU-pinned):

```bash
CUDA_VISIBLE_DEVICES="0" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

- Synthetic fast contracts:

```bash
python -m pytest tests/scripts/test_cli_entrypoint_bootstrap.py \
  tests/torch/test_synthetic_hybrid_resnet_gs2_integration.py \
  -m "not integration" -q
```

- Hybrid GS1, GS2, and C4-CI quality changes must pass their respective
  one-epoch public-command preflight before calibration. Each fixture update
  uses exactly two fitting runs and one unchanged holdout, scores raw
  reconstruction arrays, and requires both automated `comparison.png`
  integrity checks and recorded manual visual adjudication. The fixtures are
  immutable at test runtime and apply only to their recorded CUDA/software
  fingerprint. Exact preflight, candidate, and sealed five-epoch selectors
  plus the fail-closed debugging order live in `docs/TESTING_GUIDE.md`. The
  separate CNN C4/count-intensity one-epoch smoke remains operational
  coverage; the Hybrid-ResNet C4-CI contract carries its own sealed envelope.

- Visual parity evidence for pipeline changes:
  `python scripts/tools/patch_parity_helper.py --tf-npz ... --torch-npz ...`
  (aligns shared `sample_indices`, writes comparison grids under `tmp/patch_parity/`).

Commands, selectors, and evidence requirements: `docs/TESTING_GUIDE.md`.

---

**Related Documentation:**
- <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref> — architectural principles and anti-patterns
- <doc-ref type="guide">docs/architecture_torch.md</doc-ref> — torch architecture and component contracts
- <doc-ref type="spec">docs/specs/spec-ptycho-config-bridge.md</doc-ref> — TF ↔ torch config mapping
- <doc-ref type="spec">specs/ptychodus_api_spec.md</doc-ref> — backend dispatch and execution-config contracts (§4.8–4.9)
- <doc-ref type="contract">docs/specs/spec-ptycho-core.md</doc-ref> — NPZ data contract
- <doc-ref type="guide">docs/findings.md</doc-ref> — known-issue registry for the entries cited above
