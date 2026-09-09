# PyTorch Workflow Guide

This guide is the authority for configuring and running the PyTorch backend of
PtychoPINN: the Lightning-based training stack under `ptycho_torch/` with a generator
core builder for architecture selection. PyTorch (torch ≥ 2.2) is a mandatory dependency.

## 1. Overview

There are four ways to run the backend, from highest-level to lowest:

| Entry point | Use when |
| --- | --- |
| `ptycho_synthetic` | You want the supported synthetic simulate → train → strict-reload → mmap barycentric reconstruct → evaluate workflow |
| Unified CLIs: `ptycho_train` / `ptycho_inference` with `--backend pytorch` | You want the backend-agnostic workflow (same flags as TensorFlow, plus `--torch-*` execution flags) |
| Native CLIs: `python -m ptycho_torch.train` / `python -m ptycho_torch.inference` | You want direct control of torch execution flags |
| Programmatic: `ptycho_torch.train.train` / `ptycho_torch.inference.reconstruct` | You have an NPZ and want a model path followed by amplitude/phase arrays |

Key properties:

- **Training configuration** is resolved by the existing Torch factory. Direct callers
  pass its canonical keys in the `settings` mapping; the complete checked table
  is in [Configuration](../CONFIGURATION.md#canonical-programmatic-torch-training-settings).
  A legacy projection is created only for a remaining legacy consumer.
- **Training** runs through `PtychoPINN_Lightning` (`ptycho_torch/model.py`) with
  deterministic settings, Lightning checkpointing, and the full physics loss for every
  architecture.
- **Data contract** uses the shared standalone NPZ keys and shapes while
  supporting both legacy normalized-amplitude measurements and
  `ci_intensity_v2` count-intensity measurements. The resolved configuration
  must match the stored measurement domain; normative schema:
  <doc-ref type="contract">specs/data_contracts.md</doc-ref>.
- **Configuration is not a quality ranking.** This guide documents supported
  settings; validate reconstruction quality on the intended acquisition.

### Configuration and identity lifecycle

Training seals identity; inference restores it:

```text
authored training settings
  -> resolved payload (TrainingPayload)                     [create_training_payload]
  -> sealed identity (ModelSpec)                             [checkpoint / bundle write]
  -> restored identity (strict bundle/checkpoint decode)     [decode_checkpoint_hparams + load_inference_bundle_torch]
```

Training consumers use the resolved payload. Inference calls `reconstruct`
with the strict bundle identity, validated inference knobs, and separately
resolved runtime settings; it does not construct another payload or model config.

### Training entry-point convergence

The direct Python API, native Torch CLI, and unified CLI's Torch branch all use
the same public `train` body. Synthetic training keeps its specialized truth,
seed, and batch-order preparation local and calls the same retained training
component directly:

```text
Python/Jupyter ─┐
native CLI ─────┼─> train -> Torch resolver -> RawData -> train_cdi_model_torch
unified CLI ────┘
synthetic -> specialized preparation -----------------> train_cdi_model_torch
```

`train` resolves before creating its output directory, loads and if necessary
rescales the full acquisition, applies the optional raw-frame cap, then lets the
existing container and Lightning service own grouping, training, and the
bundle. It returns the nonempty `wts.h5.zip` path directly.

## 2. Prerequisites

- `pip install .` installs torch ≥ 2.2, `lightning`, and `tensordict` automatically.
  For a specific CUDA build, install PyTorch manually first
  ([instructions](https://pytorch.org/get-started/locally/)), then `pip install .`
- Input NPZ files conforming to `specs/data_contracts.md`. Legacy files
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
`specs/ptychodus_api_spec.md` §4.9. Direct Torch users pass strings and the
resolver settings mapping to `train` (§4.4); they do not construct public
`TrainingConfig`, `Path`, or `params.cfg`. `update_legacy_dict` is required only
inside an explicitly declared TensorFlow or surviving legacy-component
boundary.

#### Training-only CI profile

The direct `train` function defaults to `profile="ci"`; the native Torch CLI
delegates with the same default. Both select the same named starting bundle. It
locks `scale_contract_version=ci_intensity_v2`,
`measurement_domain=count_intensity`,
`physics_forward_mode=rectangular_scaled`, `torch_loss_mode=poisson`, and
`loss_function=Poisson`. Contradictions fail closed; the overrideable
`rect_s1s2_init` profile default is `dose_closure`, and an explicit `ones`
selects unit initialization. With `profile=None`, ordinary resolution applies
without a named bundle and the bare model default remains `ones`.

This training-only profile is distinct from the synthetic runner's
`--profile cnn-lines-ci`, which also chooses the simulation recipe
and defaults to `dose_closure`. Persisted model, data, training, inference,
and artifact identity controls inference, so loading does not require selecting either
profile name again. See the
[configuration guide](../CONFIGURATION.md#torch-training-only-ci-profile) for
the complete field tables.

For metadata-free input, omit `nphotons` when the diffraction is already in
the expected count scale. Supply positive finite `nphotons` when the stored
values are normalized amplitudes that must be converted. Declared CI counts
are never scaled twice.

### 3.2. Architecture Selection

`config.model.architecture` routes through the core builder in
`ptycho_torch/model.py`. Every architecture
trains through `PtychoPINN_Lightning` with the same physics pipeline. Selectable
architectures:

- `cnn` (default) — U-Net-style CNN encoder/decoder pair
- `fno`, `fno_vanilla`, `ffno` — Fourier-operator stacks (see `fno_modes`,
  `fno_width`, `fno_blocks`, `fno_cnn_blocks`, `fno_input_transform`)
- `neuralop_uno` — wraps external `neuraloperator==2.0.0` U-NO (locked to the
  Lines128 CDI path: `N=128`, `gridsize=1`, `C=1`, `real_imag`)
- `fno_li` — Li-style FNO with a pointwise linear path
- `vit` — isotropic ViT (`vit_patch_size`, `vit_width`, `vit_depth`, `vit_heads`)

To implement, configure, train, save, and reload a new architecture, follow the
[Custom PyTorch CDI Architecture Guide](custom_torch_architecture.md). The
generator-package README is a lower-level reference for existing modules.
Topology settings live on Torch `ModelConfig`, not execution configuration.

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
`--probe-mask/--no-probe-mask`, `--probe-mask-sigma`, `--probe-mask-diameter` on the
native training CLI. Inference uses the checkpoint's probe configuration.

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

New Torch checkpoints and bundles use `torch-model-spec-v4` inside
`torch-artifact-v5`. Runtime loading accepts artifact v3 through v5; v1/v2
bundles require `python -m ptycho_torch.migrate_bundle`. Every pre-v5 upgrade
is C1-only; pre-v5 C>1 payloads fail closed under the centered-nearest
contract. See the [persistence contract](../../specs/ptychodus_api_spec.md).
The bundle version remains `2.0-pytorch` with exactly
`autoencoder` and `diffraction_to_obj`; TensorFlow bundle version `1.0` is
unchanged.

### 3.5. CNN Output and Physics-Forward Knobs

Five torch-`ModelConfig` knobs port the legacy-main CNN representation and physics as
opt-in modes. All default to the values that keep existing CNN/FNO behavior
unchanged:

| Knob | Default | Opt-in value | Effect |
|---|---|---|---|
| `cnn_output_mode` | `'amp_phase'` | `'real_imag'` (Unsupervised-only) | CNN emits `(real, imag)` via `ScaledTanh` boxes (real ∈ (−0.8, 1.2), imag ∈ (−1.2, 1.2)). Representability limit: unit-amplitude objects near `|phase| → π` are unreconstructable in this mode. |
| `use_shared_decoder` | `False` | `True` | Single shared decoder emitting `2*C_out` channels, split per branch; architecture-only knob. |
| `training_patch_weighting` | `'central_mask'` | `'probe'` (or `'uniform'`) | Public training-forward assembly policy for grouped patches: binary center mask vs `Σ|probe|²`-weighted (`'uniform'` isolates the code-path change without probe weighting). Distinct from the inference-only `InferenceConfig.patch_weighting`. |
| `physics_forward_mode` | `'amplitude'` | `'rectangular_scaled'` | Routes directly learned real/imag textures through `RectangularScaledDiffraction`; effective `real_imag` output is required. `s1`/`s2` scale object-plane components before probe multiplication and the FFT. Matching intensity-domain losses are selected automatically. |
| `rect_s1s2_init` | `'ones'` | `'dose_closure'` | Before fitting, either keep `s1=s2=1` or solve one shared startup gauge from the fixed representative 256-slot sample. `dose_closure` fails closed outside CI; the [core contract](../../specs/data_contracts.md) owns the sampling mechanics. |

`cnn_output_mode` and `physics_forward_mode` are distinct but compatibility is
constrained. The first controls how the CNN decoder parameterizes the object;
its choices are `amp_phase` and `real_imag`. Other architectures use
`generator_output_mode`, which additionally supports `amp_phase_logits`. The
model resolves the effective output first and fails construction if it is
incompatible with the selected forward.

The supported combinations are:

| Effective generator output | `physics_forward_mode='amplitude'` | `physics_forward_mode='rectangular_scaled'` |
|---|---|---|
| `amp_phase` | Supported legacy amplitude-domain path | Rejected: CI requires directly learned real/imag textures |
| `amp_phase_logits` | Supported amplitude/phase-derived registered-generator path | Rejected: CI requires directly learned real/imag textures |
| `real_imag` | Supported representation ablation using the amplitude-domain forward | Supported rectangular intensity path used by CI |

`rectangular_scaled` defines `O = s1*a_tilde + i*s2*b_tilde`, where the
generator directly learns normalized object-plane real and imaginary textures.
Both `amp_phase` and `amp_phase_logits` are amplitude/phase-derived. Converting
their learned `A, phi` fields into `A*exp(i*phi)` does not produce those
independent rectangular textures. The scales act before probe multiplication
and the FFT; their effect on detector intensity is downstream and depends on
the acquisition gauge and probe normalization. The CI profiles select the
required `real_imag` pairing.

Physical semantics of `s1`/`s2` and known residual differences: see the
rectangular-scaled diffraction entry in `docs/findings.md`.

`dose_closure` adopts a unit-object convention, so it is startup conditioning,
not physical probe calibration. Bare Torch `ModelConfig` defaults to `ones`;
the training-only `ci` and synthetic `cnn-lines-ci` profiles default
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
route general NPZ reconstruction through public
`ptycho_torch.inference.reconstruct`; fixed-pitch synthetic tiled
reconstruction retains its specialized path.

With `patch_weighting='probe'`, barycentric stitching overlap-adds predicted
object patches with the cropped probe intensity `sum_modes(|P|^2)` and divides
by that accumulated weight canvas. This is reconstruction weighting, not a
downstream metric weight: ordinary object metrics consume the finalized canvas
unweighted unless their own contract says otherwise. A study that locks probe
weighting should serialize and verify the setting and weight canvas; it must
reject a uniform or implicit fallback rather than silently score it.

A third, runtime-only knob lives outside `InferenceConfig` because it is not
bundle identity: `patch_phase_alignment` (`reconstruct(...,
patch_phase_alignment='overlap')`, `ReconstructionRuntimeParams`, CLI
`--patch-phase-alignment overlap` on both the native and unified inference
doors; default `'none'`). A diffraction pattern cannot see the constant phase
of its own patch, so the network's patches carry independent phase constants
and the plain stitch averages them away. With `'overlap'` the assembler fits
one constant per patch from the patch overlaps (seeded by spectral
synchronisation of the pairwise overlaps, then Jacobi refinement of the
probe-weighted patch-versus-consensus disagreement until the RMS wrapped step
is below 1e-3 rad, gauge fixed to zero mean rotation) and
stitches the rotated, VarPro-calibrated patches; the weight
canvas is unchanged and `prescale_canvas` stays unaligned. It holds every
cropped patch in memory during the fit. See
`ptycho_torch/reassembly_phase_alignment.py` and the reassembly clause in
`docs/specs/spec-ptycho-workflow.md`.

## 4. User-Facing Workflows

### 4.1. Synthetic Generation Through Evaluation

`ptycho_synthetic` runs simulation, training, strict bundle reload, and reconstruction.
Use `ptycho_synthetic --help` for stage selection and dataset settings.

```bash
ptycho_synthetic --profile synthetic-lines --output-root outputs/synthetic_lines
```

Use a CUDA device for training. For reconstruction-quality validation, acquire
many overlapping patterns from one object, stitch its reconstruction, and compare
amplitude and phase against that object's truth. Mixed-object patches are not an
object-quality benchmark.

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
| Data/model | `--train_data_file`, `--test_data_file`, `--output_dir`, `--n_images` (native-door spelling mapped once to canonical `training_groups`), `--gridsize`, `--batch_size`, `--max_epochs` |
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
`PyTorchExecutionConfig`; native Torch resolution does not read, mutate, or
populate legacy `params.cfg`.

This native CLI does not accept `--config`. Its
`--rect-s1s2-init` argparse default is `None`: omission preserves the
training-only `ci` profile's `dose_closure` default or the bare `ones` default,
while an explicit spelling is forwarded as the caller override. Use
`ptycho_synthetic --config` for the structured synthetic workflow.

### 4.4. Programmatic

```python
from ptycho_torch.inference import reconstruct
from ptycho_torch.train import train

data = "datasets/Run1084_recon3_postPC_shrunk_3.npz"
model = train(data, "outputs/run1084_cnn", {
    "architecture": "cnn",
    "training_groups": 256,
    "nphotons": 1e9,
    "epochs": 1,
})
result = reconstruct(model, data)
```

`train` returns the nonempty bundle path consumed directly by `reconstruct`.
Use `help(train)` for common settings and the
[canonical resolver table](../CONFIGURATION.md#canonical-programmatic-torch-training-settings)
for every accepted field and its owner.

Advanced in-memory/Ptychodus component callers may still pass an already
prepared `RawData`, container, or `PrebuiltPtychoDataModule` to the retained
component seam. That is not a second ordinary dataset API; held-out evaluation
mmaps remain separate from the training validation split.

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
  [core contract](../../specs/data_contracts.md)
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
  --accelerator cuda --quiet
```

Additional flags: `--num-workers`, `--inference-batch-size` (default: reuse training
batch size), `--log-patch-stats`, and `--patch-stats-limit`. A legacy MLflow-run mode
(`--run_id`, `--infer_dir`, `--file_index`) still exists but is not the default path.
The CLI routes through `reconstruct`, which strictly reloads the bundle,
reconstructs the full held-out scan through mmap, and calls the barycentric
reassembler.
`--groups-per-center` controls only that runtime route and does not alter the
persisted training selection.

The unified Torch door rejects `--inference_groups`, `--n_groups`, `--n_images`,
and configured inference counts: omit them for full-scan reconstruction. These
counts remain TensorFlow controls, not aliases for `--groups-per-center`.
The native probe-mask flags and inference payload factory API are retired.

The ordinary Python flow is:

```python
from ptycho_torch.inference import reconstruct
from ptycho_torch.train import train

model = train("dataset.npz", "any/output/name", {
    "architecture": "cnn", "training_groups": 256, "nphotons": 1e9,
})
result = reconstruct(model, "dataset.npz")
```

Strings are accepted; callers need not construct `Path` objects. With no
`work_dir`, `reconstruct` owns and removes a temporary mmap workspace. With a
`work_dir`, it creates the temporary workspace beneath that directory and
still removes only the workspace it created.

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
[study runners](../../scripts/studies/README.md) for the retained entry points.

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
  workflow (`cnn`); `ptycho_study` arms that delegate to that
  workflow enforce the same restriction.
- Shape mismatches in a TensorFlow or explicitly legacy component may mean its
  `update_legacy_dict(params.cfg, config)` bridge was skipped; direct Torch
  `train`/`reconstruct` does not use that bridge. See
  `docs/debugging/TROUBLESHOOTING.md`.

## 11. Legacy-Bundle Migration

Pre-JSON or metadata-free PyTorch archives (`manifest.dill` + per-model
`params.dill`, or a JSON manifest without a sealed identity) are recovered
offline with the era-detecting migrator:

```bash
python -m ptycho_torch.migrate_bundle SOURCE_DIR OUT_DIR
```

Both arguments are directories holding a `wts.h5.zip`. The migrator detects
the source era, rebuilds
the model from the archived weights, seals a fresh current-era
(`torch-artifact-v5`) identity, and writes the migrated archive to `OUT_DIR`.
Pre-v5 recovery is C1-only; C>1 identities require retraining under the
centered-nearest contract and are rejected rather than relabeled.
Migrating an already-current bundle is a no-op. Errors name the missing
bundle or the offending member; the module import itself is torch/dill-free.

## 12. Testing

Run the CPU gate with eight workers:

```bash
bash ci/run_ci_tests.sh
```

Run CUDA integration tests serially. The public train/reload/reconstruct lifecycle
is covered in `tests/torch/test_integration_workflow_torch.py`; architecture
checkpoint round trips are in `tests/torch/test_lightning_checkpoint.py`.

Publishing exclusions are documented in
[Public Branch Port Exclusions](../../scripts/main_overlay/README.md).
