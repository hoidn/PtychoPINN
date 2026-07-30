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

- **Configuration** uses the same canonical dataclasses as TensorFlow
  (`ptycho.config.config.TrainingConfig` / `InferenceConfig`), bridged to the
  torch-side config singletons (`ptycho_torch/config_params.py`) via
  `ptycho_torch/config_bridge.py`. Normative field mapping:
  <doc-ref type="spec">docs/specs/spec-ptycho-config-bridge.md</doc-ref>.
- **Training** runs through `PtychoPINN_Lightning` (`ptycho_torch/model.py`) with
  deterministic settings, Lightning checkpointing, and the full physics loss for every
  architecture.
- **Data contract** is identical to TensorFlow: standalone NPZ per
  <doc-ref type="contract">docs/specs/spec-ptycho-core.md</doc-ref>.
- **Recommended parameter baselines** live in
  [docs/model_baselines.md](../model_baselines.md); this guide documents the available
  knobs, not the current best-practice combination.

## 2. Prerequisites

- `pip install .` installs torch ≥ 2.2, `lightning`, and `tensordict` automatically.
  For a specific CUDA build, install PyTorch manually first
  ([instructions](https://pytorch.org/get-started/locally/)), then `pip install .`
- Input NPZ files conforming to `docs/specs/spec-ptycho-core.md`
  (`diffraction` as amplitude, `xcoords`/`ycoords`, complex `probeGuess`,
  and optional `objectGuess`).

## 3. Configuration

### 3.1. Scientific Configuration and Runtime Resolution

1. **Canonical and Torch scientific configs** describe data, model topology,
   optimization, and inference. Learning rate, scheduler, gradient clipping,
   and accumulation resolve into Torch `TrainingConfig`; topology resolves into
   Torch `ModelConfig`. The pure `resolve_training_payload` and
   `resolve_inference_payload` functions leave global configuration and
   filesystem state untouched. The compatibility `create_training_payload` and
   `create_inference_payload` wrappers add one bounded CONFIG-001 projection
   through `_project_legacy_config` and `populate_legacy_params` before
   returning their explicit payloads. Surviving legacy leaves are scoped
   separately.
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
    n_groups=512,
    batch_size=4,
    nepochs=10,
    output_dir=Path('outputs/my_experiment'),
)
update_legacy_dict(params.cfg, config)   # MANDATORY before data loading
```

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
  `Adaptive`); the plateau/warmup schedulers are exposed by the study runners and the
  unified `--torch-scheduler` flag.
- `TrainingConfig.subsample_seed` seeds `lightning.pytorch.seed_everything`;
  `sequential_sampling=True` gives deterministic first-N grouping. Subsampled indices
  are persisted (`raw.sample_indices`, `tmp/subsample_seed{X}_indices.txt`) and
  asserted equal across backends.

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

New Torch checkpoints and bundles use `torch-model-spec-v2` inside
`torch-artifact-v2`. Existing v1 artifacts remain readable through frozen
v1 field inventories and deterministic upgrade. The bundle version remains
`2.0-pytorch` with exactly `autoencoder` and `diffraction_to_obj`; TensorFlow
bundle version `1.0` is unchanged.

### 3.5. CNN Output / Physics-Forward Knobs (Main-Parity Stack)

Four torch-`ModelConfig` knobs port the legacy-main CNN representation and physics as
opt-in modes. All default to the values that keep existing CNN/FNO/hybrid behavior
unchanged:

| Knob | Default | Opt-in value | Effect |
|---|---|---|---|
| `cnn_output_mode` | `'amp_phase'` | `'real_imag'` (Unsupervised-only) | CNN emits `(real, imag)` via `ScaledTanh` boxes (real ∈ (−0.8, 1.2), imag ∈ (−1.2, 1.2)); prerequisite for `rectangular_scaled`. Representability limit: unit-amplitude objects near `|phase| → π` are unreconstructable in this mode. |
| `use_shared_decoder` | `False` | `True` | Single shared decoder emitting `2*C_out` channels, split per branch; architecture-only knob. |
| `training_patch_weighting` | `'central_mask'` | `'probe'` (or `'uniform'`) | Public training-forward assembly policy for grouped patches: binary center mask vs `Σ|probe|²`-weighted (`'uniform'` isolates the code-path change without probe weighting). Distinct from the inference-only `InferenceConfig.patch_weighting`. |
| `physics_forward_mode` | `'amplitude'` | `'rectangular_scaled'` | Routes patches through `RectangularScaledDiffraction` (analytic real/imag intensity model with per-dataset trainable `s1`/`s2` unless `rect_s1s2_trainable=False`). Requires `cnn_output_mode='real_imag'`; the matching intensity-domain losses (`RectangularPoissonLoss` / `RectangularMAELoss`) are selected automatically. |

Physical semantics of `s1`/`s2` and known residual differences: see the
rectangular-scaled diffraction entry in `docs/findings.md`.

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
path) and never touch training numerics. The `python -m ptycho_torch.inference` CLI
does not consume them — it always uses uniform
`helper.reassemble_patches_position_real`. Call `reconstruct_image_barycentric`
directly when these knobs must take effect.

### 3.6. CNN Parity Diagnostic Knobs (Not A Baseline)

These knobs remain available for controlled parity diagnostics, but they do not
replace the canonical baseline in `docs/model_baselines.md`:

| Knob | Where | Diagnostic value | Default |
|---|---|---|---|
| `cbam_encoder` | torch `ModelConfig` | `False` | `True` |
| `parity_init_scheme` | `PtychoPINN_Lightning` kwarg (`"default"` \| `"tf_glorot"`) | `"tf_glorot"` | `"default"` (kaiming) |
| `scheduler` | `TrainingConfig` | `"ReduceLROnPlateau"` | `"Default"` |

An additional default-off mechanism, `parity_scale_mode`
(`PtychoPINN_Lightning` kwarg; `"off"` \| `"tied"` \| `"input"` \| `"output"` \|
`"fixed"`), controls the TF-parity global intensity scale; it is forwarded by
`ptycho_torch/train_lightning_only.py` and driven from
`scripts/studies/varpro_probe_ablation_runner.py`
(`--cbam-encoder on|off`, `--parity-init-scheme`, `--scheduler`).

Cautions:
- CBAM-off is not a recommended fix: the support-on Task 30 comparison left
  post-VarPro amplitude and phase quality essentially unchanged, so CBAM was
  closed as the cause of the support failure.
- Do NOT set `intensity_scale_trainable=True` alongside the parity kwargs — the dead
  `IntensityScalerModule` machinery silently overwrites the input-side parity scale
  (see the dead intensity-scaler entry in `docs/findings.md`).
- Root-cause record: the N=128 flat-amplitude collapse entry in `docs/findings.md`;
  evidence: `docs/plans/2026-07-08-cnn-n128-tf-parity.md`.

## 4. Training

### 4.1. Unified CLI (backend selection)

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

### 4.2. Native CLI

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
| Data/model | `--train_data_file`, `--test_data_file`, `--output_dir`, `--n_images` (number of groups), `--gridsize`, `--batch_size`, `--max_epochs`, `--config <yaml>` |
| Execution | `--accelerator {auto,cuda,cpu,tpu,mps}`, `--deterministic/--no-deterministic`, `--num-workers`, `--quiet` |
| Optimization | `--learning-rate`, `--scheduler {Default,Exponential,MultiStage,Adaptive}`, `--accumulate-grad-batches` |
| Checkpointing | `--enable-checkpointing/--disable-checkpointing`, `--checkpoint-save-top-k`, `--checkpoint-monitor` (default `val_loss`, auto-aliased to the model's actual metric, e.g. `poisson_val_loss`), `--checkpoint-mode`, `--early-stop-patience` |
| Loss/probe | `--torch-loss-mode {poisson,mae}`, `--probe-mask/--no-probe-mask`, `--probe-mask-sigma`, `--probe-mask-diameter` |
| Logging | `--logger {csv,tensorboard,mlflow,none}`, `--log-patch-stats`, `--patch-stats-limit` |
| Deprecated | `--device` (→ `--accelerator`), `--disable_mlflow` (→ `--logger none` + `--quiet`) |

The CLI builds an `ExecutionRequest` and a separate Torch training patch through
`ptycho_torch/cli/shared.py`
(`build_execution_request_from_args`, `build_training_config_patch_from_args`,
`validate_paths`), then calls `create_training_payload` once. Unlike the pure
resolver, this compatibility wrapper performs one bounded CONFIG-001
projection through `_project_legacy_config` and `populate_legacy_params`
before returning the explicit `TrainingPayload`. That payload owns the
resolved execution, data, model, and training configuration and is forwarded
through mmap construction to `run_cdi_example_torch` via `resolved_payload`;
the workflow does not perform another full projection at entry. Optional
stitching has a separate narrow scoped projection in
`_reassemble_position_with_legacy_geometry`.

The native training CLI stages only the exact NPZ selected by
`--train_data_file` and, when supplied, the exact NPZ selected by
`--test_data_file`; neighboring NPZ files are not discovered or included. It
builds persistent `PtychoDataset` maps at
`<output_dir>/mmap_workspace/{train,test}/mmap/memmap`, with the corresponding
state and manifest files in each role's `mmap` directory. Each invocation
removes and rebuilds each selected role workspace before training: `train` is
always built and rebuilt, while `test` is built and rebuilt only when
`--test_data_file` is supplied. These paths are run-local working data rather
than a reusable cross-invocation cache.

This native mmap entry point is supported on Linux with procfs mounted and
accessible at `/proc/self/fd`. Before it creates `output_dir` or stages any
data, the command verifies that a live directory descriptor can be resolved
through procfs and that descriptor-relative, no-follow filesystem operations
are available. An unsupported platform or inaccessible procfs fails with an
actionable error; there is no path-based safety fallback. Programmatic
in-memory training is unaffected by this native-CLI requirement.

For this entry point, `--n_images=N` means exactly `N` grouped mmap rows. The
limit is applied before memory-map allocation, and the command fails if the
selected NPZ does not contain at least `N` candidate groups. With
`sequential_sampling=True`, selection takes the first records in stable
file/group order. Otherwise selection is deterministic without replacement,
using `subsample_seed` when configured and `42` when it is absent.
`objectGuess` is optional for both unsupervised and supervised mmap input.
Supervised input still requires `label`; when `objectGuess` is absent, its
phase correction is `0.0`.

### 4.3. Programmatic

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
    do_stitching=True,            # False → (None, None, results) training-only
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

Programmatic `RawData` and `RawDataTorch` inputs continue through this in-memory
`PtychoDataContainerTorch` path; they do not create the native CLI's
`mmap_workspace` tree.

## 5. Checkpoints, Persistence, Reproducibility

- **Determinism:** `deterministic=True` + `seed_everything(config.subsample_seed)`.
- **Checkpoints:** `{output_dir}/checkpoints/last.ckpt` (Lightning), with
  hyperparameters embedded via `save_hyperparameters()` — checkpoints reload without
  manual config kwargs.
- **Bundle:** the final model persists as `{output_dir}/wts.h5.zip`. The
  `intensity_scale` is captured (learned value if trainable, else the spec fallback
  `sqrt(nphotons)/(N/2)`) and stored in the bundle's `params.dill`, so inference uses
  the same normalization as training.
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
selection, count-scale modes, structural-search sweeps, parity presets) are owned by
the study CLIs, not this guide:

- `scripts/studies/grid_lines_torch_runner.py` — grid-lines training/eval for the
  registry architectures (`--architecture`, plateau/warmup schedulers, probe-source
  validation, reassembly strategy knobs).
- `scripts/studies/grid_lines_compare_wrapper.py` — TF-vs-torch multi-model
  comparisons (`--architectures`, `--dataset-source`, probe scaling/masking).
- `scripts/studies/varpro_probe_ablation_runner.py` — parity-preset ablations
  (§3.6 flags).

Consult each runner's `--help` and `scripts/studies/README.md`.

## 10. Constraints and Known Pitfalls

- **Gradient accumulation:** `PtychoPINN_Lightning` uses manual optimization, which is
  incompatible with gradient accumulation — `--accumulate-grad-batches > 1` raises a
  `RuntimeError` before training. Keep the default (`1`).
- **Supervised mode** (`model_type='supervised'`) requires `label_amp` /
  `label_phase` keys in the NPZ; experimental datasets lack them and fail dataloader
  validation. Use PINN mode or generate labeled synthetic data.
- **Gridsize > 1 support** is architecture-gated in the study runners (`cnn`,
  `hybrid_resnet`); other architectures currently reject `gridsize > 1` there.
- **N=128 CNN collapse** without the parity preset (§3.6).
- **`intensity_scale_trainable=True`** conflicts with the parity scale path (§3.6).
- Shape mismatches at load time usually mean the `update_legacy_dict(params.cfg,
  config)` bridge was skipped — see
  `docs/debugging/TROUBLESHOOTING.md`.

## 11. Testing

- Fast suite: `pytest tests/torch -m "not slow"` (this is what the public-main CI
  gate runs).
- End-to-end regression (train→save→load→infer, GPU-pinned):

```bash
CUDA_VISIBLE_DEVICES="0" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

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
