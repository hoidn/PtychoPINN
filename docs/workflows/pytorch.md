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
  `ptycho_torch/config_bridge.py`.
- **Training** runs through `PtychoPINN_Lightning` (`ptycho_torch/model.py`) with
  deterministic settings, Lightning checkpointing, and the full physics loss for every
  architecture.
- **Data contract** is identical to TensorFlow: the same standalone NPZ format
  consumed by the TensorFlow workflows (see §2).

## 2. Prerequisites

- `pip install .` installs torch ≥ 2.2, `lightning`, and `tensordict` automatically.
  For a specific CUDA build, install PyTorch manually first
  ([instructions](https://pytorch.org/get-started/locally/)), then `pip install .`
- Input NPZ files with `diffraction` stored as amplitude (sqrt of intensity),
  `xcoords`/`ycoords` scan positions, a complex `probeGuess`, and `objectGuess`.

## 3. Configuration

### 3.1. The Two Config Layers

1. **Canonical configs** (`TrainingConfig`, `InferenceConfig`, `ModelConfig`) describe
   the model and data. They bridge to `params.cfg` and to the torch singletons.
   `update_legacy_dict(params.cfg, config)` MUST run before any data
   loading or legacy-module import. The CLIs do this automatically via
   `ptycho_torch/config_factory.py`; programmatic callers must do it themselves.
2. **`PyTorchExecutionConfig`** (`ptycho.config.config`) holds runtime-only knobs
   (accelerator, workers, learning rate, scheduler, checkpointing, logger, structural
   search fields). Execution config must NEVER populate `params.cfg`.
   Full field catalog and validation rules: `specs/ptychodus_api_spec.md` §4.9.

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
- `neuralop_uno` — wraps external `neuraloperator==2.0.0` U-NO (locked to the
  Lines128 CDI path: `N=128`, `gridsize=1`, `C=1`, `real_imag`)

To implement, configure, train, save, and reload a new architecture, follow the
[Custom PyTorch CDI Architecture Guide](custom_torch_architecture.md). The
generator-package README is a lower-level reference for the existing modules.

**Reliability caveat:** the stock `cnn` under the
count-Poisson recipe at `N=128` collapses to a flat-amplitude output with
near-certainty unless the TF-parity preset below is applied (§3.6).

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

### 3.5. CNN Output / Physics-Forward Knobs (Main-Parity Stack)

Four torch-`ModelConfig` knobs port the legacy-main CNN representation and physics as
opt-in modes. All default to the values that keep existing CNN and FNO behavior
unchanged:

| Knob | Default | Opt-in value | Effect |
|---|---|---|---|
| `cnn_output_mode` | `'amp_phase'` | `'real_imag'` (Unsupervised-only) | CNN emits `(real, imag)` via `ScaledTanh` boxes (real ∈ (−0.8, 1.2), imag ∈ (−1.2, 1.2)); prerequisite for `rectangular_scaled`. Representability limit: unit-amplitude objects near `|phase| → π` are unreconstructable in this mode. |
| `use_shared_decoder` | `False` | `True` | Single shared decoder emitting `2*C_out` channels, split per branch; architecture-only knob. |
| `training_patch_weighting` | `'central_mask'` | `'probe'` (or `'uniform'`) | Training-forward reassembly weighting: binary center mask vs `Σ|probe|²`-weighted (`'uniform'` isolates the code-path change without probe weighting). Distinct from the inference-only `InferenceConfig.patch_weighting`. |
| `physics_forward_mode` | `'amplitude'` | `'rectangular_scaled'` | Routes patches through `RectangularScaledDiffraction` (analytic real/imag intensity model with per-dataset trainable `s1`/`s2` unless `rect_s1s2_trainable=False`). Requires `cnn_output_mode='real_imag'`; the matching intensity-domain losses (`RectangularPoissonLoss` / `RectangularMAELoss`) are selected automatically. |

Two further knobs are **inference-only** (`InferenceConfig.patch_weighting`,
`InferenceConfig.varpro_scaling`): they affect only
`ptycho_torch.reassembly.reconstruct_image_barycentric` (the in-process reconstruction
path) and never touch training numerics. The `python -m ptycho_torch.inference` CLI
does not consume them — it always uses uniform
`helper.reassemble_patches_position_real`. Call `reconstruct_image_barycentric`
directly when these knobs must take effect.

### 3.6. TF-Parity Preset for the Torch CNN (N=128 reliability)

Three knobs on the standard `cnn` close the collapse gap against the TensorFlow
reference (no separate registry entry):

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

Cautions:
- Do NOT set `intensity_scale_trainable=True` alongside the parity kwargs — the dead
  `IntensityScalerModule` machinery silently overwrites the input-side parity scale.

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
| Execution | `--accelerator {auto,cuda,cpu,tpu,mps}`, `--deterministic/--no-deterministic`, `--num-workers`, `--learning-rate`, `--scheduler {Default,Exponential,MultiStage,Adaptive}`, `--accumulate-grad-batches`, `--quiet` |
| Checkpointing | `--enable-checkpointing/--disable-checkpointing`, `--checkpoint-save-top-k`, `--checkpoint-monitor` (default `val_loss`, auto-aliased to the model's actual metric, e.g. `poisson_val_loss`), `--checkpoint-mode`, `--early-stop-patience` |
| Loss/probe | `--torch-loss-mode {poisson,mae}`, `--probe-mask/--no-probe-mask`, `--probe-mask-sigma`, `--probe-mask-diameter` |
| Logging | `--logger {csv,tensorboard,mlflow,none}`, `--log-patch-stats`, `--patch-stats-limit` |
| Deprecated | `--device` (→ `--accelerator`), `--disable_mlflow` (→ `--logger none` + `--quiet`) |

The CLI builds `PyTorchExecutionConfig` through `ptycho_torch/cli/shared.py`
(`resolve_accelerator`, `build_execution_config_from_args`, `validate_paths`), which
also performs the mandatory `params.cfg` bridging via the config factory.

### 4.3. Programmatic

```python
from ptycho.raw_data import RawData
from ptycho_torch.workflows.components import run_cdi_example_torch

train_data = RawData.from_file(str(config.train_data_file))
test_data = RawData.from_file(str(config.test_data_file)) if config.test_data_file else None

amplitude, phase, results = run_cdi_example_torch(
    train_data, test_data, config,
    do_stitching=True,            # False → (None, None, results) training-only
    execution_config=exec_cfg,    # optional PyTorchExecutionConfig
)
```

`run_cdi_example_torch` normalizes the data into `PtychoDataContainerTorch`, seeds and
instantiates `PtychoPINN_Lightning`, runs `Trainer.fit()` (deterministic, checkpoints
under `{output_dir}/checkpoints/`), persists the bundle, and — when
`do_stitching=True` — runs Lightning prediction and reassembles the image
(`flip_x`/`flip_y`/`transpose` args control coordinate transforms, `M` the stitch
window).

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
- **Gridsize > 1 support** is architecture-gated in the study runners (only `cnn`
  is ported); other architectures currently reject `gridsize > 1` there.
- **N=128 CNN collapse** without the parity preset (§3.6).
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

Commands, selectors, and evidence requirements: `docs/TESTING_GUIDE.md`.

---

**Related Documentation:**
- <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref> — architectural principles and anti-patterns
- <doc-ref type="spec">specs/ptychodus_api_spec.md</doc-ref> — backend dispatch and execution-config contracts (§4.8–4.9)
- <doc-ref type="guide">docs/TESTING_GUIDE.md</doc-ref> — test commands and evidence requirements
