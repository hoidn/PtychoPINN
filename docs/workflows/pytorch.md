# PyTorch Workflow Guide

This guide explains how to run the PyTorch version of PtychoPINN using the modern
`ptycho_torch/workflows/components.py` orchestration layer, which mirrors the
TensorFlow workflows while providing Lightning-based training execution.

## 1. Overview

- **Architecture parity**: `ptycho_torch/model.py` mirrors the TensorFlow U-Net +
  physics stack described in <doc-ref type="guide">docs/architecture.md</doc-ref>.
- **Configuration**: Uses the same TensorFlow dataclass configs (`TrainingConfig`, `ModelConfig`)
  bridged from PyTorch config singletons via `ptycho_torch.config_bridge`. See the
  normative mapping: <doc-ref type="spec">docs/specs/spec-ptycho-config-bridge.md</doc-ref>.
- **Execution engine**: Training uses PyTorch Lightning (`PtychoPINN_Lightning`) with
  deterministic settings and checkpoint management.
- **Data contract**: Identical `.npz` requirements as documented in
  <doc-ref type="contract">specs/data_contracts.md</doc-ref>.
- **Workflow orchestration**: Provided by `ptycho_torch.workflows.components.run_cdi_example_torch`,
  which integrates data loading, training, and optional stitching in a single call.

## 2. Prerequisites

- **PyTorch >= 2.2 (REQUIRED)**: Installed automatically via `setup.py` when running `pip install -e .`.
  If you need a specific CUDA version, install PyTorch manually first following
  [PyTorch installation instructions](https://pytorch.org/get-started/locally/) before installing this package.
- **Lightning and dependencies**: Automatically installed as package dependencies (`lightning`, `tensordict`).
- **Input NPZ files**: Must conform to the project data contract (see <doc-ref type="contract">specs/data_contracts.md</doc-ref>).

## 3. Configuration Setup

PyTorch workflows use the **same configuration system** as TensorFlow. Create a `TrainingConfig`
dataclass instance with your training parameters:

```python
from pathlib import Path
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params

# 1. Define model architecture
model_config = ModelConfig(
    N=64,                    # Diffraction pattern size
    gridsize=2,              # Group size (e.g., 2x2 = 4 patterns)
    model_type='pinn',       # 'pinn' or 'supervised'
    amp_activation='silu',
    n_filters_scale=1,
)

# 2. Define training parameters
config = TrainingConfig(
    model=model_config,
    train_data_file=Path('datasets/my_train.npz'),
    test_data_file=Path('datasets/my_test.npz'),  # Optional
    n_groups=512,            # Number of grouped samples to use
    batch_size=4,
    nepochs=10,
    nphotons=1e6,
    neighbor_count=4,
    output_dir=Path('outputs/my_experiment'),
    debug=False,             # Set True for progress bars and verbose logging
)

# 3. Bridge to legacy params.cfg (MANDATORY before data loading)
update_legacy_dict(params.cfg, config)
```

**Key configuration fields for PyTorch workflows:**
- `config.debug`: Controls progress bars and logging verbosity (default: `False`)
- `config.output_dir`: Directory for checkpoints and artifacts (required for persistence)
- `config.subsample_seed`: RNG seed for reproducible sampling (default: deterministic behavior enabled)
- `config.sequential_sampling`: Use deterministic sequential grouping instead of random (default: `False`)
- `config.torch_loss_mode`: Selects the active loss head (`'poisson'` for physics-weighted Poisson NLL or `'mae'` for amplitude-only MAE). Exposed via the CLI flag `--torch-loss-mode {poisson,mae}`.

### Config Mappings (subset)

Small subset of the TF ↔ PyTorch configuration mapping. See the full spec for all fields and rules: <doc-ref type="spec">docs/specs/spec-ptycho-config-bridge.md</doc-ref>

| PyTorch (config_params) | TensorFlow (dataclass) | Transform / Notes |
|---|---|---|
| `DataConfig.grid_size: (h,w)` | `ModelConfig.gridsize: int` | Require square; use `h` (error if `h!=w`). |
| `ModelConfig.mode: 'Unsupervised'|'Supervised'` | `ModelConfig.model_type: 'pinn'|'supervised'` | Map: Unsupervised→pinn, Supervised→supervised. |
| `TrainingConfig.epochs: int` | `TrainingConfig.nepochs: int` | Rename field. |
| `DataConfig.K: int` | `TrainingConfig.neighbor_count: int` | Semantic rename (K=neighbors). |
| `DataConfig.N: int` | `ModelConfig.N: {64,128,256}` | Validate against allowed set. |
| `ModelConfig.amp_activation: 'silu'|'SiLU'|...` | `ModelConfig.amp_activation` | Map silu/SiLU→swish; others must be supported by TF enum. |


## 4. Loading Data

Use the same `RawData` loading utilities as TensorFlow:

```python
from ptycho.raw_data import RawData

# Load training data (uses params.cfg populated above)
train_data = RawData.from_file(str(config.train_data_file))

# Optional: load test data for validation during training
test_data = RawData.from_file(str(config.test_data_file)) if config.test_data_file else None
```

Data must conform to <doc-ref type="contract">specs/data_contracts.md</doc-ref>:
- `diffraction`: Amplitude (sqrt of intensity), not raw intensity
- `xcoords`, `ycoords`: Scan position coordinates
- `probeGuess`: Complex probe (H×W)
- `objectGuess`: Complex object (M×M, larger than probe)
- When `load_data(..., subsample_seed=X)` is used, the actual sample indices are stored on the `RawData` object (`raw.sample_indices`) and persisted alongside the run as `tmp/subsample_seed{X}_indices.txt`. PyTorch container creation asserts that these recorded indices match the indices used on the Torch side, preventing silent divergence across backends.

## 5. Running Complete Training Workflow

Execute the end-to-end workflow using `run_cdi_example_torch`:

```python
from ptycho_torch.workflows.components import run_cdi_example_torch

# Run training + optional stitching + save
amplitude, phase, results = run_cdi_example_torch(
    train_data=train_data,
    test_data=test_data,       # Optional validation data
    config=config,
    do_stitching=True,         # Enable inference + image reconstruction (Phase D2.C)
)

# Results contain:
# - amplitude: Reconstructed amplitude array (if do_stitching=True)
# - phase: Reconstructed phase array (if do_stitching=True)
# - results: Dict with training history, containers, and model handles
```

**What happens during execution:**

1. **Data normalization**: Converts `RawData` → `PtychoDataContainerTorch` with grouped patches
2. **Probe initialization**: Sets up initial probe from data (integrated with Lightning module as of Phase D2.B)
3. **Lightning training**:
   - Instantiates `PtychoPINN_Lightning` module with PyTorch config objects
   - Builds train/val dataloaders with deterministic seeding
   - Configures `lightning.pytorch.Trainer` with:
     - `max_epochs` from `config.nepochs`
     - `deterministic=True` for reproducibility
     - `default_root_dir` from `config.output_dir`
     - `enable_progress_bar` controlled by `config.debug`
   - Executes `trainer.fit()` and captures loss history
4. **Model persistence**: Saves checkpoint bundle to `config.output_dir/wts.h5.zip` (if output_dir specified)
5. **Optional stitching** (if `do_stitching=True`): Runs inference via Lightning `predict()`, applies flip/transpose transforms, and reassembles full image using TensorFlow reassembly helper for parity (Phase D2.C complete as of 2025-10-19)

## 6. Checkpoint Management and Reproducibility

**Deterministic Behavior:**
- PyTorch workflows enforce `deterministic=True` in Lightning Trainer
- RNG seeding controlled via `config.subsample_seed` (passed to `lightning.pytorch.seed_everything`)
- Sequential sampling available via `config.sequential_sampling=True`

**Checkpoint Storage:**
- Checkpoints saved to `{config.output_dir}/checkpoints/last.ckpt` during training
- Final model bundle persisted as `{config.output_dir}/wts.h5.zip` (Phase D4.C persistence contract)
- Hyperparameters embedded in checkpoint via `model.save_hyperparameters()` for state-free reload

**Loading Trained Models:**
```python
from ptycho_torch.workflows.components import load_inference_bundle_torch

# Load model + config from checkpoint directory
models_dict, loaded_config = load_inference_bundle_torch(config.output_dir)

# Extract Lightning module for inference
lightning_module = models_dict['lightning_module']
```

## 7. Inference and Reconstruction

For standalone inference without retraining:

```python
from ptycho_torch.workflows.components import _reassemble_cdi_image_torch

# Load test data and trained model
test_data = RawData.from_file('datasets/my_test.npz')
models_dict, infer_config = load_inference_bundle_torch(Path('outputs/trained_model'))

# Run inference + stitching
recon_amp, recon_phase, results = _reassemble_cdi_image_torch(
    test_data=test_data,
    config=infer_config,
    flip_x=False,
    flip_y=False,
    transpose=False,
    M=infer_config.model.N * 8  # Reassembly grid size parameter
)
```

**Implementation Status (Phase D2.C, complete as of 2025-10-19):**
- `_reassemble_cdi_image_torch` now performs Lightning inference and reconstructs full images
- Supports `flip_x`, `flip_y`, and `transpose` coordinate transforms for data alignment
- Uses TensorFlow `tf_helper.reassemble_position` for MVP parity (native PyTorch reassembly planned)
- Includes dtype safeguards (float32 enforcement per specs/data_contracts.md §1)
- Channel-order conversion: detects channel-first layout, permutes to channel-last, reduces to single channel
- Artifacts/evidence: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T092448Z/phase_d2_completion/`

## 8. Experiment Tracking and Logging

**Current Status (Phase D2 complete as of 2025-10-19):**
- **MLflow**: Not yet integrated in workflow orchestration (TensorFlow baseline also lacks it; future enhancement)
- **Logging**: Standard Python logging controlled by `config.debug`:
  - `debug=False`: INFO-level messages, progress bars suppressed
  - `debug=True`: INFO-level messages, progress bars enabled
- **Checkpoints**: Lightning automatic checkpoint management to `{output_dir}/checkpoints/`
  - Hyperparameters now embedded via `save_hyperparameters()` for state-free reload (Phase D1c)
  - Checkpoint loading restores dataclass configs automatically (no kwargs required)

**Legacy API Note:**
The deprecated `ptycho_torch/api/` modules (e.g., `example_train.py`) include MLflow autologging,
but are not part of the modern workflow stack. See <doc-ref type="plan">plans/active/ADR-003-BACKEND-API/implementation.md</doc-ref>
for migration guidance away from the legacy API.

## 9. Differences from TensorFlow Workflows

| Aspect | TensorFlow (`ptycho/workflows/components.py`) | PyTorch (`ptycho_torch/workflows/components.py`) |
|--------|----------------------------------------------|--------------------------------------------------|
| Training engine | `ptycho.train_pinn.train()` | `lightning.pytorch.Trainer.fit()` |
| Configuration | Direct `TrainingConfig` dataclass | Bridged via `ptycho_torch.config_bridge` (see spec) |
| Determinism | Manual seed management | `deterministic=True` + `seed_everything()` |
| Progress output | Enabled by default | Controlled by `config.debug` |
| Checkpoints | Keras `.h5` format | Lightning `.ckpt` + bundled `.h5.zip` |
| MLflow | Not implemented (TODO comment) | Not implemented (future enhancement) |

## 10. Common Workflows

### Training-Only (No Inference)

```python
# Fastest workflow for model development
amp, phase, results = run_cdi_example_torch(
    train_data, test_data, config, do_stitching=False
)
# Returns: (None, None, {'history': {...}, 'models': {...}})
```

### Training + Validation Metrics

```python
# Include test_data for validation loss tracking
config.test_data_file = Path('datasets/validation.npz')
test_data = RawData.from_file(str(config.test_data_file))

results = run_cdi_example_torch(train_data, test_data, config, do_stitching=False)
print(results['history'])  # {'train_loss': [...], 'val_loss': [...]}
```

### Full Training + Reconstruction Pipeline

```python
# Complete workflow with image stitching (Phase D2.C required)
amp, phase, results = run_cdi_example_torch(
    train_data, test_data, config, do_stitching=True
)
# Visualize reconstruction
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(121); plt.imshow(amp); plt.title('Amplitude')
plt.subplot(122); plt.imshow(phase); plt.title('Phase')
plt.show()
```

## 11. Regression Test & Runtime Expectations

The PyTorch integration workflow is validated by a comprehensive pytest regression test that exercises the complete train→save→load→infer cycle.

### Test Selector

```bash
CUDA_VISIBLE_DEVICES="0" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

**Environment Requirement:** Pin the regression to a specific CUDA device (e.g., `CUDA_VISIBLE_DEVICES="0"`) so every automated run exercises the GPU backend. The pytest fixture `cuda_gpu_env` enforces this contract by masking other devices.

### Runtime Performance

**Current Performance (Phase B3 Minimal Fixture):**
- **Smoke Test Runtime:** 3.82s on legacy CPU runs (fixture validation suite, 7 tests). GPU timing TBD after first CUDA run; update runtime_profile.md accordingly.
- **Integration Test Runtime:** 14.53s on CPU legacy evidence. Future GPU baselines must be recorded once CUDA runs are executed.
- **Test Dataset:** `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` (64 scan positions, 25 KB)
- **CI Budget:** ≤90s on a single CUDA device (initial value mirrored from CPU budget; adjust after capturing GPU telemetry)
- **Warning Threshold:** 45s (temporary value; recompute after GPU runtime capture)

**Historical Baselines:**
- Phase D1 Runtime: 35.9s ± 0.5s (canonical dataset, 1087 positions)
- Phase B1 Runtime: 21.91s (canonical dataset, n_groups=64 override)
- Phase B3 Improvement: 33.7% faster vs Phase B1 baseline

**Environment:** Legacy evidence captured on Python 3.11.13, PyTorch 2.8.0+cu128, Lightning 2.5.5, Ryzen 9 5950X (32 CPUs), 128GB RAM. **New requirement:** future regression evidence must cite the CUDA GPU model/driver (e.g., NVIDIA A100 40GB, CUDA 12.4) in addition to host specs.

**Performance Profile:** See `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md` for full telemetry.

### Determinism Guarantees

- Lightning `deterministic=True` + `seed_everything()` enforce reproducible runs
- Checkpoint persistence with embedded hyperparameters (Phase D1c, INTEGRATE-PYTORCH-001 Attempts #32-34)
- State-free model reload (no manual config kwargs required)

### Test Coverage

The regression validates:
1. **Training Phase:** Lightning orchestration with grouped data, 2 epochs, checkpoint save
2. **Persistence Phase:** Checkpoint bundle stored at `{output_dir}/checkpoints/last.ckpt`
3. **Load Phase:** Model restored from checkpoint without manual config injection
4. **Inference Phase:** Lightning prediction + image reassembly + PNG export
5. **Artifact Validation:** Reconstruction images (amplitude/phase) exist with >1KB size

### Data Contract Compliance

- **POLICY-001:** PyTorch >=2.2 is mandatory (see `docs/findings.md#POLICY-001`)
- **FORMAT-001:** NPZ auto-transpose guard handles legacy (H,W,N) format (see `docs/findings.md#FORMAT-001`)
- **Test Dataset:** Minimal fixture at `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` (64 scan positions, stratified sampling, canonical (N,H,W) format, float32/complex64 dtypes per DATA-001)
- **Fixture Generation:** Reproducible via `python scripts/tools/make_pytorch_integration_fixture.py --source <canonical_dataset> --output <fixture_path> --subset-size 64` (SHA256 checksum: 6c2fbea0dcadd950385a54383e6f5f731282156d19ca4634a5a19ba3d1a5899c)

### CI Integration Notes

- **Recommended Timeout:** 90s (6.2× current runtime, conservative buffer for CI infrastructure variance)
- **Retry Policy:** 1 retry on timeout (accounts for CI jitter)
- **Markers:** Consider `@pytest.mark.integration` + `@pytest.mark.slow` for selective execution
- **GPU Enforcement:** Set `CUDA_VISIBLE_DEVICES="0"` (or an explicit GPU selection) in CI to guarantee CUDA execution. If a CPU fallback run is required, label it clearly in the report and rerun on GPU as soon as resources are available.

**Reference:** See `plans/active/TEST-PYTORCH-001/implementation.md` for phased test development history and `plans/active/TEST-PYTORCH-001/reports/2025-10-19T233500Z/phase_b_fixture/summary.md` for Phase B3 fixture integration details.

## 12. CLI Execution Configuration Flags

The PyTorch backend CLI exposes execution-level configuration knobs that control training and inference behavior through command-line flags. These flags complement the model/data configuration and provide fine-grained control over runtime execution.

### Training Execution Flags

The following execution config flags are available in `ptycho_torch/train.py`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--accelerator` | str | `'cuda'` | Hardware accelerator: `'cuda'` (default single-GPU run), `'auto'` (detect accelerator), `'cpu'` (fallback), `'tpu'`, `'mps'`. The execution-config dataclass now defaults to `'cuda'`; override explicitly when a GPU is unavailable. |
| `--deterministic` / `--no-deterministic` | bool | `True` | Enable deterministic training (reproducibility) |
| `--num-workers` | int | `0` | Number of DataLoader worker processes (0 = main thread) |
| `--learning-rate` | float | `1e-3` | Optimizer learning rate |
| `--scheduler` | str | `'Default'` | Learning rate scheduler type: `'Default'` (no scheduler), `'Exponential'` (exponential decay), `'MultiStage'` (step-wise decay), `'Adaptive'` (plateau-based reduction) |
| `--accumulate-grad-batches` | int | `1` | Gradient accumulation steps. Simulates larger effective batch sizes (effective batch = batch_size × accumulate_grad_batches). Values >1 reduce GPU memory usage but may affect training dynamics. |
| `--quiet` | flag | `False` | Suppress progress bars and reduce console logging |
| `--enable-checkpointing` / `--disable-checkpointing` | bool | `True` | Enable automatic model checkpointing (default: enabled). Use `--disable-checkpointing` to turn off checkpoint saving. |
| `--checkpoint-save-top-k` | int | `1` | Number of best checkpoints to keep (1 = save only best, -1 = save all, 0 = disable) |
| `--checkpoint-monitor` | str | `'val_loss'` | Metric to monitor for checkpoint selection (default: `'val_loss'`). The literal `'val_loss'` is dynamically aliased to `model.val_loss_name` (e.g., `'poisson_val_loss'` for PINN models) during Lightning configuration. Falls back to `model.train_loss_name` when validation data is unavailable. |
| `--checkpoint-mode` | str | `'min'` | Checkpoint metric optimization mode ('min' for loss metrics, 'max' for accuracy metrics) |
| `--early-stop-patience` | int | `100` | Early stopping patience in epochs. Training stops if monitored metric doesn't improve for this many consecutive epochs. |
| `--logger` | str | `'csv'` | Experiment tracking backend. Options: `'csv'` (CSVLogger, zero dependencies, CI-friendly, stores metrics in `{output_dir}/lightning_logs/version_N/metrics.csv`), `'tensorboard'` (TensorBoardLogger, enables rich visualization via `tensorboard --logdir {output_dir}/lightning_logs/`), `'mlflow'` (MLFlowLogger, requires mlflow package and server URI configuration), `'none'` (disable logging, discards all metrics from `self.log()` calls). Use `'none'` with `--quiet` to suppress all output. |

**Monitor Metric Aliasing:**
The checkpoint monitor metric (`--checkpoint-monitor`) uses dynamic aliasing to handle backend-specific metric naming conventions. When you specify `--checkpoint-monitor val_loss` (the default), the training workflow automatically resolves this to the model's actual validation loss metric name (e.g., `poisson_val_loss` for PINN models). This aliasing ensures compatibility across different loss formulations without requiring users to know internal metric names. When validation data is unavailable, the system automatically falls back to the corresponding training metric (`model.train_loss_name`).

**Gradient Accumulation Considerations:**
Gradient accumulation (`--accumulate-grad-batches`) simulates larger effective batch sizes by accumulating gradients over multiple forward/backward passes before updating model weights. The effective batch size equals `batch_size × accumulate_grad_batches`. While this technique improves memory efficiency (allowing larger effective batches on memory-constrained hardware), values >1 may affect training dynamics, convergence rates, and Poisson loss stability. For PINN models with physics-informed losses, start with the default (`1`) and increase conservatively only when memory constraints require it. Monitor training curves when changing accumulation settings, as the optimizer sees fewer but larger gradient updates per epoch.

**IMPORTANT - Manual Optimization Limitation (EXEC-ACCUM-001):**
The PyTorch backend uses manual optimization (`PtychoPINN_Lightning.automatic_optimization=False`) for custom physics loss integration. Lightning's manual optimization mode is **incompatible** with gradient accumulation (`--accumulate-grad-batches > 1`). Attempting to use accumulation with manual optimization will raise a clear `RuntimeError` before training starts. If you need gradient accumulation for memory management, you must stay with the default (`1`) or restructure to automatic optimization. See `docs/findings.md#EXEC-ACCUM-001` for technical details.

**Supervised Mode Data Requirements (DATA-SUP-001):**
Supervised training (`--model_type supervised`) requires labeled datasets with ground-truth amplitude and phase reconstructions stored as `label_amp` and `label_phase` keys in the NPZ file. Experimental datasets (e.g., `fly001`) and most synthetic datasets lack these labels and will raise a `RuntimeError` during dataloader validation. To use supervised mode, either: (1) generate labeled synthetic data using `ptycho_torch/notebooks/create_supervised_datasets.ipynb`, or (2) switch to unsupervised PINN mode (`--model_type pinn`) for physics-informed self-supervised training. See `docs/findings.md#DATA-SUP-001` for details.

**Logger Backend Details:**
The default CSV logger captures all metrics logged via `self.log()` calls in the Lightning module (train/validation losses, learning rates) without requiring additional dependencies. Metrics are saved as CSV files under `{output_dir}/lightning_logs/version_N/metrics.csv` for easy parsing and analysis. The TensorBoard backend enables interactive visualization (line plots, histograms) but requires the `tensorboard` package (auto-installed via TensorFlow dependency). The MLflow backend integrates with MLflow tracking servers for experiment management but requires manual server setup and the `mlflow` package. Use `--logger none` to completely disable metric capture (e.g., for quick smoke tests or when metrics are not needed).

**DeprecationWarning for --disable_mlflow:**
The legacy `--disable_mlflow` flag now emits a DeprecationWarning directing users to the modern alternatives:
- To disable experiment tracking: use `--logger none`
- To suppress progress bars: use `--quiet`
This flag currently maps to `--logger none` internally for backward compatibility but will be removed in a future release. Update your scripts to use the explicit `--logger` flag instead.

**Deprecated Flags:**
- `--device`: Superseded by `--accelerator`. Using `--device` will emit a deprecation warning and map to `--accelerator` automatically. Remove from scripts; this flag will be dropped in a future release.
- `--disable_mlflow`: **DEPRECATED.** Use `--logger none` (disable tracking) + `--quiet` (suppress progress) instead. Emits DeprecationWarning; will be removed in future release.

**Example CLI command with execution flags:**
```bash
CUDA_VISIBLE_DEVICES="0" python -m ptycho_torch.train \
  --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
  --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
  --output_dir /tmp/cli_smoke \
  --n_images 64 \
  --gridsize 2 \
  --batch_size 4 \
  --max_epochs 1 \
  --accelerator cuda \
  --deterministic \
  --num-workers 0 \
  --learning-rate 1e-3 \
  --quiet
```

**Dose/Overlap Study Fast Path (2025-11-12):** For initiative `STUDY-SYNTH-FLY64-DOSE-OVERLAP-001`, the training CLI must now produce real bundles before any further manifest/test tweaks. Reuse the shared artifact hub `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/` and invoke either `plans/active/.../bin/run_phase_e_job.py --dose 1000 --view dense --gridsize 2` or the explicit CLI command above (TensorFlow backend optional) to capture:
- `cli/` stdout with `bundle_path` + `bundle_sha256`
- `data/` copy of the emitted `wts.h5.zip`
- Updated manifest showing the real artifact paths
- Loss guardrail: After copying the manifest into the hub, run `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_training_loss.py --reference plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/reference/training_manifest.json --candidate <current_manifest> --dose <value> --view <value> --gridsize <value>` and archive the log beside the manifest. Update the reference manifest whenever a newer, visually verified run becomes the baseline.
Only after those outputs land (bundles + passing loss check) may you iterate on additional CLI/test polish or advance the comparison harness. This guardrail prevents repeated plan-only loops and keeps the SSIM/ms-SSIM study focused on shipping metrics.

**Helper-Based Configuration Flow (Phase D.B3, 2025-10-20):**
The training CLI delegates to shared helper functions in `ptycho_torch/cli/shared.py`:
- `resolve_accelerator()`: Handles `--device` → `--accelerator` backward compatibility with deprecation warnings
- `build_execution_config_from_args()`: Constructs `PyTorchExecutionConfig` with validation
- `validate_paths()`: Checks file existence and creates output directories

These helpers enforce CONFIG-001 compliance by calling factory functions that populate `params.cfg` via `update_legacy_dict()` before data loading or model construction. See `ptycho_torch/config_factory.py` for factory implementation details.

**PyTorch Execution Configuration:** For the complete catalog of execution configuration fields (17 total, including programmatic-only parameters like scheduler and logger backend), see <doc-ref type="spec">specs/ptychodus_api_spec.md</doc-ref> §4.9 "PyTorch Execution Configuration Contract". The spec documents validation rules, priority levels, and CONFIG-001 isolation guarantees.

**Evidence:** Phase C4.D validation confirmed gridsize=2 training with execution config flags completes successfully. See `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/manual_cli_smoke_gs2.log` for full smoke test output.

### Inference Execution Flags

The following execution config flags are available in `ptycho_torch/inference.py`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--accelerator` | str | `'cuda'` | Hardware accelerator type (`'cuda'` default for single-GPU inference, `'auto'`, `'cpu'`, `'tpu'`, `'mps'`). Override explicitly when running on CPU-only infrastructure. |
| `--num-workers` | int | `0` | Number of DataLoader worker processes |
| `--inference-batch-size` | int | `None` | Batch size for inference (default: None = reuse training batch_size from checkpoint) |
| `--quiet` | flag | `False` | Suppress progress bars and reduce console logging |

**Deprecated Flags:**
- `--device`: Superseded by `--accelerator`. Using `--device` will emit a deprecation warning and map to `--accelerator` automatically. This flag will be removed in Phase E (post-ADR acceptance).

**Helper-Based Configuration Flow (Phase D.C, 2025-10-20):**
The inference CLI delegates to the same shared helper functions as training (`ptycho_torch/cli/shared.py`):
- `resolve_accelerator()`: Auto-detects hardware or applies user choice, handles `--device` backward compatibility with deprecation warnings
- `build_execution_config_from_args()`: Constructs `PyTorchExecutionConfig` with inference-mode validation
- `validate_paths()`: Checks file existence and creates output directories

Inference orchestration is extracted to `_run_inference_and_reconstruct()` helper (see `ptycho_torch/inference.py:520-640`) which loads the checkpoint bundle, prepares data, runs Lightning prediction, and saves amplitude/phase reconstructions as PNG artifacts. This delegation ensures CONFIG-001 compliance (factory functions populate `params.cfg` via `update_legacy_dict()` before data loading) and maintains parity with training CLI architecture.

**Example CLI Command:**
```bash
# Run inference with minimal dataset fixture on a single CUDA device
CUDA_VISIBLE_DEVICES="0" python -m ptycho_torch.inference \
  --model_path outputs/trained_model \
  --test_data tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
  --output_dir outputs/inference_results \
  --n_images 64 \
  --accelerator cuda \
  --quiet
```

**Expected Output Artifacts:**
- `<output_dir>/reconstructed_amplitude.png`: Reconstructed amplitude image
- `<output_dir>/reconstructed_phase.png`: Reconstructed phase image

**Evidence:** Phase D.C C3 implementation validated thin wrapper behavior with 9/9 passing tests. See `tests/torch/test_cli_inference_torch.py` for delegation contract tests.

### CONFIG-001 Compliance

**CRITICAL:** PyTorch workflows require the same CONFIG-001 initialization as TensorFlow. When using CLI scripts, this happens automatically during factory instantiation via the shared helper functions in `ptycho_torch/cli/shared.py`. The helper-based flow ensures `update_legacy_dict(params.cfg, config)` is called before data loading or model construction.

When using **programmatic entry points** (not CLI), you **MUST** manually call `update_legacy_dict(params.cfg, config)` before any data loading or model construction to ensure legacy modules observe synchronized parameters.

**Reference:** For complete configuration bridge details, see `specs/ptychodus_api_spec.md` §4.8 and `ptycho_torch/config_factory.py` implementation. For CLI helper implementation, see `ptycho_torch/cli/shared.py`.

## 13. Backend Selection in Ptychodus Integration

When PtychoPINN is integrated into Ptychodus, the backend (TensorFlow or PyTorch) can be selected via configuration. This section explains how backend selection works and what guarantees are provided.

### Configuration API

Backend selection is controlled through the `backend` field in configuration dataclasses:

```python
from ptycho.config.config import TrainingConfig, InferenceConfig

# Select PyTorch backend for training
config = TrainingConfig(
    model=model_config,
    train_data_file=Path('data.npz'),
    backend='pytorch',  # or 'tensorflow' (default)
    # ... other parameters
)

# Select PyTorch backend for inference
infer_config = InferenceConfig(
    model=model_config,
    model_path=Path('trained_model/'),
    test_data_file=Path('test.npz'),
    backend='pytorch',  # or 'tensorflow' (default)
    # ... other parameters
)
```

**Default Behavior:** Both `TrainingConfig.backend` and `InferenceConfig.backend` default to `'tensorflow'` to maintain backward compatibility with existing Ptychodus integrations.

### Dispatcher Routing

Per the specification in `specs/ptychodus_api_spec.md` §4.8, the dispatcher guarantees:

1. **TensorFlow Path** (`backend='tensorflow'`): Delegates to `ptycho.workflows.components` entry points without attempting PyTorch imports
2. **PyTorch Path** (`backend='pytorch'`): Delegates to `ptycho_torch.workflows.components` entry points and returns the same `(amplitude, phase, results_dict)` structure
3. **CONFIG-001 Enforcement**: The dispatcher calls `update_legacy_dict(ptycho.params.cfg, config)` before backend inspection to ensure legacy subsystems observe synchronized parameters
4. **Result Metadata**: The returned `results_dict` includes `results['backend']` for downstream logging

### Error Handling

**PyTorch Unavailability:** If `backend='pytorch'` is selected but PyTorch cannot be imported, the system raises an actionable `RuntimeError`:

```
RuntimeError: PyTorch backend selected but torch module unavailable.
Install PyTorch: pip install torch>=2.2
See docs/workflows/pytorch.md for installation guidance.
```

Silent fallbacks to TensorFlow are prohibited per `docs/findings.md#POLICY-001`. This fail-fast behavior ensures users are immediately aware of missing dependencies.

**Invalid Backend:** If `config.backend` contains an unsupported value (not `'tensorflow'` or `'pytorch'`), the dispatcher raises `ValueError` with guidance.

### Checkpoint Compatibility

- **Backend-Specific Formats**: TensorFlow checkpoints use `.h5.zip` format, PyTorch checkpoints use Lightning `.ckpt` format
- **Cross-Backend Loading**: Loading a TensorFlow checkpoint with `backend='pytorch'` (or vice versa) raises a descriptive error
- **Persistence Contract**: See `specs/ptychodus_api_spec.md` §4.8 for full persistence guarantees

### Test Selectors

Backend selection behavior is validated by:

```bash
# Backend routing and error handling tests
pytest tests/torch/test_backend_selection.py::test_backend_field_defaults -vv
pytest tests/torch/test_backend_selection.py::test_pytorch_backend_routes_correctly -vv
pytest tests/torch/test_backend_selection.py::test_tensorflow_backend_routes_correctly -vv
pytest tests/torch/test_backend_selection.py::test_invalid_backend_raises_value_error -vv

# Cross-backend checkpoint loading tests
pytest tests/torch/test_model_manager.py::test_load_tensorflow_checkpoint_with_pytorch_backend -vv
```

**Full Selector:** `pytest tests/torch/test_backend_selection.py -vv` (lines 59-170 in test file)

### Integration Example (Ptychodus)

When `PtychoPINNTrainableReconstructor` is invoked from Ptychodus, the backend is selected based on user settings:

```python
# In ptychodus.model.ptychopinn.reconstructor.py

from ptycho.config.config import InferenceConfig, update_legacy_dict
import ptycho.params

# User selects backend via Ptychodus UI
selected_backend = 'pytorch'  # or 'tensorflow'

# Create configuration with backend selection
config = InferenceConfig(
    model=model_config,
    model_path=checkpoint_path,
    test_data_file=data_path,
    backend=selected_backend,
    # ... other parameters
)

# Bridge to legacy system (REQUIRED before backend-specific code)
update_legacy_dict(ptycho.params.cfg, config)

# Dispatcher routes to appropriate backend
# (handled internally by ptycho.workflows.backend_selector or equivalent)
```

**Reference Implementation:** See `ptycho/workflows/backend_selector.py:121-165` for dispatcher logic (if available in your codebase).

## 14. Troubleshooting

### PyTorch Import Errors

**Symptom:** `RuntimeError: PyTorch backend requires torch>=2.2 and lightning.`

**Solution:** Install PyTorch extras:
```bash
pip install -e .[torch]
```
See <doc-ref type="findings">docs/findings.md#policy-001</doc-ref> for PyTorch requirement policy.

### Shape Mismatch Errors

**Symptom:** Tensor dimension errors during training

**Solution:** Ensure `update_legacy_dict(params.cfg, config)` was called before data loading.
See <doc-ref type="troubleshooting">docs/debugging/TROUBLESHOOTING.md#shape-mismatch-errors</doc-ref>.

### Checkpoint Loading Failures

**Symptom:** `TypeError: missing 4 required positional arguments` when loading checkpoint

**Cause:** Phase D2.B2 implementation embeds hyperparameters in checkpoint; older checkpoints may lack them.

**Solution:** Retrain with current codebase or use legacy load path (under development in Phase D4).

## 15. Keeping Parity with TensorFlow

When introducing new features to PyTorch workflows:

1. **Update both guides**: Modify this document AND `docs/WORKFLOW_GUIDE.md` to note behavioral differences
2. **Module docstrings**: Reference conceptual docs using `<doc-ref type="guide">...</doc-ref>` tags
3. **Configuration parity**: Ensure `ptycho_torch.config_bridge` adapters maintain field compatibility
4. **Test coverage**: Add parity tests in `tests/torch/test_config_bridge.py` or `tests/torch/test_workflows_components.py`

Following these steps ensures developers can move between TensorFlow and PyTorch
implementations without losing architectural context or workflow clarity.

---

**Related Documentation:**
- <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref> — Core architectural principles
- <doc-ref type="spec">specs/ptychodus_api_spec.md</doc-ref> — API contracts and reconstructor lifecycle
- <doc-ref type="contract">specs/data_contracts.md</doc-ref> — NPZ data format requirements
- <doc-ref type="plan">plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md</doc-ref> — Current implementation status
- **Loss/metric parity**: Training logs two amplitude metrics:
  - `amp_inv_mae_epoch`: amplitude MAE in the measurement domain (legacy visibility metric)
  - `amp_mae_tf_scale_epoch`: new metric computed in the same normalized domain used by TensorFlow (`pred_scaled` vs `target_scaled`). This ensures the Poisson-vs-MAE loss curves can be compared directly to TF’s amplitude MAE.
- **Physics weighting**: `torch_loss_mode='poisson'` keeps physics weighting pinned at 1.0 for all epochs (single-stage Poisson training). `torch_loss_mode='mae'` rotates the model into MAE-only training with `physics_weight=0`.

### Patch Parity Evidence

Implementation work that touches training/inference pipelines should include a visual parity check. Use the helper script added in this change:

```bash
python scripts/tools/patch_parity_helper.py \
    --tf-npz tmp/tf_epoch50_patches.npz \
    --torch-npz tmp/torch_epoch50_patches.npz \
    --epoch 50 \
    --num-patches 6
```

The script aligns shared sample ids (using the persisted `sample_indices`) and writes grids to `tmp/patch_parity/{tensorflow, pytorch}_epoch50.png`. These grids provide quick qualitative evidence when numeric losses differ.
