# PyTorch Workflow Guide

This guide explains how to run the PyTorch version of PtychoPINN using the modern
`ptycho_torch/workflows/components.py` orchestration layer, which mirrors the
TensorFlow workflows while providing Lightning-based training execution.

## 1. Overview

- **Architecture parity**: `ptycho_torch/model.py` mirrors the TensorFlow U-Net +
  physics stack described in <doc-ref type="guide">docs/architecture.md</doc-ref>.
- **Configuration**: Uses the same TensorFlow dataclass configs (`TrainingConfig`, `ModelConfig`)
  that are automatically translated to PyTorch equivalents via `ptycho_torch.config_bridge`.
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
| Configuration | Direct `TrainingConfig` dataclass | Auto-translated via `ptycho_torch.config_bridge` |
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
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
```

**Environment Requirement:** `CUDA_VISIBLE_DEVICES=""` enforces CPU-only execution per test contract (enforced via `cuda_cpu_env` fixture).

### Runtime Performance

- **Baseline Runtime:** 35.9s ± 0.5s (observed mean: 35.92s, variance: 0.17%)
- **CI Budget:** ≤90s on modern CPU hardware (2.5× baseline; allows for slower CI infrastructure)
- **Warning Threshold:** 60s (1.7× baseline triggers investigation)

**Environment:** Verified on Python 3.11.13, PyTorch 2.8.0+cu128, Lightning 2.5.5, Ryzen 9 5950X (32 CPUs), 128GB RAM.

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
- Dataset: Canonical format per `specs/data_contracts.md` §1 (diffraction=amplitude, float32)

### CI Integration Notes

- **Recommended Timeout:** 120s (conservative 3.3× baseline)
- **Retry Policy:** 1 retry on timeout (accounts for CI jitter)
- **Markers:** Consider `@pytest.mark.integration` + `@pytest.mark.slow` for selective execution

**Reference:** See `plans/active/TEST-PYTORCH-001/implementation.md` for phased test development history.

## 12. Troubleshooting

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

## 13. Keeping Parity with TensorFlow

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
