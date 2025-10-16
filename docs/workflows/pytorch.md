# PyTorch Workflow Guide

This guide explains how to run the PyTorch version of PtychoPINN, mirroring the
TensorFlow workflows documented elsewhere in the project while highlighting
PyTorch-specific configuration and tooling choices.

## 1. Overview

- Architecture parity: `ptycho_torch/model.py` mirrors the TensorFlow U-Net +
  physics stack described in <doc-ref type="guide">docs/architecture.md</doc-ref>.
- Configuration: runtime settings are provided via the singleton helpers in
  `ptycho_torch/config_params.py` instead of dataclasses.
- Execution engine: training uses PyTorch Lightning with optional MLflow autologging.
- Data contract: identical `.npz` requirements as documented in
  <doc-ref type="contract">specs/data_contracts.md</doc-ref>.

## 2. Prerequisites

- PyTorch >= 2.2 with CUDA support (if training on GPU).
- `lightning`, `mlflow`, and `tensordict` installed.
- Input NPZ files generated according to the project data contract.
- Optional: running MLflow tracking server (set `MLFLOW_TRACKING_URI` if using a remote instance).

## 3. Configure Runtime Parameters

```python
from ptycho_torch.config_params import (
    ModelConfig, TrainingConfig, DataConfig,
    model_config_default, training_config_default, data_config_default,
)

ModelConfig().set_settings(model_config_default)
TrainingConfig().set_settings(training_config_default)
DataConfig().set_settings(data_config_default)

# Optional overrides
ModelConfig().add("loss_function", "Poisson")
DataConfig().add("N", 64)
TrainingConfig().add("device", "cuda")
```

Keep overrides consistent with the high-level guidelines in
<doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>.

## 4. Prepare Data

1. Organize diffraction NPZ files under a directory (one file per experiment).
2. Place probe NPZ files in a companion directory; set `DataConfig().add("probe_dir_get", True)`
   to load them automatically.
3. The `PtychoDataset` class memory-maps the diffraction stack into
   `data/memmap/`. Use `remake_map=True` to rebuild the map when upstream data changes.
4. Inspect `ptycho_torch/dset_loader_pt_mmap.py` for tensor shapes and ensure they align with the
   expectations in `specs/data_contracts.md`.

## 5. Launch Training

```python
from ptycho_torch import train

ptycho_dir = "/path/to/diffraction_npz"
probe_dir = "/path/to/probe_npz"

train.main(ptycho_dir, probe_dir)
```

The training loop performs:

1. Dataset creation via `PtychoDataset`.
2. Batch streaming through `TensorDictDataLoader`, which yields `(images, coords, probe, scale)`.
3. Model construction using `ptycho_torch.train.PtychoPINN` (LightningModule) and the underlying
   physics layers in `ptycho_torch/model.py`.
4. Lightning `Trainer.fit` with gradient clipping and automatic device selection.
5. MLflow autologging of metrics, hyperparameters, and checkpoints.

Adjust `max_epochs`, `devices`, or `gradient_clip_val` in `train.py` as needed.

## 6. Run Inference or Validation

- To obtain complex object reconstructions, set `model.predict = True` or call
  `LightningModule.forward_predict(...)` to bypass the physics loss.
- Use `trainer.predict` with the same dataloader to generate outputs without gradient updates.
- For patch reassembly or visualization, reuse utilities in `ptycho_torch.helper`
  (see cross references in its docstrings).

## 7. Experiment Tracking

- By default, `mlflow.pytorch.autolog` records checkpoints, metrics, and parameters under the
  experiment name "PtychoPINN vanilla".
- Override the tracking URI via `MLFLOW_TRACKING_URI` or uncomment the lines in `train.py`.
- `print_auto_logged_info` provides a quick summary of the run for interactive sessions.

## 8. Keeping Parity with TensorFlow

When introducing new features:

1. Update both this guide and the corresponding TensorFlow documentation to note behavioural
   differences.
2. Extend the module-level docstrings in `ptycho_torch/` to reference relevant conceptual docs
   using `<doc-ref type="guide">...</doc-ref>` tags.
3. Verify that the singleton configuration schema remains synchronized with
   `ptycho/config.py` and the global data contract.

Following these steps ensures that developers can move between the TensorFlow and PyTorch
implementations without losing architectural context or workflow clarity.
