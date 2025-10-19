# PyTorch-Specific Execution Knobs Catalog

**Status**: Complete enumeration of PyTorch execution parameters NOT represented in TensorFlow canonical dataclasses
**Generated**: 2025-10-19
**Scope**: ptycho_torch/{config_params.py, train.py, train_utils.py, workflows/components.py, config_bridge.py}

---

## Overview

This document catalogs all execution knobs specific to the PyTorch backend that are **not already represented** in the TensorFlow canonical dataclasses (`ptycho/config/config.py`). This serves as a foundation for designing `PyTorchExecutionConfig` dataclass or spec extensions.

---

## Knob Catalog

| Knob Name | Current Definition | File:Line | Default Value | Purpose | TensorFlow Equivalent | Proposed Home | Parity Notes |
|-----------|-------------------|-----------|---------------|---------|----------------------|----------------|--------------|
| **LIGHTNING TRAINER PARAMETERS** |
| `max_epochs` | `training.epochs` → `L.Trainer(max_epochs=...)` | train.py:265 | 100 (CLI arg) | Lightning trainer max epoch count | TrainingConfig.nepochs | No change needed | PyTorch uses `epochs`; TF uses `nepochs` (mapped via config_bridge) |
| `accelerator` | `'gpu' if cuda else 'cpu'` → `L.Trainer(accelerator=...)` | train.py:268 | auto (inferred) | Hardware accelerator ('cpu', 'gpu', 'tpu', 'mps', 'auto') | Not in TrainingConfig | PyTorchExecutionConfig.accelerator | TensorFlow trainers abstract hardware; PyTorch explicit per Lightning |
| `devices` | `training_config.n_devices` → `L.Trainer(devices=...)` | train.py:267 | 1 | Number of compute devices (GPUs/TPUs) to use | TrainingConfig.n_devices | No change needed | Semantic equivalence; PyTorch explicit via `devices` parameter |
| `strategy` | `get_training_strategy(n_devices)` → `L.Trainer(strategy=...)` | train.py:272 | 'auto' (1 device) or DDPStrategy | Distributed training strategy ('auto', 'ddp', 'fsdp', 'deepspeed') | TrainingConfig.strategy (optional) | PyTorchExecutionConfig.strategy | TensorFlow legacy; PyTorch explicit via Lightning strategy object |
| `check_val_every_n_epoch` | hardcoded `1` | train.py:273 | 1 | Validation frequency in epochs | Not in TrainingConfig | PyTorchExecutionConfig.check_val_every_n_epoch | TensorFlow has implicit validation during training; PyTorch explicit |
| `enable_checkpointing` | hardcoded `True` | train.py:274 | True | Enable Lightning checkpoint callbacks | Implicit in TF (checkpoint_dir) | No change needed | TensorFlow checkpointing automatic; PyTorch explicit flag |
| `enable_progress_bar` | hardcoded `True` | train.py:275 | True | Enable Lightning progress bar output | Not in TrainingConfig | PyTorchExecutionConfig.quiet (inverse) | TensorFlow silent by default; PyTorch verbose |
| `default_root_dir` | `checkpoint_root` (inferred from output_dir) | train.py:266 | parent of cwd or args.output_dir | Root directory for Lightning artifacts (checkpoints, logs) | TrainingConfig.output_dir | No change needed | Semantic equivalence |
| `deterministic` | hardcoded `True` | workflows/components.py:572 | True | Enforce deterministic/reproducible training | Not in TrainingConfig | PyTorchExecutionConfig.deterministic | TensorFlow has seed but no deterministic flag; PyTorch explicit |
| `logger` | hardcoded `False` | workflows/components.py:573 | False | Lightning logger instance (disabled in MVP; MLflow in train.py) | Not in TrainingConfig | PyTorchExecutionConfig.logger_backend | TensorFlow abstract; PyTorch explicit logger object or string |
| **DISTRIBUTED TRAINING (DDP-SPECIFIC)** |
| `n_devices` | `training_config.n_devices` | train.py:268, 493 | 1 | Number of devices for DDP | TrainingConfig.n_devices | No change needed | Exposed at CLI level (inferred from device='cuda' and torch.cuda.device_count()) |
| `find_unused_parameters` | `False` in DDPStrategy | train_utils.py:75 | False | DDP optimizer to skip unused params (reduces communication) | Not in TrainingConfig | PyTorchExecutionConfig.ddp_find_unused_parameters | Advanced tuning knob; not in TF baseline |
| `static_graph` | `True` in DDPStrategy | train_utils.py:76 | True | DDP static computation graph optimization | Not in TrainingConfig | PyTorchExecutionConfig.ddp_static_graph | Advanced tuning knob; TensorFlow has no equivalent |
| `gradient_as_bucket_view` | `True` in DDPStrategy | train_utils.py:77 | True | DDP memory optimization (bucket gradients as views) | Not in TrainingConfig | PyTorchExecutionConfig.ddp_gradient_as_bucket_view | Advanced tuning knob; unique to PyTorch DDP |
| `process_group_backend` | `'nccl'` in DDPStrategy | train_utils.py:78 | 'nccl' | DDP backend ('nccl', 'gloo', 'mpi') | Not in TrainingConfig | PyTorchExecutionConfig.ddp_backend | TensorFlow abstracted; PyTorch explicit choice |
| `RANK` (env var) | `int(os.environ.get('RANK', 0))` | train_full.py:233 | 0 | Current process rank in DDP (set by DDP launcher) | Not in TrainingConfig | Environment variable (read-only) | Set by torch.distributed.launch or lightning; not user-configurable |
| `LOCAL_RANK` (env var) | `int(os.environ.get('LOCAL_RANK', 0))` | train_full.py:233 | 0 | Local rank on current node in DDP | Not in TrainingConfig | Environment variable (read-only) | Set by DDP launcher; not user-configurable |
| **MLFLOW INTEGRATION KNOBS** |
| `disable_mlflow` | CLI flag `--disable_mlflow` | train.py:396-397 | False | Disable MLflow autologging and experiment tracking | Not in TrainingConfig | PyTorchExecutionConfig.disable_mlflow | TensorFlow has no MLflow integration; PyTorch uses MLflow extensively |
| `experiment_name` | `training_config.experiment_name` | train.py:289, config_params.py:127 | "Synthetic_Runs" | MLflow experiment name for grouping runs | TrainingConfig.experiment_name | No change needed (already in TrainingConfig) | Already exposed; mapped to TF via config_bridge |
| `notes` | `training_config.notes` (optional tag) | train.py:299-300, config_params.py:128 | "" | MLflow run notes/description tag | TrainingConfig.notes | No change needed (already in TrainingConfig) | Already exposed; optional metadata |
| `model_name` | `training_config.model_name` (tag) | train.py:298, config_params.py:129 | "PtychoPINNv2" | MLflow model_name tag for model registry | TrainingConfig.model_name | No change needed (already in TrainingConfig) | Already exposed; used by MLflow model registry |
| `checkpoint_monitor` | `val_loss_label` → `mlflow.pytorch.autolog()` | train.py:283 | derived from val_loss_name | MLflow checkpoint monitoring metric | Not in TrainingConfig | PyTorchExecutionConfig.checkpoint_monitor_metric | TensorFlow checkpointing metric selection implicit |
| **LEARNING RATE & OPTIMIZER KNOBS** |
| `learning_rate` | `training_config.learning_rate` (PyTorch) | config_params.py:106 | 1e-3 | Base learning rate for optimizer | TrainingConfig implicit (not exposed field) | PyTorchExecutionConfig.learning_rate | TensorFlow doesn't expose LR as config field; set in script |
| `scaled_learning_rate` (computed) | `base_lr * sqrt(ebs / batch_per_gpu)` | train_utils.py:86 | computed | Scaled LR based on effective batch size (Krizhevsky scaling) | Not in TrainingConfig | Implicit in loss scaling algorithms | PyTorch explicit scaling; TF implicit in optimizer configs |
| `scheduler` | `training_config.scheduler` | config_params.py:111 | 'Default' | Learning rate scheduler type ('Default', 'Exponential', 'MultiStage', 'Adaptive') | Not in TrainingConfig (loss_weights and stages exposed separately) | PyTorchExecutionConfig.scheduler_type | PyTorch unified scheduler selection; TF has stage-specific configs |
| `stage_1_epochs` | `training_config.stage_1_epochs` | config_params.py:118 | 0 | Epoch count for stage 1 (RMS normalization) | Not directly exposed; phases are separate | TrainingConfig (already present) | Already exposed for multistage; partial parity with TF |
| `stage_2_epochs` | `training_config.stage_2_epochs` | config_params.py:119 | 0 | Epoch count for stage 2 (weighted transition) | Not directly exposed | TrainingConfig (already present) | Already exposed for multistage |
| `stage_3_epochs` | `training_config.stage_3_epochs` | config_params.py:120 | 0 | Epoch count for stage 3 (physics only) | Not directly exposed | TrainingConfig (already present) | Already exposed for multistage |
| `physics_weight_schedule` | `training_config.physics_weight_schedule` | config_params.py:121 | 'cosine' | Physics loss weight schedule ('linear', 'cosine', 'exponential') | Not in TrainingConfig | Implicit in loss weighting (mae_weight, nll_weight) | PyTorch explicit; TF uses fixed weights |
| `stage_3_lr_factor` | `training_config.stage_3_lr_factor` | config_params.py:124 | 0.1 | LR reduction factor for stage 3 physics | Not in TrainingConfig | PyTorchExecutionConfig.stage_3_lr_factor | PyTorch explicit stage-specific LR; TF implicit |
| `fine_tune_gamma` | `training_config.fine_tune_gamma` | config_params.py:110 | 0.1 | LR scale factor for fine-tuning phase | Not in TrainingConfig | PyTorchExecutionConfig.fine_tune_lr_factor | Fine-tuning specific; TF doesn't have built-in fine-tune workflow |
| `gradient_clip_val` | `training_config.gradient_clip_val` | config_params.py:114 | None | Gradient clipping threshold (None = disabled) | Not in TrainingConfig | PyTorchExecutionConfig.gradient_clip_val | TensorFlow has gradient clipping but not exposed at config level |
| **DATA LOADING KNOBS** |
| `num_workers` | `training_config.num_workers` | config_params.py:112 | 4 | DataLoader worker processes for parallel data loading | Not in TrainingConfig | PyTorchExecutionConfig.num_workers | TensorFlow data pipeline abstracted; PyTorch explicit |
| `accum_steps` | `training_config.accum_steps` (manual impl) | config_params.py:113, train.py:270 (commented) | 1 | Gradient accumulation steps for effective batch size scaling | Not in TrainingConfig | PyTorchExecutionConfig.accum_grad_batches | TensorFlow implicit in loss scaling; PyTorch explicit parameter |
| `pin_memory` | hardcoded `True` | train_utils.py:286, 297 | True | Pin DataLoader tensors to GPU memory (DDP optimization) | Not in TrainingConfig | PyTorchExecutionConfig.pin_memory | TensorFlow memory management automatic; PyTorch explicit tuning |
| `persistent_workers` | hardcoded `True` | train_utils.py:287, 298 | True | Keep DataLoader workers alive across epochs (memory/startup trade-off) | Not in TrainingConfig | PyTorchExecutionConfig.persistent_workers | PyTorch-specific optimization; no TF equivalent |
| `prefetch_factor` | hardcoded `4` | train_utils.py:288, 299 | 4 | Number of batches to prefetch per worker | Not in TrainingConfig | PyTorchExecutionConfig.prefetch_factor | PyTorch performance tuning; TF pipeline implicit |
| `shuffle` (training) | `True` → hardcoded in train_utils.py | train_utils.py:283 | True | Shuffle training data across epochs | TrainingConfig.sequential_sampling (inverse) | No change needed | TensorFlow always shuffles; PyTorch explicit; controlled via sequential_sampling |
| `shuffle` (validation) | hardcoded `False` | train_utils.py:294 | False | Never shuffle validation data | Implicit in TF (val data always sequential) | Implicit behavior | Fixed by design; no knob needed |
| `drop_last` (training) | Implicit (not set, defaults False) | workflows/components.py:449-455 | False | Drop incomplete final batch in DataLoader | Not in TrainingConfig | PyTorchExecutionConfig.drop_last_batch | TensorFlow implicit; PyTorch explicit control |
| `val_split` | `0.05` (hardcoded in train.py:210) | train.py:210 | 0.05 | Fraction of data to reserve for validation | Not in TrainingConfig | PyTorchExecutionConfig.val_split | TensorFlow explicit train/test split; PyTorch infers from data |
| `val_seed` | `42` (hardcoded in train.py:211) | train.py:211 | 42 | Random seed for reproducible train/val split | Not in TrainingConfig | PyTorchExecutionConfig.val_split_seed | TensorFlow explicit seed param; PyTorch implicit |
| **EARLY STOPPING / CHECKPOINTING** |
| `early_stop_patience` | hardcoded `100` | train.py:247 | 100 | EarlyStopping patience (epochs without improvement) | Not in TrainingConfig | PyTorchExecutionConfig.early_stop_patience | TensorFlow has no built-in early stopping; PyTorch via callbacks |
| `checkpoint_save_top_k` | hardcoded `1` | train.py:237 | 1 | ModelCheckpoint: keep top-k best models | Not in TrainingConfig | PyTorchExecutionConfig.checkpoint_save_top_k | PyTorch Lightning callback; TF implicit |
| `checkpoint_monitor` | `val_loss_label` (derived) | train.py:236 | dynamically computed | ModelCheckpoint monitoring metric | Not in TrainingConfig | PyTorchExecutionConfig.checkpoint_monitor_metric | PyTorch explicit; TF implicit |
| `checkpoint_mode` | hardcoded `'min'` | train.py:237 | 'min' | ModelCheckpoint mode ('min' for loss, 'max' for accuracy) | Not in TrainingConfig | PyTorchExecutionConfig.checkpoint_mode | PyTorch explicit; TF implicit |
| `save_last` | hardcoded `True` | train.py:240 | True | Save final model checkpoint for recovery | Not in TrainingConfig | PyTorchExecutionConfig.save_last_checkpoint | PyTorch callback feature; TF implicit |
| **DATA GENERATION / SIMULATION KNOBS** |
| `nphotons` | `DataConfig.nphotons` | config_params.py:20 | 1e5 | Simulated photon count for diffraction pattern normalization | TrainingConfig.nphotons | No change needed | Already exposed in TrainingConfig; config_bridge validates against TF default divergence |
| `objects_per_probe` | `DatagenConfig.objects_per_probe` | config_params.py:143 | 4 | Number of unique synthetic objects per probe function | Not in TrainingConfig | PyTorchExecutionConfig or DatagenConfig | Datagen-specific; not training knob |
| `diff_per_object` | `DatagenConfig.diff_per_object` | config_params.py:144 | 7000 | Number of diffraction images per unique object | Not in TrainingConfig | PyTorchExecutionConfig or DatagenConfig | Datagen-specific |
| `object_class` | `DatagenConfig.object_class` | config_params.py:145 | 'dead_leaves' | Synthetic object class for datagen | Not in TrainingConfig | PyTorchExecutionConfig or DatagenConfig | Datagen-specific |
| `beamstop_diameter` | `DatagenConfig.beamstop_diameter` | config_params.py:148 | 4 | Beamstop diameter for forward model simulation | Not in TrainingConfig | PyTorchExecutionConfig or DatagenConfig | Datagen-specific |
| **INFERENCE KNOBS** |
| `middle_trim` | `InferenceConfig.middle_trim` | config_params.py:134 | 32 | Trim pixels from middle of inference output (edge artifacts) | Not in InferenceConfig (TF) | PyTorchExecutionConfig.middle_trim | Inference-specific; PyTorch feature |
| `batch_size` (inference) | `InferenceConfig.batch_size` | config_params.py:135 | 1000 | Batch size for inference (typically larger than training) | InferenceConfig.batch_size implicit (not exposed) | PyTorchExecutionConfig.inference_batch_size | Inference knob; separate from training batch_size |
| `pad_eval` | `InferenceConfig.pad_eval` | config_params.py:137 | True | Pad evaluation edges for Nyquist frequency compliance | Not in InferenceConfig (TF) | PyTorchExecutionConfig.pad_eval | Inference-specific knob |
| `window` | `InferenceConfig.window` | config_params.py:138 | 20 | Window padding around reconstruction (edge error mitigation) | Not in InferenceConfig (TF) | PyTorchExecutionConfig.reconstruction_window | Inference post-processing knob |
| **REPRODUCIBILITY & SEEDING** |
| `subsample_seed` | `config.subsample_seed` (optional) | config_params.py not exposed; workflows/components.py:305 | 42 (default if not set) | Seed for reproducible data subsampling | TrainingConfig.subsample_seed | No change needed | Already exposed in TrainingConfig |
| `set_seed(seed, n_devices)` | `random.seed, np.random.seed, torch.manual_seed, etc.` | train_utils.py:49-60 | 42 | Global seed setting for reproducibility (RNG, CUDA, torch, numpy, python) | Not in TrainingConfig | Implicit via subsample_seed | PyTorch explicit seed setting; TF has single seed param |
| `PYTHONHASHSEED` (env var) | `os.environ["PYTHONHASHSEED"] = str(seed)` | train_utils.py:60 | str(42) | Python hash seed for reproducible hash randomization | Not in TrainingConfig | Environment variable (auto-set by set_seed) | Necessary for Python 3.3+ reproducibility |
| **ATTENTION / MODEL ARCHITECTURE (from PyTorch config, no TF equiv)** |
| `eca_encoder` | `ModelConfig.eca_encoder` | config_params.py:70 | False | Enable Efficient Channel Attention in encoder | Not in TensorFlow ModelConfig | No change needed (architecture tuning) | PyTorch-specific attention mechanism; TensorFlow uses different defaults |
| `cbam_encoder` | `ModelConfig.cbam_encoder` | config_params.py:71 | False | Enable CBAM module in encoder | Not in TensorFlow ModelConfig | No change needed (architecture tuning) | PyTorch-specific; TensorFlow not supported |
| `cbam_bottleneck` | `ModelConfig.cbam_bottleneck` | config_params.py:72 | False | Enable CBAM in bottleneck | Not in TensorFlow ModelConfig | No change needed | PyTorch-specific |
| `cbam_decoder` | `ModelConfig.cbam_decoder` | config_params.py:73 | False | Enable CBAM in decoder | Not in TensorFlow ModelConfig | No change needed | PyTorch-specific |
| `eca_decoder` | `ModelConfig.eca_decoder` | config_params.py:74 | False | Enable ECA in decoder | Not in TensorFlow ModelConfig | No change needed | PyTorch-specific |
| `spatial_decoder` | `ModelConfig.spatial_decoder` | config_params.py:75 | False | Enable spatial attention in decoder | Not in TensorFlow ModelConfig | No change needed | PyTorch-specific |
| `decoder_spatial_kernel` | `ModelConfig.decoder_spatial_kernel` | config_params.py:76 | 7 | Kernel size for spatial attention in decoder | Not in TensorFlow ModelConfig | No change needed | PyTorch-specific tuning |
| **INFERENCE WORKFLOW PARAMETERS (from workflows/components.py)** |
| `do_stitching` | Function parameter (not config) | workflows/components.py:91 | False | Whether to perform image stitching after training | Not in TrainingConfig | Workflow parameter (not config) | Runtime flag; implicit in output_dir logic |
| `flip_x` / `flip_y` / `transpose` | Function parameters | workflows/components.py:87-89 | False | Coordinate transformations for reassembly | Not in TrainingConfig | Workflow parameters (not config) | Inference-time flags; not training knobs |
| `M` (reassemble parameter) | Function parameter | workflows/components.py:90 | 20 | Patch reassembly parameter | Not in TrainingConfig | Workflow parameter (not config) | Runtime flag; not training knob |

---

## Summary by Category

### Lightning Trainer Configuration (9 knobs)
- **Currently missing from TF config**: `accelerator`, `strategy`, `check_val_every_n_epoch`, `deterministic`, `logger`
- **Already in TF config**: `max_epochs` (mapped from PyTorch `epochs` via config_bridge), `devices` (maps to `n_devices`)
- **Implicit/hardcoded in PyTorch**: `enable_checkpointing`, `enable_progress_bar`

### Distributed Training (6 knobs)
- **DDP-specific tuning**: `find_unused_parameters`, `static_graph`, `gradient_as_bucket_view`, `process_group_backend`
- **Environment variables**: `RANK`, `LOCAL_RANK` (read-only, set by launcher)
- **Status**: All DDP knobs are advanced tuning parameters not exposed in TF

### MLflow Integration (4 knobs)
- **Already in TF config**: `experiment_name`, `notes`, `model_name`
- **PyTorch-specific**: `disable_mlflow`, `checkpoint_monitor`
- **Status**: Core MLflow fields already in TrainingConfig; `disable_mlflow` is CLI-only flag

### Learning Rate & Optimizer (8 knobs)
- **Already in TF config**: `stage_1_epochs`, `stage_2_epochs`, `stage_3_epochs` (phase system)
- **PyTorch-specific**: `learning_rate` (not exposed in TF as config field), `scheduler`, `physics_weight_schedule`, `stage_3_lr_factor`, `fine_tune_gamma`, `gradient_clip_val`
- **Status**: PyTorch has explicit scheduler selection; TF uses implicit stage-based approach

### Data Loading (7 knobs)
- **Already in TF config**: `sequential_sampling` (inverse of shuffle)
- **PyTorch-specific**: `num_workers`, `accum_steps`, `pin_memory`, `persistent_workers`, `prefetch_factor`, `drop_last`, `val_split`, `val_seed`
- **Status**: PyTorch has granular dataloader tuning; TF abstracted

### Checkpointing & Early Stopping (4 knobs)
- **PyTorch-specific**: `early_stop_patience`, `checkpoint_save_top_k`, `checkpoint_monitor`, `checkpoint_mode`, `save_last`
- **Status**: No TF equivalent; PyTorch uses Lightning callbacks

### Inference (4 knobs)
- **PyTorch-specific**: `middle_trim`, `pad_eval`, `window`, `batch_size` (inference variant)
- **Status**: Inference-specific; not training knobs

### Data Generation/Simulation (4 knobs)
- **PyTorch-specific**: `nphotons` (already in TF), `objects_per_probe`, `diff_per_object`, `object_class`, `beamstop_diameter`
- **Status**: Datagen-specific; should remain separate from training config

---

## Proposed Resolution Strategy

### 1. **PyTorchExecutionConfig Dataclass** (Recommended)
Create a new dataclass in `ptycho/config/config.py` (or `ptycho_torch/config.py` if PyTorch-specific):

```python
@dataclass
class PyTorchExecutionConfig:
    """PyTorch Lightning runtime configuration not in canonical TensorFlow configs."""
    # Lightning Trainer
    accelerator: str = 'auto'  # 'cpu', 'gpu', 'tpu', 'mps', 'auto'
    strategy: Optional[str] = None  # 'auto', 'ddp', 'fsdp', 'deepspeed'; None = auto-select
    check_val_every_n_epoch: int = 1
    deterministic: bool = True
    logger_backend: str = 'tensorboard'  # 'tensorboard', 'wandb', 'mlflow', None
    
    # DDP Advanced Tuning
    ddp_find_unused_parameters: bool = False
    ddp_static_graph: bool = True
    ddp_gradient_as_bucket_view: bool = True
    ddp_backend: str = 'nccl'  # 'nccl', 'gloo', 'mpi'
    
    # Learning Rate & Optimization
    learning_rate: float = 1e-3
    scheduler_type: Literal['default', 'exponential', 'multistage', 'adaptive'] = 'default'
    physics_weight_schedule: str = 'cosine'  # 'linear', 'cosine', 'exponential'
    fine_tune_lr_factor: float = 0.1
    gradient_clip_val: Optional[float] = None
    
    # Data Loading Performance
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    drop_last_batch: bool = False
    val_split: float = 0.05
    val_split_seed: int = 42
    
    # Checkpointing & Early Stopping
    early_stop_patience: int = 100
    checkpoint_save_top_k: int = 1
    checkpoint_monitor_metric: str = 'val_loss'  # Dynamic; depends on loss config
    checkpoint_mode: Literal['min', 'max'] = 'min'
    save_last_checkpoint: bool = True
    
    # MLflow
    disable_mlflow: bool = False
    
    # Inference
    middle_trim: int = 32
    pad_eval: bool = True
    reconstruction_window: int = 20
    inference_batch_size: int = 1000
    
    # Reproducibility
    seed: int = 42
```

### 2. **Spec Update** (Medium-term)
Add §6 to `specs/ptychodus_api_spec.md` documenting PyTorch execution parameters and their semantic relationships to TensorFlow parameters.

### 3. **Config Bridge Enhancement** (Short-term)
Update `ptycho_torch/config_bridge.py` to handle `PyTorchExecutionConfig`:
- `to_pytorch_execution_config(tf_training_config, overrides)` for forward mapping (TF → PyTorch)
- Inverse function for reverse mapping where applicable

### 4. **Factory Override Dictionary** (MVP approach)
If full dataclass is not feasible, maintain factory overrides in `ptycho_torch/config_bridge.py`:

```python
PYTORCH_TRAINER_DEFAULTS = {
    'accelerator': 'auto',
    'devices': 1,
    'strategy': None,
    'check_val_every_n_epoch': 1,
    'enable_checkpointing': True,
    'enable_progress_bar': True,
    'deterministic': True,
    # ... etc
}
```

---

## Risk Matrix: Missing Knobs

| Knob | Impact | Ease of Addition | Notes |
|------|--------|------------------|-------|
| `learning_rate` | HIGH | EASY | Currently hardcoded in workflows/components.py:538; should be exposed |
| `num_workers` | MEDIUM | EASY | Already in PyTorch config; needs bridge mapping |
| `scheduler_type` | MEDIUM | MEDIUM | Requires scheduler factory refactor |
| `deterministic` | LOW | EASY | Reproducibility flag; good to expose |
| `ddp_backend` | LOW | HARD | Advanced tuning; rarely changed |
| `early_stop_patience` | MEDIUM | EASY | Currently hardcoded; should be configurable |
| `accelerator` | HIGH | EASY | Critical for hardware selection; should be exposed |

---

## Next Steps

1. **Phase E3**: Design `PyTorchExecutionConfig` dataclass
2. **Phase E4**: Integrate with `config_bridge` for TensorFlow↔PyTorch mapping
3. **Phase E5**: Update CLI (`ptycho_torch/train.py:cli_main`) to accept PyTorch-specific knobs as CLI flags
4. **Phase E6**: Update specs/ptychodus_api_spec.md §6 with PyTorch execution parameter contracts
5. **Phase E7**: Add validation and tests for new execution config

---

## References

- Source files analyzed:
  - `/home/ollie/Documents/PtychoPINN2/ptycho_torch/config_params.py` (DataConfig, ModelConfig, TrainingConfig, InferenceConfig)
  - `/home/ollie/Documents/PtychoPINN2/ptycho_torch/train.py` (CLI interface, Lightning trainer instantiation)
  - `/home/ollie/Documents/PtychoPINN2/ptycho_torch/train_utils.py` (Trainer setup, DDP strategy, seeding)
  - `/home/ollie/Documents/PtychoPINN2/ptycho_torch/workflows/components.py` (Lightning workflow orchestration)
  - `/home/ollie/Documents/PtychoPINN2/ptycho/config/config.py` (TensorFlow canonical configs for comparison)
  - `/home/ollie/Documents/PtychoPINN2/ptycho_torch/config_bridge.py` (Current PyTorch→TensorFlow adapter)

- Key specs:
  - Phase E2.C1: PyTorch CLI interface design (train.py)
  - CONFIG-001: Legacy params.cfg initialization requirement
  - POLICY-001: PyTorch mandatory dependency (Phase F3.1/F3.2)

