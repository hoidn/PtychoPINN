# CLAUDE.md - PtychoPINN Project Guide

## Project Overview
**PtychoPINN** (Ptychographic Physics-Informed Neural Network) is a deep learning framework for coherent diffraction imaging reconstruction. It combines UNet-based encoder-decoder architectures with physics-informed forward models to reconstruct complex-valued objects from ptychographic diffraction measurements. Training is orchestrated with PyTorch Lightning.

## Environment & Scope

- Conda environment: `PtychoPINN_torch`, located at `/local/miniconda3/envs/PtychoPINN_torch`
- **Only `ptycho_torch/` may be modified.** Other directories (e.g. `ptycho/`) contain other implementations we cannot and will not touch.

## Directory Layout

```
ptycho_torch/
├── api/              # High-level training/inference interfaces + usage examples
├── beta_modules/     # Experimental variants: rectangular-coordinate net, UNet, weighted reassembly
├── cli/              # Shared CLI helpers
├── configs/          # JSON configuration files for experiments
├── datagen/          # Synthetic data generation (objects, probes, diffraction)
├── eval/             # Evaluation metrics (PSNR, FRC/FSC)
├── generators/       # Generator networks (cnn.py)
└── notebooks/        # Analysis and manuscript figures
```

Key top-level modules (read the source for API details — signatures drift faster than docs):

- `model.py` — `PtychoPINN_Lightning` (main Lightning module), `Autoencoder`, `ForwardModel`, loss functions (Poisson, MAE, TV, MeanDeviation)
- `model_attention.py` — CBAM/ECA attention blocks
- `dataloader.py` — `PtychoDataset` (TensorDict memory-mapped, DDP-aware), `Collate_Lightning`
- `patch_generator.py` — coordinate grouping (`group_coords`, quadrant/KDTree neighbor search)
- `train_lightning_only.py` — `main()` training entry point (Lightning-only, no MLFlow)
- `train_utils.py` — `PtychoDataModuleLightning`
- `model_finetuner_modified.py` — `ModelFineTuner`, `StagedFineTuner_Lightning`
- `inference.py`, `reassembly.py` — `reconstruct_image_barycentric()` and patch-assembly machinery
- `beta_modules/reassembly.py` — `reconstruct_image_barycentric_weighted()`, `VarProScaler` (probe-weighted assembly + solved scaling constants)
- `config_params.py` — `DataConfig`, `ModelConfig`, `TrainingConfig`, `InferenceConfig` dataclasses
- `helper.py` — `reassemble_patches_position_real()`, sub-pixel `Translation()`
- `datagen/datagen.py` — `simulate_multiple_experiments()`, `simulate_synthetic_objects()`, `simulate_synthetic_probes()`
- `eval/eval_metrics.py`, `eval/frc.py` — PSNR, FSC (van Heel & Schatz 2005)

## Configuration Gotchas

- `DataConfig.scan_pattern` must match the experiment's scan pattern — mismatch is a common failure mode
- `DataConfig.C`: 1 = CDI, 4 = ptychography; `ModelConfig.object_big` (overlap constraint) should be True when C > 1
- `DataConfig.x_bounds` / `y_bounds` filter edge scan positions

## Common Patterns

### Training a Model
```python
from ptycho_torch.train_lightning_only import main

run_dir = main(
    ptycho_dir='data/synthetic/',
    config_path='configs/my_config.json',
    output_dir='training_outputs/'
)
```

### Inference on New Data
```python
from ptycho_torch.lightning_utils import load_checkpoint_with_configs
from ptycho_torch.reassembly import reconstruct_image_barycentric

model, configs = load_checkpoint_with_configs(
    'training_outputs/.../best-checkpoint.ckpt',
    PtychoPINN_Lightning
)
canvas, dataset, stats = reconstruct_image_barycentric(
    model, ptycho_dataset,
    training_config, data_config, model_config, inference_config
)
```

### Generating Synthetic Data
```python
from ptycho_torch.datagen.datagen import (
    simulate_multiple_experiments, simulate_synthetic_objects, simulate_synthetic_probes
)

obj_list = simulate_synthetic_objects(
    img_shape=(250, 250), data_config=data_config,
    nimages=10, obj_method='dead_leaves', obj_arg={}
)
probe_list = simulate_synthetic_probes(
    data_config, nimages=10, probe_method='zernike', probe_arg={}
)
simulate_multiple_experiments(
    obj_list, probe_list, images_per_experiment=5000,
    img_shape=(250, 250), data_config=data_config,
    probe_arg={}, save_dir='data/synthetic/'
)
```

## Coding Conventions

1. **Config Objects**: Pass `DataConfig`, `ModelConfig`, etc. as arguments (never hardcode values)
2. **Complex Tensors**: Use `torch.complex64`, handle via `torch.real()` / `torch.imag()` for operations not supporting complex
3. **DDP Synchronization**: Always use `dist.barrier()` for cross-rank coordination
4. **Memory Management**: Call `torch.cuda.empty_cache()` and `gc.collect()` after large batches
5. **Logging**: Use `print()` with rank guards: `if is_effectively_global_rank_zero():`
6. **Device Placement**: Lightning handles `.to(device)`, but dataloaders need explicit pin_memory

## Repo conventions 
- don't mention claude or claude code in commit messages 

## Key Architecture Decisions

- **Rectangular Coordinates**: Specialized `4_quadrant` neighbor function for anisotropic scans (±12 μm in x, ±2 μm in y)
- **Multi-Stage Training**: Gradual transition from RMS → Physics normalization prevents loss landscape jumps
- **Lightning-Only Mode**: Removed MLFlow dependency for simpler deployment and debugging
- **Memory-Mapped Datasets**: TensorDict enables multi-TB datasets without RAM limits
- **Probe-Weighted Assembly**: Uses |Probe|² for physically accurate overlap weighting
- **Incoherent Multi-Mode Probes**: The beta modules (`model_unet.py`, `reassembly.py`) support multiple incoherent probe modes via the tensor layout `[B, C, P, H, W]` where P is the probe mode dimension. The dataloader auto-detects 2D (single-mode) vs 3D (multi-mode) probes from NPZ files. The forward model uses incoherent summation: $I = \sum_p |\mathcal{F}\{O \cdot P_p\}|^2$. For inference assembly, stitching weights use total probe intensity $W(\mathbf{r}) = \sum_p |P_p(\mathbf{r})|^2$. The VarPro scaler computes mode-summed basis images before solving for scaling constants. All changes are backward-compatible — when P=1, behavior is identical to the single-mode case.

## Common Issues

**"Memory map contains zeros on Rank 1"**
- Check `os.sync()` is called after Rank 0 creates map
- Verify `dist.barrier()` between `prepare_data()` and `setup()`

**"Phase reconstruction has constant offset"**
- Enable `phase_subtraction=True` in DataConfig
- Verify `batch_norm=True` in phase decoder

**"Model gradients are NaN"**
- Check `gradient_clip_val` is set (recommend 1.0-5.0)
- Reduce learning rate or increase `finetune_stage1_epochs`

**"Reconstruction has stitching artifacts"**
- Increase `middle_trim` (32 → 48 for N=64)
- Check probe normalization is disabled during inference
- Verify `x_bounds`/`y_bounds` filter edge positions

## Plugins

- **Context7**: `/pytorch/pytorch` and `/lightning-ai/pytorch-lightning` for library docs
- **GitHub**: available for issues/PRs/commits

## Thinking Scaffolding

**Ensure physics and scientific rigor**
- Ground suggestions in physics with mathematical rigor
- Format scientific explanatory outputs with LaTeX-compatible equations
