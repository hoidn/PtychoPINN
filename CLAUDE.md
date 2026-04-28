# CLAUDE.md - PtychoPINN Project Guide

## Project Overview
**PtychoPINN** (Ptychographic Physics-Informed Neural Network) is a deep learning framework for coherent diffraction imaging reconstruction. It combines UNet-based encoder-decoder architectures with physics-informed forward models to reconstruct complex-valued objects from ptychographic diffraction measurements. The project uses PyTorch Lightning for training orchestration.

## Available Plugins

### Context7 Documentation Plugin
- Use `/pytorch/pytorch` for PyTorch documentation
- Use `/lightning-ai/pytorch-lightning` for PyTorch Lightning documentation
- Example: "query pytorch docs for nn.Module best practices"

### GitHub Plugin
- Available for GitHub operations (issues, PRs, commits)
- Project repo: `anthropics/claude-code` for feedback/issues
- Do not add claude coauthorship to commit notes/messages

### Conda environment
The correct conda environment for python to import is called PtychoPINN_torch, located at `/local/miniconda3/envs/PtychoPINN_torch`

## Directory Structure

```
ptycho_torch/
├── api/              # High-level training/inference interfaces
├── beta_modules/     # Experimental model variants (rectangular nets, UNet, scaling modules)
├── configs/          # JSON configuration files for experiments
├── datagen/          # Synthetic data generation (objects, probes, diffraction)
├── eval/             # Evaluation metrics (MSE, SSIM, FRC)
└── notebooks/        # Analysis and manuscript figures
```

Other directories such as ptycho contain other implementations that we cannot, and will not modify. We are concerned only with the ptycho_torch repository, which is the pytorch implementation of the architecture.

---

## Core Files

### Model Architecture (`model.py`, `beta_modules/model_unet.py`, `beta_modules/latent_model.py`)

**PtychoPINN_Lightning** - Main Lightning module for training
- Multi-stage training with configurable RMS/physics normalization weights
- Automatic gradient accumulation and clipping for DDP
- Supports supervised and unsupervised modes
- Methods: `freeze_encoder()`, `get_encoder_bottom_params()`, `print_trainable_status()`

**Autoencoder** - Base encoder-decoder architecture with skip connections for U-net
- Encoder: ConvPoolBlock layers with optional CBAM/ECA attention
- Decoder_amp: Amplitude decoder with SiLU activation
- Decoder_phase: Phase decoder with tanh activation (batch norm enabled)
- Forward: Returns (amplitude, phase) tuple

**Latent Autoencoder** - Alternate latent autoencoder using joint latents encoding shared overlapping information
- Encoder: Either conv or Fourier Neural Operator + conv into latent space representation of canvas
- Positional encoding: Fourier based positional encoding for latent pixel distances from measurement positions
- Measurement fusion: Uses geometry-aware cross-attention to merge measurement latents
- Decoder: Neural field decoder extracting objects from joint representation

**ForwardModel** - Physics-informed forward pass
- Reassembles patches using `reassemble_patches_position_real()`
- Applies probe illumination via hadamard product
- Performs FFT to simulate diffraction
- Optional trainable intensity scaling per dataset (mandatory for model_unet.py)

**Loss Functions**
- PoissonLoss: Negative log-likelihood for photon counting
- MAELoss: Mean absolute error on intensities
- TotalVariationLoss: Smoothness regularization
- MeanDeviationLoss: Penalizes amplitude/phase variations

---

### Attention Mechanisms (`model_attention.py`)

**CBAM** (Convolutional Block Attention Module)
- ChannelGate: MLP-based channel attention with avg/max pooling
- SpatialGate: Conv-based spatial attention (kernel size 7)
- Used in encoder (optional), bottleneck, and decoder

---

### Data Loading (`dataloader.py`)

**PtychoDataset** - Memory-mapped dataset for large ptychography scans
- Creates TensorDict memory maps for efficient I/O
- Handles multi-experiment datasets with coordinate filtering
- Supports DDP with rank-aware map creation
- Fields: images, coords_global, coords_relative, nn_indices, scaling_constant
- Methods: `get_experiment_dataset()` to index specific dataset for multi-dataset construction

**Coordinate Grouping** (`patch_generator.py`)
- `group_coords()`: Creates overlapping patch groups for physics constraints
- `get_fixed_quadrant_neighbors_c4()`: 4-quadrant sampling for rectangular scans with fixed relative positioning
- `get_neighbors_indices_within_bounds()`: KDTree-based neighbor search with distance filters

**Collate_Lightning**
- Pin memory for CUDA transfers
- Device-agnostic (Lightning handles placement)

---

### Training (`train_lightning_only.py`, `train_utils.py`)

**main()** - Lightning-only training (no MLFlow)
- Creates TensorBoard/CSV loggers with unified run naming
- Configurable multi-GPU via DDPStrategy
- Checkpoint callbacks with best/last model saving
- Optional fine-tuning with frozen encoder

**PtychoDataModuleLightning**
- `prepare_data()`: Rank 0 creates memory map
- `setup()`: All ranks load existing map (Lightning adds barrier)
- Returns train/val dataloaders with persistent workers

**ModelFineTuner** (`model_finetuner_modified.py`)
- Freezes encoder, scales LR by `fine_tune_gamma`
- Creates separate run directory for fine-tuning stage
- Logs configs and metadata via Lightning callbacks

**StagedFineTuner_Lightning** (Beta)
- Stage 1: Decoder-only (adapt object space)
- Stage 2: Partial encoder + discriminative LR
- Stage 3: Full network with conservative LR
- Configurable per-stage LR multipliers

---

### Inference & Reconstruction (`inference.py`, `reassembly.py`, `beta_modules/reassembly.py`)

**reconstruct_image_barycentric()** - Multi-GPU reconstruction
- Vectorized barycentric interpolation for patch assembly
- Mixed precision support (FP16/FP32)
- DataParallel wrapper for multi-GPU inference
- Returns assembled canvas with normalized counts

**reconstruct_image_barycentric_weighted()** - Multi-GPU reconstruction in beta modules
- Vectorized barycentric interpolation for patch assembly
- Weighted by probe intensities instead of pixel-dependent image contribution count
- Accumulates predicted intensities and actual intensities for fitting by VarProScaler
- Returns scaled, assembled canvas with solved physics scaling constants

**reassemble_multi_channel()**
- Assembles overlapping patches into solution regions
- Translates solution patches to global canvas
- Weighted averaging based on overlap counts

**VectorizedBarycentricAccumulator**
- Bilinear interpolation weights for sub-pixel positioning
- Scatter-add operations for efficient accumulation
- Handles bounds checking for patch validity

**VectorizedWeightedccumulator**
- Identical to VectorizedBarycentricAccumulator except for weighted normalization canvas
- Uses probe magnitude to calculate normalization weights

**VarProScaler**
- Accumulates predicted and real intensity values for whole dataset during inference
- Uses various optimization methods (e.g. Newton, LBFGS) to solve for photon scaling constants

---

### Data Generation (`datagen/datagen.py`, `datagen/objects.py`)

**simulate_multiple_experiments()**
- Generates synthetic datasets with controllable parameters
- Saves: diff3d, label, objectGuess, probeGuess, coords
- Supports batch processing with timing metrics

**Synthetic Object Methods**
- `create_dead_leaves()`: Overlapping shapes with realistic textures
- `create_white_noise_object()`: Blurred noise with material parameters
- `create_simplex_noise_object()`: Perlin/simplex noise-based structures
- `generate_perlin_object()`: Batch generation with correlated amp/phase

**Probe Generation** (`datagen/probe.py`)
- `generate_random_zernike()`: Zernike polynomial-based probes
- `generate_random_fzp()`: Fresnel zone plate probes
- Optional ramp removal and normalization

---

### Configuration (`config_params.py`)

**DataConfig**
- N: Diffraction pattern size (64/128/256)
- C: Number of channels (1=CDI, 4=ptychography)
- normalize: 'Batch' or 'Group' normalization strategy
- x_bounds, y_bounds: Coordinate filtering (avoid edges)
- scan_pattern: Matches experiment scan pattern, common failure mode if scan pattern is NOT matched

**ModelConfig**
- mode: 'Supervised' or 'Unsupervised'
- loss_function: 'Poisson' or 'MAE'
- cbam_encoder/decoder: Enable CBAM attention
- object_big: Enable overlap constraint (True when C > 1)

**TrainingConfig**
- epochs, learning_rate, batch_size
- stage_1/2/3_epochs: Multi-stage normalization schedule
- scheduler: 'Cosine', 'MultiStage', 'Adaptive'
- enable_staged_finetuning: Cross-domain transfer learning

**InferenceConfig**
- middle_trim: Central region size for reconstruction
- batch_size: Inference batch size (reduce for memory)
- experiment_number: Dataset index for multi-experiment inference

---

### Utilities (`helper.py`, `utils.py`)

**reassemble_patches_position_real()** (`helper.py`)
- Translates patches to global coordinates
- Aggregates overlaps with weighted averaging
- Returns merged canvas + validity mask

**Translation()** (`helper.py`)
- Sub-pixel translation using grid_sample
- Supports complex tensors via split real/imag
- Jitter support for data augmentation

**config_to_json_serializable_dict()** (`utils.py`)
- Converts dataclasses to JSON (skips Tensors)
- Used for checkpoint metadata logging

**load_all_configs_from_mlflow()** (`utils.py`)
- Deserializes configs from MLFlow run parameters
- Post-processing for legacy attribute fixes

---

### Evaluation (`eval/eval_metrics.py`, `eval/frc.py`)

**FSC()** - Fourier Shell Correlation
- Frequency-domain resolution metric
- Returns FSC curve, threshold, and spatial frequencies
- Based on van Heel & Schatz (2005)

**PSNR()** - Peak Signal-to-Noise Ratio
- Separate metrics for amplitude and phase
- Energy-based normalization for fair comparison
- Phase normalized to [0, 1] via 2π wrapping

---

## Beta Modules

### `beta_modules/model.py` - Rectangular Coordinate Net
- Specialized for anisotropic scan patterns
- Custom quadrant-based coordinate grouping
- Modified reassembly for non-square grids

### `beta_modules/model_unet.py` - UNet Variant
- Skip connections between encoder/decoder
- Residual blocks with batch normalization
- Alternative architecture for comparison

### `beta_modules/reassembly.py` - Weighted Reassembly
- Probe-weighted patch assembly
- Circular mean for phase averaging
- S1/S2 scaling factor optimization (Newton's method)

---

## Common Patterns

### Training a Model
```python
from ptychopinn_torch.train_lightning_only import main
from ptychopinn_torch.utils import load_config_from_json

# Load config and train
run_dir = main(
    ptycho_dir='data/synthetic/',
    config_path='configs/my_config.json',
    output_dir='training_outputs/'
)
```

### Inference on New Data
```python
from ptychopinn_torch.lightning_utils import load_checkpoint_with_configs
from ptychopinn_torch.reassembly import reconstruct_image_barycentric

# Load model
model, configs = load_checkpoint_with_configs(
    'training_outputs/.../best-checkpoint.ckpt',
    PtychoPINN_Lightning
)

# Reconstruct
canvas, dataset, stats = reconstruct_image_barycentric(
    model, ptycho_dataset,
    training_config, data_config, model_config, inference_config
)
```

### Generating Synthetic Data
```python
from ptychopinn_torch.datagen.datagen import simulate_multiple_experiments
from ptychopinn_torch.datagen.objects import simulate_synthetic_objects
from ptychopinn_torch.datagen.probe import simulate_synthetic_probes

# Create objects and probes
obj_list = simulate_synthetic_objects(
    img_shape=(250,250), data_config=data_config,
    nimages=10, obj_method='dead_leaves', obj_arg={}
)
probe_list = simulate_synthetic_probes(
    data_config, nimages=10, probe_method='zernike', probe_arg={}
)

# Generate datasets
simulate_multiple_experiments(
    obj_list, probe_list, images_per_experiment=5000,
    img_shape=(250,250), data_config=data_config,
    probe_arg={}, save_dir='data/synthetic/'
)
```

---

## Coding Conventions

1. **Config Objects**: Pass `DataConfig`, `ModelConfig`, etc. as arguments (never hardcode values)
2. **Complex Tensors**: Use `torch.complex64`, handle via `torch.real()` / `torch.imag()` for operations not supporting complex
3. **DDP Synchronization**: Always use `dist.barrier()` for cross-rank coordination
4. **Memory Management**: Call `torch.cuda.empty_cache()` and `gc.collect()` after large batches
5. **Logging**: Use `print()` with rank guards: `if is_effectively_global_rank_zero():`
6. **Device Placement**: Lightning handles `.to(device)`, but dataloaders need explicit pin_memory

---

## Key Architecture Decisions

- **Rectangular Coordinates**: Specialized `4_quadrant` neighbor function for anisotropic scans (±12 μm in x, ±2 μm in y)
- **Multi-Stage Training**: Gradual transition from RMS → Physics normalization prevents loss landscape jumps
- **Lightning-Only Mode**: Removed MLFlow dependency for simpler deployment and debugging
- **Memory-Mapped Datasets**: TensorDict enables multi-TB datasets without RAM limits
- **Probe-Weighted Assembly**: Uses |Probe|² for physically accurate overlap weighting
- **Incoherent Multi-Mode Probes**: The beta modules (`model_unet.py`, `reassembly.py`) support multiple incoherent probe modes via the tensor layout `[B, C, P, H, W]` where P is the probe mode dimension. The dataloader auto-detects 2D (single-mode) vs 3D (multi-mode) probes from NPZ files. The forward model uses incoherent summation: $I = \sum_p |\mathcal{F}\{O \cdot P_p\}|^2$. For inference assembly, stitching weights use total probe intensity $W(\mathbf{r}) = \sum_p |P_p(\mathbf{r})|^2$. The VarPro scaler computes mode-summed basis images before solving for scaling constants. All changes are backward-compatible — when P=1, behavior is identical to the single-mode case.

---

## Common Issues

**"Memory map contains zeros on Rank 1"**
- Check `os.sync()` is called after Rank 0 creates map
- Verify `dist.barrier()` between `prepare_data()` and `setup()`

**"Phase reconstruction has constant offset"**
- Enable `phase_subtraction=True` in DataConfig
- Verify `batch_norm=True` in phase decoder

**"Model gradients are NaN"**
- Check `gradient_clip_val` is set (recommend 1.0-5.0)
- Reduce learning rate or increase `stage_1_epochs`

**"Reconstruction has stitching artifacts"**
- Increase `middle_trim` (32 → 48 for N=64)
- Check probe normalization is disabled during inference
- Verify `x_bounds`/`y_bounds` filter edge positions

---

## Thinking Scaffolding

**Ensure physics and scientific rigor**
- Ensure suggestions are grounded in physics and have mathematical rigor
- Format scientific explanatory outputs with latex-compatible formatting (e.g. equations)


