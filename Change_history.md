# Change History

## 2026-05-05 — Auto GPU device detection

Added `"auto"` option for `n_devices` in `TrainingConfig`. When set, `resolve_n_devices()` calls `torch.cuda.device_count()` at training entry points, resolving to the actual GPU count before any downstream usage (LR scaling, strategy selection, Trainer). Falls back to 1 on CPU-only machines. HPC-robust: `torch.cuda.device_count()` respects `CUDA_VISIBLE_DEVICES` set by PBS/Slurm schedulers. Usage: `{"n_devices": "auto"}` in config JSON. Existing integer values continue to work unchanged.

**Files**: `ptycho_torch/config_params.py`, `ptycho_torch/train_utils.py`, `ptycho_torch/train.py`, `ptycho_torch/train_lightning_only.py`, `ptycho_torch/beta_modules/train_ccnf.py`

## 2026-05-01 — Circle regularization losses

Added three opt-in losses to push autoencoder outputs toward a scaled unit circle (uniform amplitude, rich phase). `AmplitudeVarianceLoss` penalizes spatial variance of |z|. `ModulusTargetLoss` penalizes deviation of |z| from a target with a dead zone. `ChannelEnergyBalanceLoss` penalizes real/imag energy imbalance. All disabled by default.

**Files**: `ptycho_torch/model.py`, `ptycho_torch/config_params.py`, `ptycho_torch/configs/amsc_configs/velociprobe_single_decoder_rectangular.json`

## 2026-04-30 — N-aware feature scaling for synthetic object generation

Synthetic object features (leaf dimensions, blur sigmas, Perlin scale, texture correlation) now scale automatically with `data_config.N` via `_pixel_scale(N) = N / 64`. This keeps relative feature coverage constant across all values of N — e.g., `r_min/N` stays at 7.8% whether N=64 or N=128. Default `r_min` raised from 1 to 5 pixels (at N=64 baseline) so the smallest features remain resolvable by the network. Scaling is passed from `simulate_synthetic_objects` through to all wrapper functions (`create_dead_leaves_v3`, `create_dead_leaves_reim_gmm`, `create_perlin_reim`, `create_white_noise_clustered_reim`, `create_white_noise_object`) and `generate_base_map_shapes_perlin_layers`.

**Files**: `ptycho_torch/datagen/objects.py`, `ptycho_torch/datagen/datagen.py`

## 2026-04-29 — Fix `dead_leaves_reim_gmm` Re-Im distribution sampling

### Problem

The `dead_leaves_reim_gmm` synthetic object generator produced Re-Im distributions that did not match experimental data. The root cause was the GMM perturbation step (`_perturb_gmm_config`) applying a full [0, 2pi] rotation of all cluster centers around their weighted centroid. For ptychographic transmission functions concentrated near (1, 0) in Re-Im space, this rotation maps clusters to unphysical locations — e.g., a 90-degree rotation sends vacuum at (0.95, 0) to (0, 0.95), swapping absorption and phase axes. Secondary issues included an overfit default of `n_clusters=10` and a hardcoded clip range.

### Changes

**`ptycho_torch/datagen/objects.py`**

- `_perturb_gmm_config()`: Added `perturbation_mode` parameter. New default `'physical'` applies a small phase jitter (rotation around the origin, std=0.1 rad) and log-normal amplitude scaling (std=0.03), preserving the physical Re-Im structure. Legacy `'rotation'` mode retained for backward compatibility.
- `fit_gmm_from_objects()`: Restored `n_clusters='auto'` as default (BIC-based selection, K=2..8). Made `origin_mask_radius` a configurable parameter (default 0.4).
- `generate_dead_leaves_reim_gmm()`: Made `clip_range` a configurable parameter (default (-1.2, 1.2)).
- `create_dead_leaves_reim_gmm()`: Wired new parameters (`perturbation_mode`, `phase_jitter_std`, `amplitude_scale_std`, `clip_range`) through `obj_arg`.

**`ptycho_torch/config_params.py`**

- Added fields to `DatagenConfig`: `gmm_n_clusters`, `perturbation_mode`, `phase_jitter_std`, `amplitude_scale_std`, `center_jitter_std`, `weight_dirichlet_conc`, `gmm_clip_range`, `origin_mask_radius`.

**`ptycho_torch/train_full.py`**

- `prepare_data()`: Passes `datagen_config.gmm_n_clusters` and `origin_mask_radius` to `fit_gmm_from_objects()`. Populates `obj_arg` with perturbation parameters from config.

**`ptycho_torch/configs/amsc_configs/velociprobe_single_decoder_rectangular.json`**

- Switched `object_class` from `dead_leaves_reim_hist` to `dead_leaves_reim_gmm` with `perturbation_mode: "physical"`.

### Validation

Tested against `data/amsc_ic2/AmSC_IC_2_ptychopinn.npz` (Re mean=0.92, Im mean=-0.02, corr=0.06). Physical perturbation produces Re means 0.86–0.93 and Im stds 0.31–0.37, closely matching experimental statistics. Old rotation perturbation produced Re means as low as 0.63 and Im stds as low as 0.07.
