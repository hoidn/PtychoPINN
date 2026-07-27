# File-by-File Change Summary: TrainingConfig Grouping Refactor

## `ptycho/config/config.py`

**What changed:** Nine new dataclasses added; `TrainingConfig` fields collapsed into them; `dataclass_to_legacy_dict` extended with flattening blocks; `validate_training_config` paths updated.

**New classes:**

| Class | Contains | Reason |
|---|---|---|
| `AdamConfig` | `beta1`, `beta2` | Adam/AdamW-specific hyperparams |
| `SgdConfig` | `momentum` | SGD-specific hyperparam |
| `OptimizerConfig` | `algorithm`, `weight_decay`, `adam`, `sgd` | Groups all optimizer knobs |
| `LossConfig` | `torch_loss_mode`, `torch_mae_pred_l2_match_target` | PyTorch-only loss settings |
| `TFLossConfig` | `mae_weight`, `nll_weight`, `realspace_mae_weight`, `realspace_weight` | TF-only loss weights; `TF` prefix marks them for eventual deletion |
| `GradientClipConfig` | `val`, `algorithm` | Two clipping knobs always travel together |
| `SchedulerConfig` | `kind`, `lr_warmup_epochs`, `lr_min_ratio`, `plateau_*` | All LR-scheduler knobs |
| `DataConfig` | `train_data_file`, `test_data_file`, `nphotons` | Data paths + physics scale |
| `SamplingConfig` | `n_groups`, `n_subsample`, `subsample_seed`, `neighbor_count`, `enable_oversampling`, `neighbor_pool_size`, `sequential_sampling` | All grouped-data sampling knobs; migration from deprecated `n_images` moved here |

**`TrainingConfig` after the refactor** retains only eight flat fields (`model`, `batch_size`, `nepochs`, `positions_provided`, `probe_trainable`, `intensity_scale_trainable`, `output_dir`, `backend`) plus the seven nested configs above.

**`dataclass_to_legacy_dict`** gained seven flattening blocks — one per nested config — that unpack the sub-dicts produced by `asdict()` back into the flat key names that `params.cfg` consumers expect. Without these blocks, 23+ TF legacy modules would silently receive dict objects instead of scalars.

---

## `ptycho/metadata.py`

**What changed:** Import updated; three attribute accesses updated to nested paths.

| Old | New |
|---|---|
| `config.nphotons` | `config.data.nphotons` |
| `config.nll_weight` | `config.tf_loss.nll_weight` |
| `config.n_images` | `config.sampling.n_groups` |

`load_config_from_metadata` constructor call updated to pass `DataConfig`, `SamplingConfig`, and `TFLossConfig` instances instead of flat kwargs.

---

## `ptycho/nongrid_simulation.py`

**What changed:** Two attribute accesses and one docstring example updated.

| Old | New |
|---|---|
| `config.n_groups` / `config.n_images` | `config.sampling.n_groups` / `config.sampling.n_images` |
| `config.nphotons` | `config.data.nphotons` |

Reason: `_generate_simulated_data_legacy_params` reads these fields to set legacy global state before simulation.

---

## `ptycho/workflows/components.py`

**What changed:** Seven arguments to `data.generate_grouped_data()` inside `create_ptycho_data_container` updated from flat `config.*` to nested `config.sampling.*` and `config.data.*`.

Reason: This is the TF-side data container creation function; it reads grouping parameters directly from the config object.

---

## `ptycho/workflows/grid_lines_workflow.py`

**What changed:** `DataConfig` and `TFLossConfig` imported; `configure_legacy_params` constructor call updated.

The function builds a `TrainingConfig` from a `GridLinesConfig`. The flat kwargs `nphotons=`, `nll_weight=`, `mae_weight=`, `realspace_weight=` were replaced with `data=DataConfig(...)` and `tf_loss=TFLossConfig(...)`.

---

## `ptycho_torch/config_bridge.py`

**What changed:** All nine new config classes imported; `to_training_config` restructured to use a flat-then-nest pattern.

**Why the flat-then-nest pattern:** Callers historically pass flat override dicts (e.g., `overrides=dict(n_groups=512, nphotons=1e9, mae_weight=0.3)`). The new approach:
1. Collects all raw scalar values into a `flat` dict.
2. Applies `flat.update(overrides)` — overrides remain flat key names.
3. Validates on `flat`.
4. Constructs all nested config instances from `flat`.
5. Assembles the final `TFTrainingConfig(**kwargs)`.

This preserves full backward compatibility for all override call sites while the internal representation uses nested configs. Top-level overridable fields (`positions_provided`, `probe_trainable`, `batch_size`, `nepochs`, `backend`, `intensity_scale_trainable`) were also added to `flat` so they can be overridden by callers — previously they were hardcoded in `kwargs` and silently ignored if passed as overrides.

---

## `ptycho_torch/config_factory.py`

**What changed:** One line — `TFTrainingConfig(model=TFModelConfig()).nphotons` → `.data.nphotons`.

Used to read the TF default photon count for comparison; needed the nested path.

---

## `ptycho_torch/model.py`

**What changed:** `configure_optimizers` reads optimizer and scheduler fields through nested paths.

Two local helper variables introduced to avoid triple-nested `getattr` chains:
- `_opt = getattr(self.training_config, 'optimizer', None)` — all five optimizer sub-fields read from `_opt`
- `_sched = getattr(self.training_config, 'scheduler', None)` — `kind` and all plateau/warmup params read from `_sched`

---

## `ptycho_torch/workflows/components.py`

**What changed:** Multiple flat `config.*` accesses updated across four functions, plus the opt-field loop restructured.

| Function | Fields updated |
|---|---|
| `_resolve_nphotons` | `config.nphotons` → `config.data.nphotons` via `data_cfg` intermediate |
| `_ensure_container` | `config.neighbor_count`, `config.n_groups`, `config.train_data_file`, `config.sequential_sampling`, `config.subsample_seed` → `config.sampling.*` / `config.data.*` |
| `_train_with_lightning` | `config.n_groups`, `config.nphotons`, `config.neighbor_count`, `config.subsample_seed`, `config.torch_loss_mode`, `config.torch_mae_pred_l2_match_target` → nested paths; old flat `getattr` loop replaced with explicit extraction blocks for each nested config (`opt_cfg`, `sched_cfg`, `gc_cfg`) |

The redundant `loss_cfg` extraction block was removed — `torch_loss_mode` was already set from `config.loss.*` earlier in the dict, and the block ran after the supervised-model override, silently clobbering it.

---

## `scripts/compare_models.py`

**What changed:** `TrainingConfig` constructor call updated; `DataConfig` and `SamplingConfig` imported.

`final_config` used flat `train_data_file=`, `n_groups=`, `neighbor_count=` kwargs; replaced with `data=DataConfig(...)` and `sampling=SamplingConfig(...)`.

---

## `scripts/inference/baseline_inference.py`

**What changed:** `TrainingConfig` constructor calls and attribute accesses updated to nested paths throughout the file.

---

## `scripts/studies/cdi_natural_patch_benchmark.py`

**What changed:** Constructor calls and the factory-override extraction block updated.

The opt-field loop that used `getattr(config, 'scheduler', None)` etc. was replaced with explicit extraction from `config.scheduler.*`, `config.optimizer.*`, and `config.gradient_clip.*`.

---

## `scripts/studies/dose_response_study.py`

**What changed:** `TrainingConfig` constructor calls updated to use `DataConfig`, `SamplingConfig`, and `TFLossConfig` nested instances.

---

## `scripts/studies/grid_lines_torch_runner.py`

**What changed:** `setup_training_config` constructor and all post-init assignments updated.

Constructor now passes `data=DataConfig(...)` and `loss=LossConfig(...)` instead of flat `train_data_file=`, `torch_loss_mode=`.
Post-init assignments updated:
- `training_config.gradient_clip.val`, `training_config.gradient_clip.algorithm`
- `training_config.optimizer.algorithm`, `.weight_decay`, `.adam.beta1/2`, `.sgd.momentum`
- `training_config.scheduler.kind`, `.lr_warmup_epochs`, `.lr_min_ratio`, `.plateau_*`
- `training_config.sampling.subsample_seed`

`DataConfig` and `LossConfig` added to the local import.

---

## `scripts/studies/grid_study_dataset_builder.py`

**What changed:** `TrainingConfig` constructor calls updated to nested configs.

---

## `scripts/training/train.py`

**What changed:** `TrainingConfig` constructor calls and attribute accesses updated throughout the TF training entry point.

---

## `docs/refactors/optimizer-config-grouping.md` (new)

Changelog recording the full config tree after all rounds of grouping, the legacy-compat flattening approach, and a per-round summary of which fields moved where.

## `docs/refactors/optimizer-config-grouping-continuation.md` (new)

Agent runbook: the 8-step pattern for moving any future field, a list of remaining candidates, and a table of which files always need touching vs. which never do.
