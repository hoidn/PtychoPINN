# Optimizer Config Grouping Refactor

**Branch:** `refactor_update-configs-1`  
**Status:** In progress

## Overview

Fields in `TrainingConfig` (ptycho/config/config.py) that belong together are being extracted into dedicated nested dataclasses. The goal is a fully self-documenting config tree where every group of related knobs has its own named class. TF-specific classes are prefixed with `TF` to make the phase-out plan visible.

## Current config structure

```
TrainingConfig
├── model: ModelConfig           (unchanged)
├── batch_size / nepochs         (flat — core training loop)
├── positions_provided           (flat)
├── probe_trainable              (flat)
├── intensity_scale_trainable    (flat)
├── output_dir                   (flat)
├── backend                      (flat)
├── data: DataConfig
│   ├── train_data_file
│   ├── test_data_file
│   └── nphotons
├── sampling: SamplingConfig
│   ├── n_groups
│   ├── n_images  (deprecated)
│   ├── n_subsample
│   ├── subsample_seed
│   ├── neighbor_count
│   ├── enable_oversampling
│   ├── neighbor_pool_size
│   └── sequential_sampling
├── loss: LossConfig             (PyTorch-only)
│   ├── torch_loss_mode
│   └── torch_mae_pred_l2_match_target
├── tf_loss: TFLossConfig        (TF-only — will be removed with TF)
│   ├── mae_weight
│   ├── nll_weight
│   ├── realspace_mae_weight
│   └── realspace_weight
├── gradient_clip: GradientClipConfig
│   ├── val
│   └── algorithm
├── optimizer: OptimizerConfig
│   ├── algorithm
│   ├── weight_decay
│   ├── adam: AdamConfig
│   │   ├── beta1
│   │   └── beta2
│   └── sgd: SgdConfig
│       └── momentum
└── scheduler: SchedulerConfig
    ├── kind
    ├── lr_warmup_epochs
    ├── lr_min_ratio
    ├── plateau_factor
    ├── plateau_patience
    ├── plateau_min_lr
    └── plateau_threshold
```

## Legacy compatibility: `dataclass_to_legacy_dict`

`asdict()` recursively serializes all nested dataclasses to dicts. Each nested config has an explicit flattening block in `dataclass_to_legacy_dict` that unpacks it back to the flat legacy keys that `params.cfg` consumers expect. Order matters — the optimizer block must come after the data/sampling blocks.

## `config_bridge.py` — flat-then-nest pattern

Because callers historically pass flat overrides (e.g., `overrides=dict(n_groups=512, nphotons=1e9, train_data_file=Path(...))`), `to_training_config()` uses a two-step approach:

1. Build a `flat` dict with raw scalar values from the PyTorch-side config.
2. Apply `overrides` on top of `flat` (still flat key names).
3. Validate on `flat`.
4. Construct nested config instances from `flat`, build final `kwargs`, call `TFTrainingConfig(**kwargs)`.

This preserves backward compatibility for override call sites while the internal representation uses nested configs.

## Changes made per round

### Round 1: AdamConfig + SgdConfig
`adam_beta1`, `adam_beta2` → `optimizer.adam.beta1/beta2`  
`momentum` → `optimizer.sgd.momentum`

### Round 2: OptimizerConfig wraps Adam + Sgd + weight_decay + algorithm
`optimizer` (string) → `optimizer.algorithm`  
`weight_decay` → `optimizer.weight_decay`  
`adam`, `sgd` sub-configs move inside `optimizer`

### Round 3: Six new groups (current state)
All remaining optimizer/scheduler/loss/data/sampling fields grouped as shown in the tree above.

## Files updated per change

| File | What changed |
|---|---|
| `ptycho/config/config.py` | New classes; TrainingConfig fields; `dataclass_to_legacy_dict` flattening; `validate_training_config` |
| `ptycho_torch/config_bridge.py` | Imports; flat-then-nest kwarg construction |
| `ptycho_torch/model.py` | `_opt`/`_sched` helper vars; all nested getattr accesses |
| `ptycho_torch/workflows/components.py` | Extraction blocks for optimizer, scheduler, gradient_clip, loss |
| `scripts/studies/cdi_natural_patch_benchmark.py` | Same extraction pattern |
| `scripts/studies/grid_lines_torch_runner.py` | Constructor + post-init assignments |
| `ptycho/workflows/components.py` | `config.sampling.*`, `config.data.*` accesses |
| Tests and scripts (bulk) | Constructor calls and attribute accesses updated |

## Files intentionally left unchanged

| File | Class | Reason |
|---|---|---|
| `ptycho_torch/config_params.py` | `TrainingConfig` | PyTorch-side singleton; source for config_bridge; keeps flat fields |
| `ptycho_torch/artifact_schema.py` | field name list | References `config_params.TrainingConfig` |
| `scripts/studies/grid_lines_torch_runner.py` | `GridLinesTorchRunnerConfig` | Script-local runner config |
| `scripts/studies/grid_lines_compare_wrapper.py` | dict / kwargs | Feeds `GridLinesTorchRunnerConfig` |
| `tests/torch/test_model_training.py` | `_build_optimizer` calls | Function parameters, not config field access |
