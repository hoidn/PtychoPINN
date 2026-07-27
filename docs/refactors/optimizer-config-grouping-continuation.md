# Continuation Guide: TrainingConfig Grouping Refactor

This file gives an agent (or human) enough context to continue the refactor described in `optimizer-config-grouping.md`.

## Current state

`TrainingConfig` is now fully grouped. All remaining flat fields are intentionally flat:
- `model`, `batch_size`, `nepochs` — structural / core scalars
- `positions_provided`, `probe_trainable`, `intensity_scale_trainable` — model behaviour flags
- `output_dir`, `backend` — I/O and execution selectors

If a future round moves any of these, the same 8-file pattern applies.

## The pattern for any future move

When moving a field `foo` from `TrainingConfig` into a new or existing nested config `XyzConfig`:

1. **`config.py`** — Add `foo` to `XyzConfig`. Add `XyzConfig` to `__all__`. Remove `foo` from `TrainingConfig` flat fields. Add a flattening block in `dataclass_to_legacy_dict` that populates `d['foo']` from `d.pop('xyz')['foo']`.
2. **`config_bridge.py`** — Add `foo` to the `flat` dict; use it when constructing `TFXyzConfig(...)`.
3. **`model.py`** — Update any `getattr(self.training_config, 'foo', default)` to go through the appropriate `_xyz` helper variable.
4. **`ptycho_torch/workflows/components.py`** — Update the appropriate extraction block.
5. **`scripts/studies/cdi_natural_patch_benchmark.py`** — Same extraction block.
6. **`scripts/studies/grid_lines_torch_runner.py`** — Update constructor kwargs and post-init assignments.
7. **`ptycho/workflows/components.py`** — Update any `config.foo` access.
8. **Tests / scripts** — Update `TrainingConfig(foo=...)` constructor calls and `config.foo` accesses.

## `TF`-prefixed classes

Any config class that holds fields **used only by TensorFlow** must be named `TF*` (e.g., `TFLossConfig`). This makes the phase-out boundary explicit. When TF is removed:
1. Delete the `TF*` class.
2. Remove `tf_loss: TFLossConfig` (or equivalent) from `TrainingConfig`.
3. Remove the corresponding flattening block from `dataclass_to_legacy_dict`.
4. Remove the `'tf_loss': TFTFLossConfig(...)` kwarg from `config_bridge.to_training_config()`.

Currently `TFLossConfig` is the only `TF`-prefixed class.

## Key files to touch for every `TrainingConfig` field move

| File | What to update |
|---|---|
| `ptycho/config/config.py` | Class def + `__all__` + remove flat field + `dataclass_to_legacy_dict` block + `validate_training_config` |
| `ptycho_torch/config_bridge.py` | `flat` dict + nested constructor call in `to_training_config()` |
| `ptycho_torch/model.py` | `configure_optimizers()` helper vars and nested getattr |
| `ptycho_torch/workflows/components.py` | Extraction block after `if execution_config is not None` guards |
| `scripts/studies/cdi_natural_patch_benchmark.py` | Same extraction block |
| `scripts/studies/grid_lines_torch_runner.py` | `setup_training_config()` constructor + assignments |
| `ptycho/workflows/components.py` | Any `config.*` direct accesses |
| Tests / scripts (bulk) | Constructor calls and attribute accesses |

## Files that do NOT need changes (ever)

- `ptycho_torch/config_params.py` — PyTorch-side `TrainingConfig`; always flat; source for `config_bridge`
- `ptycho_torch/artifact_schema.py` — references `config_params.TrainingConfig` flat field names
- `scripts/studies/grid_lines_torch_runner.py` `GridLinesTorchRunnerConfig` — its own flat dataclass, feeds `setup_training_config()`
- `scripts/studies/grid_lines_compare_wrapper.py` — feeds `GridLinesTorchRunnerConfig`
- `tests/torch/test_model_training.py` — calls `_build_optimizer()` directly

## Critical: `dataclass_to_legacy_dict` and `params.cfg`

`asdict()` recursively serializes all nested configs to plain dicts. Without the flattening blocks, `params.cfg` would receive e.g. `{'data': {'train_data_file': ..., 'nphotons': 1e9}}` instead of the flat keys TF code expects.

The current flattening order in `dataclass_to_legacy_dict`:
1. `data` → `train_data_file`, `test_data_file`, `nphotons`
2. `sampling` → all sampling fields
3. `tf_loss` → `mae_weight`, `nll_weight`, `realspace_mae_weight`, `realspace_weight`
4. `loss` → `torch_loss_mode`, `torch_mae_pred_l2_match_target`
5. `gradient_clip` → `gradient_clip_val`, `gradient_clip_algorithm`
6. `scheduler` → `scheduler` (string), `lr_*`, `plateau_*`
7. `optimizer` → `optimizer` (string), `weight_decay`, `adam_beta1`, `adam_beta2`, `momentum`

Then `KEY_MAPPINGS` renames `train_data_file` → `train_data_file_path` and `test_data_file` → `test_data_file_path`.

Before moving any field that feeds a legacy TF module through `params.cfg`, check `docs/findings.md` and the `KEY_MAPPINGS` dict.

## `config_bridge.py` flat-then-nest contract

The `flat` dict in `to_training_config()` accepts any key that was historically a flat field of `TFTrainingConfig`. Callers may pass these as `overrides`:

```python
overrides=dict(n_groups=512, nphotons=1e9, train_data_file=Path('train.npz'))
```

These flat keys are applied to `flat` before building nested configs. New fields added to nested configs must also be added to `flat` with sensible defaults so the validation step can check them.
