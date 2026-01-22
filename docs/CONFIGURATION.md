# Configuration

## Legacy Config

`ptycho/params.py` defines `params.cfg`, a global dictionary used by many modules.

Key fields:
- `N`, `gridsize`, `offset`
- `batch_size`, `nepochs`
- `nphotons`, `mae_weight`, `nll_weight`

## Dataclass Config

`ptycho/config/config.py` provides `ModelConfig`, `TrainingConfig`, and `InferenceConfig`.
The CLI in `scripts/training/train.py` loads YAML and then calls `update_legacy_dict()`
to sync values into `params.cfg`.

## YAML Example

```
model:
  N: 64
  gridsize: 2
  model_type: pinn
train_data_file: data/train.npz
batch_size: 16
nepochs: 50
output_dir: training_outputs/run_01
```
