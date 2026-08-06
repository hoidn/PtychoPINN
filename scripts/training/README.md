# PtychoPINN Training CLI

`ptycho_train` is the backend-agnostic entry point for training from an existing
standalone NPZ. It validates the supplied data, resolves the public training
configuration, groups the data once, selects TensorFlow or PyTorch, and saves
the resulting model artifacts.

For simulation plus training, reconstruction, and evaluation, use
`ptycho_synthetic`. For Torch-only controls such as the `ci` profile and
`rect_s1s2_init`, use `python -m ptycho_torch.train`. The interfaces are
separate; native Torch flags should not be copied to `ptycho_train`.

## Input data

The ordinary file route expects a standalone `.npz` accepted by
`RawData.from_file()`, including:

- `diff3d` with shape `(patterns, N, N)`;
- `xcoords` and `ycoords`;
- complex `probeGuess`;
- `objectGuess` and the remaining acquisition fields required by the shared
  data contract.

`N` and the configured grid size must agree with the data. Legacy datasets may
contain placeholder or repeated `scan_index` values; use coordinates for
alignment and provenance checks. See
[the data contracts](../../specs/data_contracts.md) for the complete schema.

This entry point loads the standalone NPZ through `RawData`, groups it in
memory, and uses ordinary DataLoaders. It does not select the TensorDict mmap
route; that route belongs to `ptycho_torch.train_lightning_only`.

## Configuration shape

Public training configuration is a nested Pydantic model. YAML fields belong
to their owning sections:

```yaml
model:
  N: 64
  gridsize: 1
  architecture: cnn

data:
  train_data_file: datasets/train.npz
  test_data_file: datasets/test.npz

sampling:
  n_groups: 512
  n_subsample: 2000
  subsample_seed: 42

optimizer:
  algorithm: adam
  weight_decay: 0.0

scheduler:
  kind: ReduceLROnPlateau
  plateau_factor: 0.5
  plateau_patience: 2

backend: pytorch
batch_size: 16
nepochs: 50
output_dir: outputs/my_run
```

File values are deep-merged with explicitly supplied CLI values. Direct
`TrainingConfig` fields use plain flags; nested fields use dotted flags:

```bash
ptycho_train --config configs/my_config.yaml

ptycho_train --config configs/my_config.yaml \
  --data.train_data_file datasets/train.npz \
  --backend pytorch \
  --output_dir outputs/my_run
```

Use `ptycho_train --help` for the generated public flags. Unknown or misplaced
fields fail validation. `sampling.n_images` remains a deprecated alias for
`sampling.n_groups`; new configurations should use the canonical field.

Current `refactor` limitation: the generated parser does not yet decode
numeric or Boolean CLI values before strict validation. Put those types in
YAML; use CLI overrides for paths and literal strings until the decoder is
fixed.

## Sampling

- `sampling.n_subsample` selects raw rows before grouping.
- `sampling.n_groups` selects grouped model samples.
- `sampling.subsample_seed` makes raw-row selection reproducible.
- `sampling.sequential_sampling=true` uses the first grouping anchors within
  the already selected raw-row pool; it does not replace random raw-row
  subsampling.
- `sampling.enable_oversampling=true` explicitly enables combination-based
  oversampling for grid size greater than one; configure its candidate pool
  with `sampling.neighbor_pool_size`.

With `gridsize=1`, one group contains one frame. With `gridsize>1`, one group
contains `gridsize²` neighboring frames. `n_groups` always counts groups.

## PyTorch runtime and optimization overrides

When `backend=pytorch`, unified runtime flags use the `--torch-*` prefix. For
example:

```bash
ptycho_train \
  --data.train_data_file datasets/train.npz \
  --backend pytorch \
  --scheduler.kind ReduceLROnPlateau \
  --torch-learning-rate 0.0004 \
  --torch-accelerator auto \
  --torch-logger csv \
  --output_dir outputs/my_run
```

The public scheduler section is part of the nested configuration. Explicit
`--torch-learning-rate`, `--torch-scheduler`, and related `--torch-*` optimizer
flags form a separate Torch factory patch. See the
[PyTorch workflow guide](../../docs/workflows/pytorch.md) for that boundary.

## Outputs and logging

TensorFlow and PyTorch persist backend-specific bundles beneath `output_dir`.
PyTorch training writes its portable bundle as `wts.h5.zip` and, when enabled,
Lightning checkpoints and logger output beneath the run directory. TensorFlow
uses its model-manager archive and reconstruction outputs.

The wrapper writes `train_debug.log` in the process working directory. Console
verbosity is controlled by `--quiet`, `--verbose`, or `--console-level`.

After training, run `ptycho_inference` on the saved bundle or use the supported
evaluation workflow documented in the
[commands reference](../../docs/COMMANDS_REFERENCE.md).
