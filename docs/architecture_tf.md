# TensorFlow Architecture

The TensorFlow implementation is the primary codepath in this branch.

## Training

- Entrypoints:
  - `ptycho/train.py` (legacy params.cfg style)
  - `scripts/training/train.py` (dataclass config + params bridge)
- Model construction in `ptycho/model.py`.
- Losses and physics constraints in `ptycho/losses.py` and `ptycho/physics.py`.
- Training orchestration in `ptycho/train_pinn.py` and `ptycho/train_supervised.py`.

## Inference

- `ptycho/inference.py` provides a programmatic path.
- `scripts/inference/inference.py` is the CLI wrapper that loads a saved model and runs
  reconstruction + visualization.

## Persistence

- `ptycho/model_manager.py` saves models to `<output>/wts.h5_<model_name>` directories,
  alongside `custom_objects.dill` and `params.dill`.
