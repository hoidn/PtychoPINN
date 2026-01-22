# Data Management Guide

- Large datasets live under `datasets/`, `outputs/`, or `training_outputs/`.
- Avoid committing generated data and model artifacts to version control.
- Prefer `.npz` for datasets and the model manager output directories for trained models.
