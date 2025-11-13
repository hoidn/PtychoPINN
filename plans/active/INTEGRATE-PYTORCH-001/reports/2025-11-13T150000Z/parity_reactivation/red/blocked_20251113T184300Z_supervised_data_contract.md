# Blocker — Supervised mode requires label_amp/label_phase in data batches

- **Progress:** The supervised→MAE loss mapping fix (commit TBD) successfully resolved the `AttributeError: 'PtychoPINN_Lightning' object has no attribute 'loss_name'` blocker.
- **New blocker:** `KeyError: 'label_amp'` at `ptycho_torch/model.py:1179` during `training_step`.
- **Root cause:** The Lightning model's `compute_loss` method for supervised mode expects batch dictionaries with keys `['label_amp', 'label_phase']`, but the data loader populates unsupervised keys. The supervised model requires ground-truth reconstructions in the training data (per DATA-001), which the current fly001 dataset doesn't include.
- **Impact:** Supervised training CLI smoke cannot complete with experimental data lacking ground-truth labels. The unsupervised (PINN) path would succeed with this data.
- **Next action:** Either (a) switch smoke test to unsupervised mode (`--model_type pinn`) to validate the execution-config flags, or (b) defer supervised CLI smoke until a labeled synthetic dataset is prepared.
