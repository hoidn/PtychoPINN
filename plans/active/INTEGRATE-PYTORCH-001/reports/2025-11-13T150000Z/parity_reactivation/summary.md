### Turn Summary
Closed the execution-config Do Now: commit 9daa00b7 adds the manual-optimization guard and supervised-data detection in `_train_with_lightning`, refreshed docs/tests, and landed a clean PINN-mode training CLI log plus artifacts under the active hub.
Documented the new evidence in the artifact inventory + red blockers (supervised data still missing) and noted that the inference half of the PINN smoke needs to be rerun against `train_outputs/wts.h5.zip`.
Next: add spec-mandated defaults to `ptycho_torch/config_params.py`, update `config_factory`/`config_bridge` so the TensorFlow bridge no longer relies on ad-hoc overrides, extend `tests/torch/test_config_bridge.py`, rerun the parity selector, and capture a fresh PyTorch inference CLI log with the new bundle.
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{green/pytest_backend_selector_cli.log,cli/pytorch_cli_smoke_training/train_clean.log}

### Turn Summary
Fixed supervised model_type→MAE loss mapping to prevent loss_name AttributeError in PyTorch Lightning module; added regression test coverage and confirmed fix with GREEN pytest evidence (3 PASSED).
Supervised CLI smoke revealed two downstream blockers: manual optimization incompatibility with accumulate_grad_batches flag, and supervised mode requiring ground-truth labels (label_amp/label_phase) absent from experimental fly001 dataset.
Next: either switch smoke test to unsupervised PINN mode for execution-config validation, or defer supervised CLI smoke until labeled synthetic data is available, then refresh hub documentation.
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/ (green/pytest_backend_selector_cli.log ✅, red/blocked_*.md, analysis/artifact_inventory.txt)
