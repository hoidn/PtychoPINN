### Turn Summary
Fixed supervised model_type→MAE loss mapping to prevent loss_name AttributeError in PyTorch Lightning module; added regression test coverage and confirmed fix with GREEN pytest evidence (3 PASSED).
Supervised CLI smoke revealed two downstream blockers: manual optimization incompatibility with accumulate_grad_batches flag, and supervised mode requiring ground-truth labels (label_amp/label_phase) absent from experimental fly001 dataset.
Next: either switch smoke test to unsupervised PINN mode for execution-config validation, or defer supervised CLI smoke until labeled synthetic data is available, then refresh hub documentation.
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/ (green/pytest_backend_selector_cli.log ✅, red/blocked_*.md, analysis/artifact_inventory.txt)
