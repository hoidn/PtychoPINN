# Blocker — PyTorch training CLI smoke (`loss_name` AttributeError)

- **Command:**
  ```bash
  python scripts/training/train.py --train_data_file tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz --backend pytorch --n_images 100 --nepochs 2 --model_type supervised --output_dir plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/cli/pytorch_cli_smoke_training/train_outputs --torch-accelerator cpu --torch-num-workers 0 --torch-learning-rate 1e-3 --torch-scheduler Default --torch-logger csv
  ```
- **Result:** CLI accepts new execution-config flags and constructs the PyTorch workflow but aborts before Lightning starts training. `train_debug.log:30621-30623` records: `An error occurred during execution: 'PtychoPINN_Lightning' object has no attribute 'loss_name'`.
- **Impact:** PyTorch training smoke cannot complete, so we have no fresh bundle/PNG artifacts for the execution-config increment. Backend selector work is blocked on resolving the missing `loss_name` attribute inside the Lightning module.
- **Initial hypothesis:** The PyTorch model defaults to `loss_function='Poisson'` even when `model_type='supervised'`, so none of the `if/elif` branches in `PtychoPINN_Lightning.__init__` assign `self.loss_name`, causing the AttributeError as soon as `self.log(self.loss_name, ...)` runs.
- **Next action:** Teach the config bridge/factory to set `loss_function='MAE'` (or another supported option) whenever the canonical config requests `model_type='supervised'`, add pytest coverage to lock the behavior, and rerun the CLI smoke to generate a GREEN log.
