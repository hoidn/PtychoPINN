# Blocker — PyTorch manual optimization incompatible with accumulate_grad_batches

- **Command:**
  ```bash
  python scripts/training/train.py ... --torch-accumulate-grad-batches 2 ...
  ```
- **Result:** Lightning raises `MisconfigurationException: Automatic gradient accumulation is not supported for manual optimization. Remove Trainer(accumulate_grad_batches=2) or switch to automatic optimization.`
- **Root cause:** `PtychoPINN_Lightning.__init__` sets `self.automatic_optimization = False` (ptycho_torch/model.py:1030), enabling manual gradient management for DDP learning rate scaling. Lightning's automatic gradient accumulation requires `automatic_optimization=True`.
- **Impact:** The `--torch-accumulate-grad-batches` CLI flag cannot be used with the current PyTorch model. The supervised→MAE loss mapping works correctly; this is an independent execution-config compatibility issue.
- **Workaround:** Remove `--torch-accumulate-grad-batches` from CLI invocations. The model's manual optimization already handles gradient accumulation internally via `self.accum_steps` from TrainingConfig.
- **Next action:** Rerun the smoke test without `--torch-accumulate-grad-batches` to verify the supervised loss fix in isolation.
