diff --git a/docs/workflows/pytorch.md b/docs/workflows/pytorch.md
index c2f54b18..cd291f6c 100644
--- a/docs/workflows/pytorch.md
+++ b/docs/workflows/pytorch.md
@@ -323,10 +323,16 @@ The following execution config flags are available in `ptycho_torch/train.py`:
 | `--quiet` | flag | `False` | Suppress progress bars and reduce console logging |
 | `--enable-checkpointing` / `--disable-checkpointing` | bool | `True` | Enable automatic model checkpointing (default: enabled). Use `--disable-checkpointing` to turn off checkpoint saving. |
 | `--checkpoint-save-top-k` | int | `1` | Number of best checkpoints to keep (1 = save only best, -1 = save all, 0 = disable) |
-| `--checkpoint-monitor` | str | `'val_loss'` | Metric to monitor for checkpoint selection (e.g., val_loss, train_loss). Falls back to train_loss when validation data unavailable. |
+| `--checkpoint-monitor` | str | `'val_loss'` | Metric to monitor for checkpoint selection (default: `'val_loss'`). The literal `'val_loss'` is dynamically aliased to `model.val_loss_name` (e.g., `'poisson_val_loss'` for PINN models) during Lightning configuration. Falls back to `model.train_loss_name` when validation data is unavailable. |
 | `--checkpoint-mode` | str | `'min'` | Checkpoint metric optimization mode ('min' for loss metrics, 'max' for accuracy metrics) |
 | `--early-stop-patience` | int | `100` | Early stopping patience in epochs. Training stops if monitored metric doesn't improve for this many consecutive epochs. |
 
+**Monitor Metric Aliasing:**
+The checkpoint monitor metric (`--checkpoint-monitor`) uses dynamic aliasing to handle backend-specific metric naming conventions. When you specify `--checkpoint-monitor val_loss` (the default), the training workflow automatically resolves this to the model's actual validation loss metric name (e.g., `poisson_val_loss` for PINN models). This aliasing ensures compatibility across different loss formulations without requiring users to know internal metric names. When validation data is unavailable, the system automatically falls back to the corresponding training metric (`model.train_loss_name`).
+
+**Gradient Accumulation Considerations:**
+Gradient accumulation (`--accumulate-grad-batches`) simulates larger effective batch sizes by accumulating gradients over multiple forward/backward passes before updating model weights. The effective batch size equals `batch_size × accumulate_grad_batches`. While this technique improves memory efficiency (allowing larger effective batches on memory-constrained hardware), values >1 may affect training dynamics, convergence rates, and Poisson loss stability. For PINN models with physics-informed losses, start with the default (`1`) and increase conservatively only when memory constraints require it. Monitor training curves when changing accumulation settings, as the optimizer sees fewer but larger gradient updates per epoch.
+
 **Deprecated Flags:**
 - `--device`: Superseded by `--accelerator`. Using `--device` will emit a deprecation warning and map to `--accelerator` automatically. Remove from scripts; this flag will be dropped in a future release.
 - `--disable_mlflow`: MLflow integration is not yet implemented; this flag is accepted but has no effect. Use `--quiet` to suppress progress output instead.
diff --git a/specs/ptychodus_api_spec.md b/specs/ptychodus_api_spec.md
index 91c63f2a..bef1816a 100644
--- a/specs/ptychodus_api_spec.md
+++ b/specs/ptychodus_api_spec.md
@@ -275,7 +275,7 @@ The PyTorch backend exposes a dedicated execution configuration dataclass (`PyTo
    - `enable_progress_bar` (bool, default `False`): Show training progress. Derived from `--quiet` flag inversion.
    - `enable_checkpointing` (bool, default `True`): Enable Lightning automatic checkpointing during training. Exposed via `--enable-checkpointing` / `--disable-checkpointing`.
    - `checkpoint_save_top_k` (int, default `1`): Number of best checkpoints to retain. MUST be ≥ 0. Set to -1 to save all checkpoints, 0 to disable saving. Exposed via `--checkpoint-save-top-k`.
-   - `checkpoint_monitor_metric` (str, default `'val_loss'`): Metric for best checkpoint selection. Uses validation loss by default; falls back to training loss when validation data unavailable. Exposed via `--checkpoint-monitor`.
+   - `checkpoint_monitor_metric` (str, default `'val_loss'`): Metric for best checkpoint selection. The literal `'val_loss'` is dynamically mapped to `model.val_loss_name` (typically `'poisson_val_loss'` for PINN models) during Lightning configuration, ensuring compatibility with the model's actual metric names. Falls back to `model.train_loss_name` when validation data is unavailable. Exposed via `--checkpoint-monitor`.
    - `checkpoint_mode` (str, default `'min'`): Mode for checkpoint metric optimization. MUST be `'min'` (lower metric is better) or `'max'` (higher metric is better). Exposed via `--checkpoint-mode`.
    - `early_stop_patience` (int, default `100`): Early stopping patience epochs. MUST be > 0. Training stops if monitored metric doesn't improve for this many epochs. Exposed via `--early-stop-patience`.
    - `logger_backend` (str|None, default `None`): Experiment tracking backend. Pending governance decision (Phase E.B3).
@@ -388,7 +388,7 @@ These flags map to `PyTorchExecutionConfig` fields via factory override preceden
 | `--quiet` | flag | `False` | `PyTorchExecutionConfig.enable_progress_bar` | Suppress progress bars and reduce console logging. Inverted to populate `enable_progress_bar` (`--quiet` → `False`). |
 | `--enable-checkpointing` / `--disable-checkpointing` | bool | `True` | `PyTorchExecutionConfig.enable_checkpointing` | Enable automatic model checkpointing during training (default: enabled). Checkpoints are saved based on monitored metric performance. Use `--disable-checkpointing` to turn off. |
 | `--checkpoint-save-top-k` | int | `1` | `PyTorchExecutionConfig.checkpoint_save_top_k` | Number of best checkpoints to keep (default: 1). Set to -1 to save all checkpoints, 0 to disable saving. Best checkpoints are determined by `--checkpoint-monitor` metric. |
-| `--checkpoint-monitor` | str | `'val_loss'` | `PyTorchExecutionConfig.checkpoint_monitor_metric` | Metric to monitor for checkpoint selection (default: val_loss). Falls back to train_loss when validation data unavailable. Common choices: val_loss, train_loss, val_accuracy. |
+| `--checkpoint-monitor` | str | `'val_loss'` | `PyTorchExecutionConfig.checkpoint_monitor_metric` | Metric to monitor for checkpoint selection (default: `'val_loss'`). The literal `'val_loss'` is dynamically aliased to `model.val_loss_name` (e.g., `'poisson_val_loss'`) during Lightning configuration. Falls back to `model.train_loss_name` when validation data is unavailable. Common choices: val_loss, train_loss, val_accuracy. |
 | `--checkpoint-mode` | str | `'min'` | `PyTorchExecutionConfig.checkpoint_mode` | Mode for checkpoint metric optimization (default: min). Use 'min' when lower metric values are better (e.g., loss), 'max' when higher values are better (e.g., accuracy). |
 | `--early-stop-patience` | int | `100` | `PyTorchExecutionConfig.early_stop_patience` | Early stopping patience in epochs (default: 100). Training stops if monitored metric doesn't improve for this many consecutive epochs. Set to large value (e.g., 1000) to effectively disable early stopping. |
 
@@ -402,8 +402,8 @@ These flags map to `PyTorchExecutionConfig` fields via factory override preceden
 
 **Planned Exposure (Phase E.B Backlog):**
 The following `PyTorchExecutionConfig` fields are not yet exposed via CLI but are accessible programmatically:
-- Scheduler / accumulation: `--scheduler`, `--accumulate-grad-batches` (Phase E.B2)
 - Logger backend: Decision pending governance (Phase E.B3)
+- Advanced trainer knobs: `gradient_clip_val`, `strategy`, `prefetch_factor`, `pin_memory`, `persistent_workers` (deferred pending user demand)
 
 #### 7.2. Inference CLI Execution Flags
 
