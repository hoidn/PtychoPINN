diff --git a/specs/ptychodus_api_spec.md b/specs/ptychodus_api_spec.md
index bef1816a..bcba4c57 100644
--- a/specs/ptychodus_api_spec.md
+++ b/specs/ptychodus_api_spec.md
@@ -278,7 +278,12 @@ The PyTorch backend exposes a dedicated execution configuration dataclass (`PyTo
    - `checkpoint_monitor_metric` (str, default `'val_loss'`): Metric for best checkpoint selection. The literal `'val_loss'` is dynamically mapped to `model.val_loss_name` (typically `'poisson_val_loss'` for PINN models) during Lightning configuration, ensuring compatibility with the model's actual metric names. Falls back to `model.train_loss_name` when validation data is unavailable. Exposed via `--checkpoint-monitor`.
    - `checkpoint_mode` (str, default `'min'`): Mode for checkpoint metric optimization. MUST be `'min'` (lower metric is better) or `'max'` (higher metric is better). Exposed via `--checkpoint-mode`.
    - `early_stop_patience` (int, default `100`): Early stopping patience epochs. MUST be > 0. Training stops if monitored metric doesn't improve for this many epochs. Exposed via `--early-stop-patience`.
-   - `logger_backend` (str|None, default `None`): Experiment tracking backend. Pending governance decision (Phase E.B3).
+   - `logger_backend` (str, default `'csv'`): Experiment tracking backend. MUST be one of `['csv', 'tensorboard', 'mlflow', 'none']`. Controls Lightning logger selection for capturing training/validation metrics:
+     - `'csv'`: CSVLogger (default) — zero dependencies, stores metrics as CSV files in `{output_dir}/lightning_logs/`. Recommended for CI/automated workflows.
+     - `'tensorboard'`: TensorBoardLogger — requires tensorboard (auto-installed via TensorFlow dependency), enables rich visualization via `tensorboard --logdir {output_dir}/lightning_logs/`.
+     - `'mlflow'`: MLFlowLogger — requires mlflow package (optional dependency), integrates with MLflow tracking server. Server URI must be configured separately.
+     - `'none'`: Disable logging — metrics from `self.log()` calls are discarded. Use with `--quiet` to suppress all output.
+     When dataclass field is `None`, factory defaults to `'csv'`. Exposed via `--logger` CLI flag. **Note:** MLflow backend currently uses legacy `mlflow.pytorch.autolog()` (ptycho_torch/train.py:75-80); migration to Lightning `MLFlowLogger` tracked as Phase EB3.C4 backlog. **Deprecation:** `--disable_mlflow` flag emits DeprecationWarning directing users to `--logger none` + `--quiet`.
 
 5. **Inference Knobs:**
    - `inference_batch_size` (int|None, default `None`): Override batch size for inference. MUST be > 0 if set. Exposed via `--inference-batch-size`. When `None`, reuses training `batch_size`.
@@ -391,10 +396,11 @@ These flags map to `PyTorchExecutionConfig` fields via factory override preceden
 | `--checkpoint-monitor` | str | `'val_loss'` | `PyTorchExecutionConfig.checkpoint_monitor_metric` | Metric to monitor for checkpoint selection (default: `'val_loss'`). The literal `'val_loss'` is dynamically aliased to `model.val_loss_name` (e.g., `'poisson_val_loss'`) during Lightning configuration. Falls back to `model.train_loss_name` when validation data is unavailable. Common choices: val_loss, train_loss, val_accuracy. |
 | `--checkpoint-mode` | str | `'min'` | `PyTorchExecutionConfig.checkpoint_mode` | Mode for checkpoint metric optimization (default: min). Use 'min' when lower metric values are better (e.g., loss), 'max' when higher values are better (e.g., accuracy). |
 | `--early-stop-patience` | int | `100` | `PyTorchExecutionConfig.early_stop_patience` | Early stopping patience in epochs (default: 100). Training stops if monitored metric doesn't improve for this many consecutive epochs. Set to large value (e.g., 1000) to effectively disable early stopping. |
+| `--logger` | str | `'csv'` | `PyTorchExecutionConfig.logger_backend` | Experiment tracking backend (default: `'csv'`). Choices: `'csv'` (CSVLogger, zero dependencies, CI-friendly), `'tensorboard'` (TensorBoardLogger, requires tensorboard from TensorFlow install), `'mlflow'` (MLFlowLogger, requires mlflow package), `'none'` (disable logging, discards metrics). CSV backend stores metrics in `{output_dir}/lightning_logs/version_N/metrics.csv`. TensorBoard enables visualization via `tensorboard --logdir {output_dir}/lightning_logs/`. MLflow integrates with tracking server (URI configuration required). Use `'none'` with `--quiet` to suppress all output. |
 
 **Deprecated Flags:**
 - `--device` (str): Superseded by `--accelerator`. Using `--device` emits a deprecation warning and maps to `--accelerator`. Will be removed in Phase E post-ADR acceptance. Use `--accelerator` instead.
-- `--disable_mlflow` (flag): MLflow integration not yet implemented; flag is accepted but has no effect. Use `--quiet` to suppress progress output instead.
+- `--disable_mlflow` (flag): **DEPRECATED.** Emits DeprecationWarning directing users to `--logger none` for disabling experiment tracking and `--quiet` for suppressing progress bars. This flag will be removed in a future release. Current behavior: maps to `--logger none` internally.
 
 **Factory Integration:** CLI scripts call `create_training_payload()` with `overrides` dict containing CLI-specified values. The factory applies these overrides after loading defaults and before returning the `TrainingPayload` dataclass (which includes both canonical `TFTrainingConfig` and `PyTorchExecutionConfig`).
 
@@ -402,7 +408,6 @@ These flags map to `PyTorchExecutionConfig` fields via factory override preceden
 
 **Planned Exposure (Phase E.B Backlog):**
 The following `PyTorchExecutionConfig` fields are not yet exposed via CLI but are accessible programmatically:
-- Logger backend: Decision pending governance (Phase E.B3)
 - Advanced trainer knobs: `gradient_clip_val`, `strategy`, `prefetch_factor`, `pin_memory`, `persistent_workers` (deferred pending user demand)
 
 #### 7.2. Inference CLI Execution Flags
@@ -421,7 +426,7 @@ The following `PyTorchExecutionConfig` fields are not yet exposed via CLI but ar
 
 **Validation Evidence:** Phase C4.D manual CLI smoke test with gridsize=2 confirmed all execution flags operate correctly. See `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/manual_cli_smoke_gs2.log`.
 
-**Note:** For programmatic access to execution-only parameters not yet exposed via CLI (checkpoint knobs, scheduler, logger backend), instantiate `PyTorchExecutionConfig` directly and pass to factory functions. See §4.9 for complete field reference and validation rules.
+**Note:** For programmatic access to execution-only parameters not yet exposed via CLI (advanced trainer knobs), instantiate `PyTorchExecutionConfig` directly and pass to factory functions. See §4.9 for complete field reference and validation rules.
 
 ### 8. Usage Guidelines for Developers
 
