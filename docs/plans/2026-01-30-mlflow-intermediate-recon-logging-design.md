# MLflow Intermediate Recon Logging (Torch Lightning)

**Date:** 2026-01-30

## Context
We want to visualize intermediate reconstructions during PyTorch Lightning training, using MLflow as the experiment tracker. The request is Torch-only (no TensorFlow), with logging every **5 epochs**, full-resolution images, and both **local patch** samples and **stitched full images**. For patches, log **4 fixed indices** each interval (not all patches). Also log diffraction comparisons (observed vs predicted) using log-scale intensity when available.

## Goals
- Provide MLflow artifact logging of intermediate reconstructions during training.
- Log both real-space (amp/phase) and diffraction (log-intensity) views.
- Include GT and error maps when available.
- Log 4 fixed patch samples and full stitched reconstructions.
- Keep logging opt-in and safe for distributed training.

## Non-goals
- No changes to TF workflows or logging.
- No migration off MLflow or redesign of logger backends.
- No changes to physics/model logic, losses, or inference semantics.

## Decisions
- Implement a **Lightning callback** in `ptycho_torch` and attach it in `_train_with_lightning`.
- Log only when `logger_backend == 'mlflow'` and `recon_log_every_n_epochs` is set.
- Stitched logging is **explicit opt-in** via `--recon-log-stitch`.
- Patch sampling uses **4 fixed indices** chosen once per run.

## Proposed Design

### 1) Callback
Create a callback (e.g., `PtychoReconLoggingCallback`) under `ptycho_torch/workflows/` and wire it in `ptycho_torch/workflows/components.py` when MLflow logging is enabled.

Responsibilities:
- Every **N epochs** (default 5), log recon artifacts to MLflow.
- Ensure `trainer.is_global_zero` (avoid duplicate logs in DDP).
- Use `model.eval()` + `torch.no_grad()` for logging; restore training mode afterward.
- Gracefully skip when required data is missing (labels, metadata, stitching config).

### 2) Patch-level logging (4 fixed indices)
- Select **4 fixed indices** from the validation dataset on first use (or from train if no val).
- For each patch index:
  - Run `model.forward(...)` to obtain predicted diffraction and amp/phase.
  - Observed diffraction comes from batch `images`.
  - If supervised labels exist (`label_amp`, `label_phase`), log GT and error maps.
  - Log diffraction in **log-scale** (e.g., `log1p`) for visibility.

Artifacts per patch:
- `amp_pred`, `phase_pred`
- `amp_gt`, `phase_gt` (if available)
- `amp_error`, `phase_error` (if available)
- `diff_obs_logI`, `diff_pred_logI`, `diff_error_logI`

### 3) Stitched logging (full resolution)
- Run inference over test data to produce predicted complex patches using `model.forward_predict(...)`.
- Stitch via `ptycho.image.stitching.reassemble_patches` or `grid_lines_workflow.stitch_predictions` when metadata indicates grid-lines semantics.
- Stitch config is derived from NPZ `_metadata.additional_parameters`:
  - `N`, `gridsize`, `offset`, `outer_offset_test`, `nimgs_test`
- If `YY_ground_truth` exists, log GT and error maps.

Artifacts (stitched):
- `amp_pred`, `phase_pred`
- `amp_gt`, `phase_gt` (if available)
- `amp_error`, `phase_error` (if available)

### 4) MLflow logging API
- Prefer Lightning `MLFlowLogger` (`trainer.logger`) to get the run.
- Use `mlflow.log_image` when available; otherwise save PNGs to a temp dir and log via `logger.experiment.log_artifact(run_id, path)`.
- Organized artifact paths:
  - `epoch_05/patch_00/amp_pred.png`
  - `epoch_05/patch_00/diff_pred_logI.png`
  - `epoch_05/stitched/phase_gt.png`

## Configuration and CLI
Add new execution-config knobs (default disabled):
- `recon_log_every_n_epochs: Optional[int] = None`
- `recon_log_num_patches: int = 4`
- `recon_log_fixed_indices: Optional[List[int]] = None`
- `recon_log_stitch: bool = False`
- `recon_log_max_stitch_samples: Optional[int] = None` (optional guard)

Wire through:
- `PyTorchExecutionConfig` (ptycho/config/config.py)
- `TorchRunnerConfig` (scripts/studies/grid_lines_torch_runner.py)
- CLI flags in `scripts/training/train.py` and `grid_lines_torch_runner.py` (if used directly)

Stitched logging only activates when `--recon-log-stitch` is set.

## Testing Strategy
- Unit tests:
  - fixed index selection is deterministic
  - missing labels/metadata skip paths safely
  - MLflow logger presence triggers log calls (mock MLflow)
- Integration test:
  - short run with `--logger mlflow --recon-log-every-n-epochs 1 --recon-log-num-patches 4`
  - verify MLflow artifacts exist under `mlruns/`

## Risks / Mitigations
- **Performance overhead** from stitched logging: mitigate with `--recon-log-stitch` opt-in and optional `recon_log_max_stitch_samples`.
- **Missing metadata** for stitching: skip stitched outputs with a clear log line; keep patch logs.
- **DDP duplication**: guard with `trainer.is_global_zero`.

## Open Questions
None outstanding. All parameters (frequency=5, fixed 4 patches, full-res, stitched opt-in) confirmed.

---

## 2026-01-30 Addendum: Recon Logging Hardening

These updates refine the original design to handle real Lightning dataloader shapes and
best-effort stitching in non-grid-lines contexts.

### Patch Logging Updates
- **Scaling parity:** Use `rms_scaling_constant` for both input and output scaling so
  diffraction comparisons mirror training (`compute_loss` uses RMS for both).
- **Multi-channel handling:** When `C > 1` (gridsize>1), log channel 0 for amp/phase/diff
  to avoid invalid `imshow` shapes and keep visuals consistent with baseline assumptions.
- **Dataloader robustness:** Accept `trainer.val_dataloaders` as list/CombinedLoader,
  and fall back to train loader if no val loader is present (with a warning).

### Stitched Logging Updates (Best-effort)
- **Default path:** Use `ptycho.image.stitching.reassemble_patches` with a minimal config
  derived from metadata or training config. This avoids reliance on `params.cfg` state.
- **Position-based stitching:** If global offsets are available (e.g., `global_offsets` or
  `xcoords/ycoords`), use `ptycho_torch.helper.reassemble_patches_position_real` and log
  its output. Otherwise fall back to grid-based stitching.
- **Metadata extraction:** Attempt to load `norm_Y_I`, `YY_ground_truth`, and stitching
  parameters from `test_data_file` via `MetadataManager`. If missing, proceed with safe
  defaults and warn.

### CLI/Config Updates
- **Grid-lines runner:** Add flags for `--torch-logger`, recon logging knobs, and
  `--torch-recon-log-max-stitch-samples`.
- **Training CLI:** Add `--torch-recon-log-max-stitch-samples` to `scripts/training/train.py`
  and plumb it into execution config.
